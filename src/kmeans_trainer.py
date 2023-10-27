import argparse
import os
import shutil
import time
import yaml
import copy

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .dataset import get_subset, get_dataset
from .model import get_model
from .model.tools import count_parameters, safe_model_state_dict
from .optimizer import get_optimizer
from .scheduler import get_scheduler
from .utils import (
    use_seed,
    coerce_to_path_and_check_exist,
    coerce_to_path_and_create_dir,
)
from .utils.image import convert_to_img, save_gif
from .utils.logger import get_logger, print_info, print_warning
from .utils.metrics import AverageTensorMeter, AverageMeter, Metrics, Scores
from .utils.path import CONFIGS_PATH, RUNS_PATH
from .utils.plot import plot_lines, plot_bar


PRINT_TRAIN_STAT_FMT = "Epoch [{}/{}], Iter [{}/{}], train_metrics: {}".format
PRINT_VAL_STAT_FMT = "Epoch [{}/{}], Iter [{}/{}], val_metrics: {}".format
PRINT_CHECK_CLUSTERS_FMT = (
    "Epoch [{}/{}], Iter [{}/{}]: Reassigned clusters {} from cluster {}".format
)
PRINT_LR_UPD_FMT = "Epoch [{}/{}], Iter [{}/{}], LR update: lr = {}".format

TRAIN_METRICS_FILE = "train_metrics.tsv"
VAL_METRICS_FILE = "val_metrics.tsv"
VAL_SCORES_FILE = "val_scores.tsv"
FINAL_SCORES_FILE = "final_scores.tsv"
MODEL_FILE = "model.pkl"

N_TRANSFORMATION_PREDICTIONS = 4
N_CLUSTER_SAMPLES = 5
MAX_GIF_SIZE = 64
VIZ_HEIGHT = 300
VIZ_WIDTH = 500
VIZ_MAX_IMG_SIZE = 64


class Trainer:
    """Pipeline to train a NN model using a certain dataset, both specified by an YML config."""

    @use_seed()
    def __init__(
        self, config_path, run_dir, subset=None, parent_model=None, recluster=False
    ):
        self.config_path = coerce_to_path_and_check_exist(config_path)
        self.run_dir = coerce_to_path_and_create_dir(run_dir)
        self.logger = get_logger(self.run_dir, name="trainer")
        self.print_and_log_info(
            "Trainer initialisation: run directory is {}".format(run_dir)
        )

        shutil.copy(self.config_path, self.run_dir)
        self.print_and_log_info(
            "Config {} copied to run directory".format(self.config_path)
        )

        with open(self.config_path) as fp:
            cfg = yaml.load(fp, Loader=yaml.FullLoader)

        if torch.cuda.is_available():
            type_device = "cuda"
            nb_device = torch.cuda.device_count()
        else:
            type_device = "cpu"
            nb_device = None
        self.device = torch.device(type_device)
        self.print_and_log_info(
            "Using {} device, nb_device is {}".format(type_device, nb_device)
        )

        # Datasets and dataloaders
        self.dataset_kwargs = cfg["dataset"]
        self.dataset_name = self.dataset_kwargs.pop("name")
        if not isinstance(subset, type(None)):
            train_dataset = get_subset(self.dataset_name)(
                "train", subset, **self.dataset_kwargs
            )
        else:
            train_dataset = get_dataset(self.dataset_name)(
                "train", None, **self.dataset_kwargs
            )
        val_dataset = get_dataset(self.dataset_name)("val", None, **self.dataset_kwargs)

        self.n_classes = train_dataset.n_classes
        self.is_val_empty = len(val_dataset) == 0
        self.print_and_log_info(
            "Dataset {} instantiated with {}".format(
                self.dataset_name, self.dataset_kwargs
            )
        )
        self.print_and_log_info(
            "Found {} classes, {} train samples, {} val samples".format(
                self.n_classes, len(train_dataset), len(val_dataset)
            )
        )

        self.img_size = train_dataset.img_size
        self.batch_size = (
            cfg["training"]["batch_size"]
            if cfg["training"]["batch_size"] < len(train_dataset)
            else len(train_dataset)
        )
        self.n_workers = cfg["training"].get("n_workers", 4)
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            num_workers=self.n_workers,
            shuffle=True,
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, num_workers=self.n_workers
        )
        self.print_and_log_info(
            "Dataloaders instantiated with batch_size={} and n_workers={}".format(
                self.batch_size, self.n_workers
            )
        )

        self.n_batches = len(self.train_loader)
        self.n_iterations, self.n_epoches = cfg["training"].get("n_iterations"), cfg[
            "training"
        ].get("n_epoches")
        assert not (self.n_iterations is not None and self.n_epoches is not None)
        if self.n_iterations is not None:
            self.n_epoches = max(self.n_iterations // self.n_batches, 1)
        else:
            self.n_iterations = self.n_epoches * len(self.train_loader)

        # Model
        self.model_kwargs = cfg["model"]
        self.model_name = self.model_kwargs.pop("name")
        self.is_gmm = "gmm" in self.model_name
        self.model = get_model(self.model_name)(
            self.train_loader.dataset, **self.model_kwargs
        ).to(self.device)
        self.print_and_log_info(
            "Using model {} with kwargs {}".format(self.model_name, self.model_kwargs)
        )
        self.print_and_log_info(
            "Number of trainable parameters: {}".format(
                f"{count_parameters(self.model):,}"
            )
        )
        self.n_prototypes = self.model.n_prototypes

        # Optimizer
        self.opt_params = cfg["training"]["optimizer"] or {}
        self.sprite_opt_params = cfg["training"].get("sprite_optimizer", {})
        self.optimizer_name = self.opt_params.pop("name")
        self.cluster_kwargs = self.opt_params.pop("cluster", {})
        self.tsf_kwargs = self.opt_params.pop("transformer", {})
        self.sprite_optimizer_name = self.sprite_opt_params.pop("name", None)
        if self.sprite_optimizer_name in ["SGD", "sgd"]:
            self.sprite_optimizer = get_optimizer(self.sprite_optimizer_name)(
                [dict(params=self.model.cluster_parameters(), **self.cluster_kwargs)],
                **self.sprite_opt_params,
            )
            self.optimizer = get_optimizer(self.optimizer_name)(
                [dict(params=self.model.transformer_parameters(), **self.tsf_kwargs)],
                **self.opt_params,
            )
            self.model.set_optimizer(self.optimizer, self.sprite_optimizer)
        else:
            self.optimizer = get_optimizer(self.optimizer_name)(
                [dict(params=self.model.cluster_parameters(), **self.cluster_kwargs)]
                + [dict(params=self.model.transformer_parameters(), **self.tsf_kwargs)],
                **self.opt_params,
            )
            self.model.set_optimizer(self.optimizer)
        self.print_and_log_info(
            "Using optimizer {} with kwargs {}".format(
                self.optimizer_name, self.opt_params
            )
        )
        self.print_and_log_info("cluster kwargs {}".format(self.cluster_kwargs))
        self.print_and_log_info("transformer kwargs {}".format(self.tsf_kwargs))

        # Scheduler
        self.scheduler_params = cfg["training"].get("scheduler", {}) or {}
        self.scheduler_name = self.scheduler_params.pop("name", None)
        self.scheduler_update_range = self.scheduler_params.pop("update_range", "epoch")
        assert self.scheduler_update_range in ["epoch", "batch"]
        if self.scheduler_name == "multi_step" and isinstance(
            self.scheduler_params["milestones"][0], float
        ):
            n_tot = (
                self.n_epoches
                if self.scheduler_update_range == "epoch"
                else self.n_iterations
            )
            self.scheduler_params["milestones"] = [
                round(m * n_tot) for m in self.scheduler_params["milestones"]
            ]
        self.scheduler = get_scheduler(self.scheduler_name)(
            self.optimizer, **self.scheduler_params
        )
        self.cur_lr = self.scheduler.get_last_lr()[0]
        self.print_and_log_info(
            "Using scheduler {} with parameters {}".format(
                self.scheduler_name, self.scheduler_params
            )
        )

        # Pretrained / Resume
        checkpoint_path = cfg["training"].get("pretrained")
        checkpoint_path_resume = cfg["training"].get("resume")
        assert not (checkpoint_path is not None and checkpoint_path_resume is not None)
        if checkpoint_path is not None:
            self.load_from_tag(checkpoint_path)
        elif checkpoint_path_resume is not None:
            if recluster:
                self.load_from_tag(self.run_dir, resume=True)
            else:
                self.load_from_tag(checkpoint_path_resume, resume=True)
        else:
            self.start_epoch, self.start_batch = 1, 1

        self.parent_model = None
        if not isinstance(subset, type(None)):
            assert parent_model
            self.parent_model = parent_model
            if checkpoint_path_resume:
                self.load_from_tag(self.run_dir, resume=True)
            else:
                self.load_from_tag(parent_model)
            ss_id = run_dir.split("_")[-1][-1]
            if not checkpoint_path_resume:
                if ss_id == "0":
                    self.model.prototypes[1].data.copy_(self.model.prototypes[0])
                elif ss_id == "1":
                    self.model.prototypes[0].data.copy_(self.model.prototypes[1])
                else:
                    ValueError("Invalid subset id")

        # Train metrics & check_cluster interval
        metric_names = ["time/img", "loss"]
        metric_names += [f"prop_clus{i}" for i in range(self.n_prototypes)]
        metric_names += [f"loss_clus{i}" for i in range(self.n_prototypes)]
        train_iter_interval = cfg["training"]["train_stat_interval"]
        self.train_stat_interval = train_iter_interval
        self.train_metrics = Metrics(*metric_names)
        self.train_metrics_path = self.run_dir / TRAIN_METRICS_FILE
        with open(self.train_metrics_path, mode="w") as f:
            f.write(
                "iteration\tepoch\tbatch\t" + "\t".join(self.train_metrics.names) + "\n"
            )
        self.check_cluster_interval = cfg["training"]["check_cluster_interval"]

        # Val metrics & scores
        val_iter_interval = cfg["training"]["val_stat_interval"]
        self.val_stat_interval = val_iter_interval
        val_metric_names = ["loss_val"]
        train_iter_interval = cfg["training"]["train_stat_interval"]
        self.val_metrics = Metrics(*val_metric_names)
        self.val_metrics_path = self.run_dir / VAL_METRICS_FILE
        with open(self.val_metrics_path, mode="w") as f:
            f.write(
                "iteration\tepoch\tbatch\t" + "\t".join(self.val_metrics.names) + "\n"
            )

        self.val_scores = Scores(self.n_classes, self.n_prototypes)
        self.val_scores_path = self.run_dir / VAL_SCORES_FILE
        with open(self.val_scores_path, mode="w") as f:
            f.write(
                "iteration\tepoch\tbatch\t" + "\t".join(self.val_scores.names) + "\n"
            )

        # Prototypes & Variances
        self.prototypes_path = coerce_to_path_and_create_dir(
            self.run_dir / "prototypes"
        )
        [
            coerce_to_path_and_create_dir(self.prototypes_path / f"proto{k}")
            for k in range(self.n_prototypes)
        ]
        if self.is_gmm:
            self.variances_path = coerce_to_path_and_create_dir(
                self.run_dir / "variances"
            )
            [
                coerce_to_path_and_create_dir(self.variances_path / f"var{k}")
                for k in range(self.n_prototypes)
            ]

        # Transformation predictions
        self.transformation_path = coerce_to_path_and_create_dir(
            self.run_dir / "transformations"
        )
        self.images_to_tsf = next(iter(self.train_loader))[0][
            :N_TRANSFORMATION_PREDICTIONS
        ].to(self.device)
        for k in range(self.images_to_tsf.size(0)):
            out = coerce_to_path_and_create_dir(self.transformation_path / f"img{k}")
            convert_to_img(self.images_to_tsf[k]).save(out / "input.png")
            [
                coerce_to_path_and_create_dir(out / f"tsf{k}")
                for k in range(self.n_prototypes)
            ]

        # Visdom
        viz_port = cfg["training"].get("visualizer_port")
        if viz_port is not None:
            from visdom import Visdom

            os.environ["http_proxy"] = ""
            self.visualizer = Visdom(
                port=viz_port, env=f"{self.run_dir.parent.name}_{self.run_dir.name}"
            )
            self.visualizer.delete_env(self.visualizer.env)  # Clean env before plotting
            self.print_and_log_info(f"Visualizer initialised at {viz_port}")
        else:
            self.visualizer = None
            self.print_and_log_info("No visualizer initialized")

    def print_and_log_info(self, string):
        print_info(string)
        self.logger.info(string)

    def load_from_tag(self, tag, resume=False):
        self.print_and_log_info("Loading model from run {}".format(tag))
        path = coerce_to_path_and_check_exist(
            RUNS_PATH / self.dataset_name / tag / MODEL_FILE
        )
        checkpoint = torch.load(path, map_location=self.device)
        try:
            self.model.load_state_dict(checkpoint["model_state"])
        except RuntimeError:
            state = safe_model_state_dict(checkpoint["model_state"])
            self.model.load_state_dict(state)
        self.start_epoch, self.start_batch = 1, 1
        if resume:
            self.start_epoch, self.start_batch = (
                checkpoint["epoch"],
                checkpoint.get("batch", 0) + 1,
            )
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            if hasattr(self, "sprite_optimizer"):
                self.sprite_optimizer.load_state_dict(
                    checkpoint["sprite_optimizer_state"]
                )
            self.scheduler.load_state_dict(checkpoint["scheduler_state"])
            self.cur_lr = self.scheduler.get_last_lr()[0]
        self.print_and_log_info(
            "Checkpoint loaded at epoch {}, batch {}".format(
                self.start_epoch, self.start_batch - 1
            )
        )

    @property
    def score_name(self):
        return self.val_scores.score_name

    def print_memory_usage(self, prefix):
        usage = {}
        for attr in [
            "memory_allocated",
            "max_memory_allocated",
            "memory_cached",
            "max_memory_cached",
        ]:
            usage[attr] = getattr(torch.cuda, attr)() * 0.000001
        self.print_and_log_info(
            "{}:\t{}".format(
                prefix,
                " / ".join(["{}: {:.0f}MiB".format(k, v) for k, v in usage.items()]),
            )
        )

    @use_seed()
    def run(self, recluster=False):
        cur_iter = (self.start_epoch - 1) * self.n_batches + self.start_batch - 1
        prev_train_stat_iter, prev_val_stat_iter = cur_iter, cur_iter
        prev_check_cluster_iter = cur_iter
        if self.start_epoch == self.n_epoches + 1:
            self.print_and_log_info("No training, only evaluating")
            if recluster:
                distances = self.evaluate(recluster)
                return distances
            subset = self.evaluate()
            self.print_and_log_info("Training run is over")
            return subset

        for epoch in range(self.start_epoch, self.n_epoches + 1):
            batch_start = self.start_batch if epoch == self.start_epoch else 1
            for batch, (images, labels, _) in enumerate(self.train_loader, start=1):
                if batch < batch_start:
                    continue
                cur_iter += 1
                if cur_iter > self.n_iterations:
                    break

                self.single_train_batch_run(images, epoch)
                if self.scheduler_update_range == "batch":
                    self.update_scheduler(epoch, batch=batch)

                if (cur_iter - prev_train_stat_iter) >= self.train_stat_interval:
                    prev_train_stat_iter = cur_iter
                    self.log_train_metrics(cur_iter, epoch, batch)

                if (cur_iter - prev_check_cluster_iter) >= self.check_cluster_interval:
                    prev_check_cluster_iter = cur_iter
                    self.check_cluster(cur_iter, epoch, batch)

                if (cur_iter - prev_val_stat_iter) >= self.val_stat_interval:
                    prev_val_stat_iter = cur_iter
                    if not self.is_val_empty:
                        self.run_val()
                        self.log_val_metrics(cur_iter, epoch, batch)
                    self.log_images(cur_iter)
                    self.save(epoch=epoch, batch=batch)

            self.model.step()
            if self.scheduler_update_range == "epoch" and batch_start == 1:
                self.update_scheduler(epoch + 1, batch=1)

        self.save_training_metrics()
        subset = self.evaluate()

        self.print_and_log_info("Training run is over")
        return subset

    def update_scheduler(self, epoch, batch):
        self.scheduler.step()
        lr = self.scheduler.get_last_lr()[0]
        if lr != self.cur_lr:
            self.cur_lr = lr
            self.print_and_log_info(
                PRINT_LR_UPD_FMT(epoch, self.n_epoches, batch, self.n_batches, lr)
            )

    def duplicate(self):
        # Model
        self.model.duplicate(self.device)
        self.n_prototypes = self.model.n_prototypes
        self.print_and_log_info(
            "Model is duplicated. Current number of prototypes: {}".format(
                self.n_prototypes
            )
        )

        """
        #Optimizer
        self.optimizer = get_optimizer(self.optimizer_name)([
            dict(params=self.model.cluster_parameters(), **self.cluster_kwargs),
            dict(params=self.model.transformer_parameters(), **self.tsf_kwargs)],
            **self.opt_params)
        self.model.set_optimizer(self.optimizer)
        #Scheduler
        self.scheduler = get_scheduler(self.scheduler_name)(self.optimizer, **self.scheduler_params)
        self.cur_lr = self.scheduler.get_last_lr()[0]
        """

        # Train metrics & check_cluster interval
        metric_names = [
            f"prop_clus{i}" for i in range(self.n_prototypes // 2, self.n_prototypes)
        ]
        metric_names += [
            f"loss_clus{i}" for i in range(self.n_prototypes // 2, self.n_prototypes)
        ]
        self.train_metrics.add(*metric_names)
        with open(self.train_metrics_path, mode="a") as f:
            f.write(
                "iteration\tepoch\tbatch\t" + "\t".join(self.train_metrics.names) + "\n"
            )

        # Prototypes & Variances
        self.prototypes_path = coerce_to_path_and_create_dir(
            self.run_dir / "prototypes"
        )
        [
            coerce_to_path_and_create_dir(self.prototypes_path / f"proto{k}")
            for k in range(self.n_prototypes // 2, self.n_prototypes)
        ]
        if self.is_gmm:
            NotImplementedError("Model duplication is not implemented for DTI-GMM.")

        # Transformation predictions
        for k in range(self.images_to_tsf.size(0)):
            out = coerce_to_path_and_create_dir(self.transformation_path / f"img{k}")
            [
                coerce_to_path_and_create_dir(out / f"tsf{k}")
                for k in range(self.n_prototypes // 2, self.n_prototypes)
            ]

    def single_train_batch_run(self, images, epoch):
        start_time = time.time()
        B = images.size(0)
        self.model.train()
        images = images.to(self.device)

        self.optimizer.zero_grad()
        loss, dist_min_by_sample, argmin_idx = self.model(images, epoch)
        loss.backward()
        self.optimizer.step()

        if hasattr(self, "sprite_optimizer"):
            # torch.nn.utils.clip_grad_value_(self.model.prototype_params, 0.1)
            self.sprite_optimizer.step()

        average_losses = np.zeros(self.n_prototypes)
        with torch.no_grad():
            mask = torch.zeros(B, self.n_prototypes, device=self.device).scatter(
                1, argmin_idx[:, None], 1
            )
            proportions = mask.sum(0).cpu().numpy() / B

            dist_min_by_sample, argmin_idx = (
                dist_min_by_sample.cpu().numpy(),
                argmin_idx.cpu().numpy(),
            )
            argmin_idx = argmin_idx.astype(np.int32)

            for k in range(self.n_prototypes):
                sample_per_clus = argmin_idx == k
                n_clus = np.sum(sample_per_clus)
                avg_clus = (
                    np.sum(sample_per_clus * dist_min_by_sample) / n_clus
                    if n_clus
                    else 0.0
                )
                average_losses[k] = avg_clus
        self.train_metrics.update(
            {
                "time/img": (time.time() - start_time) / B,
                "loss": loss.item(),
            }
        )
        self.train_metrics.update(
            {f"prop_clus{i}": p for i, p in enumerate(proportions)}
        )
        self.train_metrics.update(
            {f"loss_clus{i}": average_losses[i] for i in range(self.n_prototypes)}
        )

    @torch.no_grad()
    def log_images(self, cur_iter):
        self.save_prototypes(cur_iter)
        self.update_visualizer_images(self.model.prototypes, "prototypes", nrow=5)
        if self.is_gmm:
            self.save_variances(cur_iter)
            variances = self.model.variances
            M = variances.flatten(1).max(1)[0][:, None, None, None]
            variances = (variances - self.model.var_min) / (
                M - self.model.var_min + 1e-7
            )
            self.update_visualizer_images(variances, "variances", nrow=5)

        tsf_imgs = self.save_transformed_images(cur_iter)
        C, H, W = tsf_imgs.shape[2:]
        self.update_visualizer_images(
            tsf_imgs.view(-1, C, H, W), "transformations", nrow=self.n_prototypes + 1
        )

    def save_prototypes(self, cur_iter=None):
        prototypes = self.model.prototypes
        for k in range(self.n_prototypes):
            img = convert_to_img(prototypes[k])
            if cur_iter is not None:
                img.save(self.prototypes_path / f"proto{k}" / f"{cur_iter}.jpg")
            else:
                img.save(self.prototypes_path / f"prototype{k}.png")

    def save_variances(self, cur_iter=None):
        variances = self.model.variances
        for k in range(self.n_prototypes):
            img = convert_to_img(variances[k])
            if cur_iter is not None:
                img.save(self.variances_path / f"var{k}" / f"{cur_iter}.jpg")
            else:
                img.save(self.variances_path / f"variance{k}.png")

    @torch.no_grad()
    def save_transformed_images(self, cur_iter=None):
        self.model.eval()
        output = self.model.transform(self.images_to_tsf)

        transformed_imgs = torch.cat([self.images_to_tsf.unsqueeze(1), output], 1)
        for k in range(transformed_imgs.size(0)):
            for j, img in enumerate(transformed_imgs[k][1:]):
                if cur_iter is not None:
                    convert_to_img(img).save(
                        self.transformation_path
                        / f"img{k}"
                        / f"tsf{j}"
                        / f"{cur_iter}.jpg"
                    )
                else:
                    convert_to_img(img).save(
                        self.transformation_path / f"img{k}" / f"tsf{j}.png"
                    )
        return transformed_imgs

    def update_visualizer_images(self, images, title, nrow):
        if self.visualizer is None:
            return None

        if max(images.shape[1:]) > VIZ_MAX_IMG_SIZE:
            images = torch.nn.functional.interpolate(
                images, size=VIZ_MAX_IMG_SIZE, mode="bilinear"
            )
        self.visualizer.images(
            images.clamp(0, 1),
            win=title,
            nrow=nrow,
            opts=dict(
                title=title, store_history=True, width=VIZ_WIDTH, height=VIZ_HEIGHT
            ),
        )

    def check_cluster(self, cur_iter, epoch, batch):
        proportions = [
            self.train_metrics[f"prop_clus{i}"].avg for i in range(self.n_prototypes)
        ]
        reassigned, idx = self.model.reassign_empty_clusters(proportions)
        msg = PRINT_CHECK_CLUSTERS_FMT(
            epoch, self.n_epoches, batch, self.n_batches, reassigned, idx
        )
        self.print_and_log_info(msg)
        self.train_metrics.reset(*[f"prop_clus{i}" for i in range(self.n_prototypes)])

    def log_train_metrics(self, cur_iter, epoch, batch):
        # Print & write metrics to file
        stat = PRINT_TRAIN_STAT_FMT(
            epoch, self.n_epoches, batch, self.n_batches, self.train_metrics
        )
        self.print_and_log_info(stat)
        with open(self.train_metrics_path, mode="a") as f:
            f.write(
                "{}\t{}\t{}\t".format(cur_iter, epoch, batch)
                + "\t".join(map("{:.4f}".format, self.train_metrics.avg_values))
                + "\n"
            )

        self.update_visualizer_metrics(cur_iter, train=True)
        self.train_metrics.reset(
            *(
                ["time/img", "loss"]
                + [f"loss_clus{i}" for i in range(self.n_prototypes)]
            )
        )

    def update_visualizer_metrics(self, cur_iter, train):
        if self.visualizer is None:
            return None

        split, metrics = (
            ("train", self.train_metrics) if train else ("val", self.val_metrics)
        )
        loss_names = [n for n in metrics.names if "loss" in n]
        y, x = [[metrics[n].avg for n in loss_names]], [[cur_iter] * len(loss_names)]
        self.visualizer.line(
            y,
            x,
            win=f"{split}_loss",
            update="append",
            opts=dict(
                title=f"{split}_loss",
                legend=loss_names,
                width=VIZ_WIDTH,
                height=VIZ_HEIGHT,
            ),
        )

        if train:
            if self.n_prototypes > 1:
                # Cluster proportions
                proportions = [
                    metrics[f"prop_clus{i}"].avg for i in range(self.n_prototypes)
                ]
                self.visualizer.bar(
                    proportions,
                    win=f"train_cluster_prop",
                    opts=dict(
                        title=f"train_cluster_proportions",
                        width=VIZ_HEIGHT,
                        height=VIZ_HEIGHT,
                    ),
                )
        else:
            # Scores
            names = list(filter(lambda name: "cls" not in name, self.val_scores.names))
            y, x = [[self.val_scores[n] for n in names]], [[cur_iter] * len(names)]
            self.visualizer.line(
                y,
                x,
                win=f"global_scores",
                update="append",
                opts=dict(
                    title=f"global_scores",
                    legend=names,
                    width=VIZ_WIDTH,
                    height=VIZ_HEIGHT,
                ),
            )

            y, x = [[self.val_scores[f"acc_cls{i}"] for i in range(self.n_classes)]], [
                [cur_iter] * self.n_classes
            ]
            self.visualizer.line(
                y,
                x,
                win=f"acc_by_cls",
                update="append",
                opts=dict(
                    title=f"acc_by_cls",
                    legend=[f"cls{i}" for i in range(self.n_classes)],
                    width=VIZ_WIDTH,
                    heigh=VIZ_HEIGHT,
                ),
            )

    @torch.no_grad()
    def run_val(self):
        self.model.eval()
        for images, labels, _ in self.val_loader:
            images = images.to(self.device)
            dist_min_by_sample, argmin_idx = self.model(images)[1:]
            loss_val = dist_min_by_sample.mean()

            self.val_metrics.update({"loss_val": loss_val.item()})
            self.val_scores.update(labels.long().numpy(), argmin_idx.cpu().numpy())

    def log_val_metrics(self, cur_iter, epoch, batch):
        stat = PRINT_VAL_STAT_FMT(
            epoch, self.n_epoches, batch, self.n_batches, self.val_metrics
        )
        self.print_and_log_info(stat)
        with open(self.val_metrics_path, mode="a") as f:
            f.write(
                "{}\t{}\t{}\t".format(cur_iter, epoch, batch)
                + "\t".join(map("{:.4f}".format, self.val_metrics.avg_values))
                + "\n"
            )

        scores = self.val_scores.compute()
        self.print_and_log_info(
            "val_scores: "
            + ", ".join(["{}={:.4f}".format(k, v) for k, v in scores.items()])
        )
        with open(self.val_scores_path, mode="a") as f:
            f.write(
                "{}\t{}\t{}\t".format(cur_iter, epoch, batch)
                + "\t".join(map("{:.4f}".format, scores.values()))
                + "\n"
            )

        self.update_visualizer_metrics(cur_iter, train=False)
        self.val_scores.reset()
        self.val_metrics.reset()

    def save(self, epoch, batch):
        state = {
            "epoch": epoch,
            "batch": batch,
            "model_name": self.model_name,
            "model_kwargs": self.model_kwargs,
            "model_state": self.model.state_dict(),
            "n_prototypes": self.n_prototypes,
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
        }
        if hasattr(self, "sprite_optimizer"):
            state["sprite_optimizer_state"] = self.sprite_optimizer.state_dict()
        save_path = self.run_dir / MODEL_FILE
        torch.save(state, save_path)
        self.print_and_log_info("Model saved at {}".format(save_path))

    def save_training_metrics(self):
        df_train = pd.read_csv(self.train_metrics_path, sep="\t", index_col=0)
        df_val = pd.read_csv(self.val_metrics_path, sep="\t", index_col=0)
        df_scores = pd.read_csv(self.val_scores_path, sep="\t", index_col=0)
        if len(df_train) == 0:
            self.print_and_log_info("No metrics or plots to save")
            return

        # Losses
        losses = list(filter(lambda s: s.startswith("loss"), self.train_metrics.names))
        df = df_train.join(df_val[["loss_val"]], how="outer")
        fig = plot_lines(df, losses + ["loss_val"], title="Loss")
        fig.savefig(self.run_dir / "loss.pdf")

        # Cluster proportions
        names = list(filter(lambda s: s.startswith("prop_"), self.train_metrics.names))
        fig = plot_lines(df, names, title="Cluster proportions")
        fig.savefig(self.run_dir / "cluster_proportions.pdf")
        s = df[names].iloc[-1]
        s.index = list(map(lambda n: n.replace("prop_clus", ""), names))
        fig = plot_bar(s, title="Final cluster proportions")
        fig.savefig(self.run_dir / "cluster_proportions_final.pdf")

        # Validation
        if not self.is_val_empty:
            names = list(filter(lambda name: "cls" not in name, self.val_scores.names))
            fig = plot_lines(df_scores, names, title="Global scores", unit_yaxis=True)
            fig.savefig(self.run_dir / "global_scores.pdf")

            fig = plot_lines(
                df_scores,
                [f"acc_cls{i}" for i in range(self.n_classes)],
                title="Scores by cls",
                unit_yaxis=True,
            )
            fig.savefig(self.run_dir / "scores_by_cls.pdf")

        # Prototypes & Variances
        size = MAX_GIF_SIZE if MAX_GIF_SIZE < max(self.img_size) else self.img_size
        with torch.no_grad():
            self.save_prototypes()
        if self.is_gmm:
            self.save_variances()
        for k in range(self.n_prototypes):
            save_gif(self.prototypes_path / f"proto{k}", f"prototype{k}.gif", size=size)
            shutil.rmtree(str(self.prototypes_path / f"proto{k}"))
            if self.is_gmm:
                save_gif(self.variances_path / f"var{k}", f"variance{k}.gif", size=size)
                shutil.rmtree(str(self.variances_path / f"var{k}"))

        # Transformation predictions
        if self.model.transformer.is_identity:
            # no need to keep transformation predictions
            shutil.rmtree(str(self.transformation_path))
            coerce_to_path_and_create_dir(self.transformation_path)
        else:
            self.save_transformed_images()
            for i in range(self.images_to_tsf.size(0)):
                for k in range(self.n_prototypes):
                    save_gif(
                        self.transformation_path / f"img{i}" / f"tsf{k}",
                        f"tsf{k}.gif",
                        size=size,
                    )
                    shutil.rmtree(str(self.transformation_path / f"img{i}" / f"tsf{k}"))

        self.print_and_log_info("Training metrics and visuals saved")

    def evaluate(self, recluster=False):
        self.model.eval()
        no_label = self.train_loader.dataset[0][1] == -1
        if no_label:
            if recluster:
                return self.distance_eval()
            subset = self.qualitative_eval()
            return subset
        else:
            self.quantitative_eval()
            subset = self.qualitative_eval()
        self.print_and_log_info("Evaluation is over")

    @torch.no_grad()
    def distance_eval(self):
        dataset = self.train_loader.dataset
        train_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.n_workers,
            shuffle=False,
        )

        paths = []
        distances = []
        ids = []
        for images, labels, path in train_loader:
            images = images.to(self.device)
            out = self.model(images)[1:]
            dist_min_by_sample, argmin_idx = map(lambda t: t.cpu().numpy(), out)
            paths.append(path)
            distances.append(dist_min_by_sample)
            ids.append(argmin_idx)

        paths = np.concatenate(paths)
        distances = np.concatenate(distances)
        ids = np.concatenate(ids)

        ret_val = []
        for p, d, i in zip(paths, distances, ids):
            ret_val.append((p, d, i))

        return np.array(ret_val)

    @torch.no_grad()
    def qualitative_eval(self):
        """Routine to save qualitative results"""
        loss = AverageMeter()
        scores_path = self.run_dir / FINAL_SCORES_FILE
        with open(scores_path, mode="w") as f:
            f.write("loss\n")
            for k in range(self.n_prototypes):
                f.write("clus_loss{" + str(k) + "}\n")
            for k in range(self.n_prototypes):
                f.write("ctr_clus_loss{" + str(k) + "}\n")
        cluster_path = coerce_to_path_and_create_dir(self.run_dir / "clusters")
        dataset = self.train_loader.dataset
        train_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.n_workers,
            shuffle=False,
        )
        # Compute difference (for hierarchical clustering)
        if self.n_prototypes == 2:
            error_averages = {k: AverageTensorMeter() for k in range(self.n_prototypes)}
            proto_averages = {k: AverageTensorMeter() for k in range(self.n_prototypes)}
            for images, labels, path in train_loader:
                images = images.to(self.device)
                out = self.model(images)[1:]
                dist_min_by_sample, argmin_idx = map(lambda t: t.cpu().numpy(), out)

                transformed_imgs = self.model.transform(images)
                transformed_c0_p0 = transformed_imgs[
                    argmin_idx == 0, 0, ...
                ]  # Transformed p0 for images in c0
                transformed_c0_p1 = transformed_imgs[
                    argmin_idx == 0, 1, ...
                ]  # Transformed p1 for images in c0
                transformed_c1_p0 = transformed_imgs[
                    argmin_idx == 1, 0, ...
                ]  # Transformed p0 for images in c1
                transformed_c1_p1 = transformed_imgs[
                    argmin_idx == 1, 1, ...
                ]  # Transformed p1 for images in c1

                transform_diff_c0 = torch.where(
                    transformed_c0_p0 - transformed_c0_p1 >= 0,
                    transformed_c0_p0 - transformed_c0_p1,
                    0,
                )
                transform_diff_c1 = torch.where(
                    transformed_c1_p0 - transformed_c1_p1 >= 0,
                    transformed_c1_p0 - transformed_c1_p1,
                    0,
                )

                if transform_diff_c0.shape[0]:
                    proto_diff_c0 = self.model.transform(
                        transform_diff_c0, inverse=True
                    )
                    proto_averages[0].update(proto_diff_c0.cpu())
                if transform_diff_c1.shape[0]:
                    proto_diff_c1 = self.model.transform(
                        transform_diff_c1, inverse=True
                    )
                    proto_averages[1].update(proto_diff_c1.cpu())

                error_averages[1].update(
                    abs(images[argmin_idx == 0] - transformed_c0_p1).cpu()
                )
                error_averages[0].update(
                    abs(images[argmin_idx == 1] - transformed_c1_p0).cpu()
                )
            for k in range(2):
                convert_to_img(error_averages[k].avg).save(
                    self.run_dir / "err_diff_avg_{:d}.png".format(k)
                )
                convert_to_img(proto_averages[k].avg[:, 0, ...]).save(
                    self.run_dir / "rec_diff_tr0_{:d}.png".format(k)
                )
                convert_to_img(proto_averages[k].avg[:, 1, ...]).save(
                    self.run_dir / "rec_diff_tr1_{:d}.png".format(k)
                )

            # Save the leaf index the image is clustered in and the reconstruction error
            leaf_id = str(self.run_dir).split("_")[-1]
            if len(leaf_id) == 6:
                leaf_id_by_sample = []
                rec_err_by_sample = []
                paths = []
                train_loader = DataLoader(
                    dataset,
                    batch_size=self.batch_size,
                    num_workers=self.n_workers,
                    shuffle=False,
                )
                for images, labels, path in train_loader:
                    images = images.to(self.device)
                    out = self.model(images)[1:]
                    dist_min_by_sample, argmin_idx = map(lambda t: t.cpu().numpy(), out)
                    leaf_id_by_sample.extend([leaf_id + str(idx) for idx in argmin_idx])
                    rec_err_by_sample.append(dist_min_by_sample)
                    paths.append(path)
                np.save(
                    self.run_dir / "rec_err_by_sample.npy",
                    np.concatenate(rec_err_by_sample, axis=0),
                )
                np.save(
                    self.run_dir / "leaf_id_by_sample.npy", np.array(leaf_id_by_sample)
                )
                np.save(self.run_dir / "paths.npy", np.concatenate(paths, axis=0))
        # Compute results
        distances, cluster_idx = np.array([]), np.array([], dtype=np.int32)
        averages = {k: AverageTensorMeter() for k in range(self.n_prototypes)}
        average_losses = {k: AverageMeter() for k in range(self.n_prototypes)}
        average_ctr_losses = {k: AverageMeter() for k in range(self.n_prototypes)}
        subset_img = [np.array([]) for k in range(self.n_prototypes)]
        cluster_by_path = []
        train_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.n_workers,
            shuffle=False,
        )
        for images, labels, path in train_loader:
            images = images.to(self.device)
            out = self.model(images)[1:]
            dist_min_by_sample, argmin_idx = map(lambda t: t.cpu().numpy(), out)

            # for cluster in range(self.n_prototypes):
            #    subset_img[cluster] = np.hstack(
            #        [subset_img[cluster], np.array(path)[argmin_idx == cluster]]
            #    )

            loss.update(dist_min_by_sample.mean(), n=len(dist_min_by_sample))
            argmin_idx = argmin_idx.astype(np.int32)
            distances = np.hstack([distances, dist_min_by_sample])
            cluster_idx = np.hstack([cluster_idx, argmin_idx])
            if hasattr(train_loader.dataset, "data_path"):
                cluster_by_path += [
                    (os.path.relpath(p, train_loader.dataset.data_path), argmin_idx[i]) 
                    for i, p in enumerate(path)]
            """
            #dist_max_by_sample, argmax_idx_ = map(
            #    lambda t: t.cpu().numpy(), dist.max(1)
            #)
            #argmax_idx_ = argmax_idx_.astype(np.int32)

            transformed_imgs = self.model.transform(images).cpu()
            for k in range(self.n_prototypes):
                imgs = transformed_imgs[argmin_idx == k, k]
                averages[k].update(imgs)

                sample_per_clus = argmin_idx == k
                n_clus = np.sum(sample_per_clus)
                avg_clus = (
                    np.sum(sample_per_clus * dist_min_by_sample) / n_clus
                    if n_clus
                    else 0.0
                )
                average_losses[k].update(avg_clus, n=n_clus)

                ctr_sample_per_clus = argmax_idx_ == k
                n_ctr_clus = np.sum(ctr_sample_per_clus)
                avg_ctr_clus = (
                    np.sum(ctr_sample_per_clus * dist_max_by_sample) / n_ctr_clus
                    if n_ctr_clus
                    else 0.0
                )
                average_ctr_losses[k].update(avg_ctr_clus, n_ctr_clus)
            """
        # Save cluster_by_path as csv
        if cluster_by_path:
            cluster_by_path = pd.DataFrame(
                cluster_by_path, columns=["path", "cluster_id"]
            ).set_index("path")
            cluster_by_path.to_csv(self.run_dir / "cluster_by_path.csv")

        self.print_and_log_info("final_loss: {:.5}".format(loss.avg))
        self.print_and_log_info(
            "".join(
                [
                    "final_clus_loss{:d}: {:.5} ".format(k, average_losses[k].avg)
                    if average_losses[k].avg
                    else "final_clus_loss{:d}: 0. ".format(k)
                    for k in range(self.n_prototypes)
                ]
            )
        )
        self.print_and_log_info(
            "".join(
                [
                    "final_ctr_clus_loss{:d}: {:.5} ".format(
                        k, average_ctr_losses[k].avg
                    )
                    if average_ctr_losses[k].avg
                    else "final_ctr_clus_loss{:d}: 0. ".format(k)
                    for k in range(self.n_prototypes)
                ]
            )
        )

        with open(scores_path, mode="a") as f:
            f.write("{:.5}\n".format(loss.avg))
            for k in range(self.n_prototypes):
                f.write("{:.5}\n".format(average_losses[k].avg)) if average_losses[
                    k
                ].avg else f.write("0.\n")
            for k in range(self.n_prototypes):
                f.write(
                    "{:.5}\n".format(average_ctr_losses[k].avg)
                ) if average_ctr_losses[k].avg else f.write("0.\n")

        # Save results
        with open(cluster_path / "cluster_counts.tsv", mode="w") as f:
            f.write("\t".join([str(k) for k in range(self.n_prototypes)]) + "\n")
            f.write(
                "\t".join([str(averages[k].count) for k in range(self.n_prototypes)])
                + "\n"
            )
        for k in range(self.n_prototypes):
            path = coerce_to_path_and_create_dir(cluster_path / f"cluster{k}")
            indices = np.where(cluster_idx == k)[0]
            top_idx = np.argsort(distances[indices])[:N_CLUSTER_SAMPLES]
            for j, idx in enumerate(top_idx):
                inp = dataset[indices[idx]][0].unsqueeze(0).to(self.device)
                convert_to_img(inp).save(path / f"top{j}_raw.png")
                if not self.model.transformer.is_identity:
                    convert_to_img(self.model.transform(inp)[0, k]).save(
                        path / f"top{j}_tsf.png"
                    )
            if len(indices) <= N_CLUSTER_SAMPLES:
                random_idx = indices
            else:
                random_idx = np.random.choice(indices, N_CLUSTER_SAMPLES, replace=False)
            for j, idx in enumerate(random_idx):
                inp = dataset[idx][0].unsqueeze(0).to(self.device)
                convert_to_img(inp).save(path / f"random{j}_raw.png")
                if not self.model.transformer.is_identity:
                    convert_to_img(self.model.transform(inp)[0, k]).save(
                        path / f"random{j}_tsf.png"
                    )
            try:
                convert_to_img(averages[k].avg).save(path / "avg.png")
            except AssertionError:
                print_warning(f"no image found in cluster {k}")

        if self.parent_model != None:
            scores_path = self.run_dir / "parent_scores.tsv"
            with open(scores_path, mode="w") as f:
                f.write("loss\n")

                for set in subset_img:
                    sub_dataset = get_subset(self.dataset_name)(
                        "train", set, **self.dataset_kwargs
                    )
                    sub_train_loader = DataLoader(
                        sub_dataset,
                        batch_size=self.batch_size,
                        num_workers=self.n_workers,
                        shuffle=False,
                    )
                    parent_model_sd = torch.load(
                        self.parent_model + "/" + MODEL_FILE, map_location=self.device
                    )
                    parent_model = copy.deepcopy(self.model)
                    parent_model.load_state_dict(parent_model_sd["model_state"])
                    loss = AverageMeter()
                    for images, labels, path in sub_train_loader:
                        images = images.to(self.device)
                        dist = parent_model(images)[1]
                        dist_min_by_sample, argmin_idx = map(
                            lambda t: t.cpu().numpy(), dist.min(1)
                        )

                        loss.update(
                            dist_min_by_sample.mean(), n=len(dist_min_by_sample)
                        )
                    self.print_and_log_info("parent_loss: {:.5}".format(loss.avg))
                    f.write("{:.5}\n".format(loss.avg))

        return subset_img

    @torch.no_grad()
    def quantitative_eval(self):
        """Routine to save quantitative results: loss + scores"""
        loss = AverageMeter()
        scores_path = self.run_dir / FINAL_SCORES_FILE
        scores = Scores(self.n_classes, self.n_prototypes)
        with open(scores_path, mode="w") as f:
            f.write("loss\t" + "\t".join(scores.names) + "\n")

        dataset = get_dataset(self.dataset_name)(
            "train", eval_mode=True, subset=None, **self.dataset_kwargs
        )
        loader = DataLoader(
            dataset, batch_size=self.batch_size, num_workers=self.n_workers
        )
        for images, labels, _ in loader:
            images = images.to(self.device)
            dist_min_by_sample, argmin_idx = self.model(images)[1:]
            loss.update(dist_min_by_sample.mean(), n=len(dist_min_by_sample))
            scores.update(labels.long().numpy(), argmin_idx.cpu().numpy())

        scores = scores.compute()
        self.print_and_log_info("final_loss: {:.5}".format(loss.avg))
        self.print_and_log_info(
            "final_scores: "
            + ", ".join(["{}={:.4f}".format(k, v) for k, v in scores.items()])
        )
        with open(scores_path, mode="a") as f:
            f.write(
                "{:.5}\t".format(loss.avg)
                + "\t".join(map("{:.4f}".format, scores.values()))
                + "\n"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pipeline to train a NN model specified by a YML config"
    )
    parser.add_argument(
        "-t",
        "--tag",
        nargs="?",
        type=str,
        required=True,
        help="Run tag of the experiment",
    )
    parser.add_argument(
        "-c", "--config", nargs="?", type=str, required=True, help="Config file name"
    )
    args = parser.parse_args()

    assert args.tag is not None and args.config is not None
    config = coerce_to_path_and_check_exist(CONFIGS_PATH / args.config)
    with open(config) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)
    seed = cfg["training"].get("seed", 4321)
    dataset = cfg["dataset"]["name"]

    run_dir = RUNS_PATH / dataset / args.tag
    run_dir = str(run_dir)
    n_dup = cfg["training"].get("n_dup", 0)
    subsets = {}
    for n in range(n_dup + 1):
        if subsets:
            keys = list(subsets.keys())
            for i, subset in enumerate(keys):
                assert len(subsets[subset]) > 0
                trainer = Trainer(
                    config,
                    run_dir + "_" + subset,
                    seed=seed,
                    subset=subsets[subset],
                    parent_model=run_dir + "_" + subset[:-1],
                )
                temp = trainer.run(seed=seed)
                subsets[subset + "0"] = temp[0]
                subsets[subset + "1"] = temp[1]
                subsets.pop(subset, None)
        else:
            trainer = Trainer(config, run_dir + "_0", seed=seed)
            temp = trainer.run(seed=seed)
            if temp:
                subsets["00"] = temp[0]
                subsets["01"] = temp[1]
