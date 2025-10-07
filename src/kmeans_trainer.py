import os
import time
import hydra

import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from .utils.consts import *
from .dataset import get_dataset
from .model import get_model
from .model.tools import safe_model_state_dict
from .optimizer import get_optimizer
from .utils import (
    use_seed,
    coerce_to_path_and_check_exist,
    coerce_to_path_and_create_dir,
)
from .utils.image import convert_to_img, unify_channels
from .utils.logger import print_warning
from .utils.metrics import AverageTensorMeter, AverageMeter, Scores
from .utils.path import RUNS_PATH

from .abstract_trainer import AbstractTrainer, run_trainer


class Trainer(AbstractTrainer):
    """Pipeline to train a NN model using a certain dataset, both specified by an YML config"""
    model_name = "dtikmeans"

    # optimizer
    sprite_optimizer = None
    parent_model = None

    @use_seed()
    def __init__(self, cfg, run_dir, save=False, *args, **kwargs):
        super().__init__(cfg, run_dir, save, *args, **kwargs)

    ######################
    #   SETUP METHODS    #
    ######################

    def get_model(self):
        """Return model instance"""
        return get_model(self.model_name)(
            self.train_loader.dataset,
            **self.model_kwargs
        )

    def setup_directories(self):
        super().setup_directories()

        if not self.save_img:
            return

        for k in range(self.images_to_tsf.size(0)):
            for j in range(self.n_prototypes):
                coerce_to_path_and_create_dir(self.transformation_path / f"img{k}" / f"tsf{j}")

        if self.is_gmm:
            self.variances_path = coerce_to_path_and_create_dir(self.run_dir / "variances")
            for k in range(self.n_prototypes):
                coerce_to_path_and_create_dir(self.variances_path / f"var{k}")

    def setup_optimizer(self, *args, **kwargs):
        """Configure optimizer for training"""
        train_params = self.cfg.get("training", {})
        opt_params = train_params.get("optimizer", {})
        sprite_opt_params = train_params.get("sprite_optimizer", {})
        optimizer_name = train_params.get("optimizer_name", "adam")
        cluster_kwargs = opt_params.get("cluster", {})
        tsf_kwargs = opt_params.get("transformer", {})
        sprite_optimizer_name = sprite_opt_params.get("name", None)

        # Create optimizers with appropriate parameter groups
        if sprite_optimizer_name in ["SGD", "sgd"]:
            # Special case: separate optimizers for clusters and transformers
            self.sprite_optimizer = get_optimizer(sprite_optimizer_name)(
                [dict(params=self.model.cluster_parameters(), **cluster_kwargs)],
                **sprite_opt_params,
            )
            self.optimizer = get_optimizer(optimizer_name)(
                [dict(params=self.model.transformer_parameters(), **tsf_kwargs)],
                **opt_params,
            )
            self.model.set_optimizer(self.optimizer, self.sprite_optimizer)
        else:
            # Default case: single optimizer with multiple parameter groups
            self.optimizer = get_optimizer(optimizer_name)(
                [dict(params=self.model.cluster_parameters(), **cluster_kwargs)]
                + [dict(params=self.model.transformer_parameters(), **tsf_kwargs)],
                **opt_params,
            )
            self.model.set_optimizer(self.optimizer)

        # Log optimizer configuration
        self.print_and_log_info(
            f"Using optimizer {optimizer_name} with kwargs {opt_params}"
        )
        self.print_and_log_info(f"cluster kwargs {cluster_kwargs}")
        self.print_and_log_info(f"transformer kwargs {tsf_kwargs}")

    @property
    def train_metric_names(self):
        metric_names = ["time/img", "loss"]
        metric_names += [f"prop_clus{i}" for i in range(self.n_prototypes)]
        metric_names += [f"proba_clus{i}" for i in range(self.n_prototypes)]
        return metric_names

    @property
    def has_sprite_optimizer(self):
        return hasattr(self, "sprite_optimizer") and self.sprite_optimizer is not None

    def setup_val_scores(self):
        self.val_scores = Scores(self.n_classes, self.n_prototypes)

    def setup_visualizer(self, *args, **kwargs):
        """Set up real-time visualization (e.g., Visdom)"""
        pass

    def setup_additional_components(self, *args, **kwargs):
        """Set up any additional trainer-specific components"""
        pass

    ######################
    #    MAIN METHODS    #
    ######################

    def load_from_tag(self, tag, resume=False):
        self.print_and_log_info(f"Loading model from run {tag}")
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
            if self.has_sprite_optimizer:
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

    @use_seed()
    def run(self):
        cur_iter = (self.start_epoch - 1) * self.n_batches + self.start_batch - 1
        prev_train_stat_iter, prev_val_stat_iter = cur_iter, cur_iter
        prev_check_cluster_iter = cur_iter
        if self.start_epoch == self.n_epochs + 1:
            self.print_and_log_info("No training, only evaluating")

            self.evaluate()
            self.print_and_log_info("Training run is over")
            return

        for epoch in range(self.start_epoch, self.n_epochs + 1):
            batch_start = self.start_batch if epoch == self.start_epoch else 1
            for batch, (images, labels, masks, _) in enumerate(self.train_loader, start=1):
                if batch < batch_start:
                    continue
                cur_iter += 1
                if cur_iter > self.n_iterations:
                    break

                self.single_train_batch_run(images, labels, masks)
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
                    if self.save_img:
                        self.log_images(cur_iter)
                    self.save(epoch=epoch, batch=batch)

            self.model.step()
            if self.scheduler_update_range == "epoch" and batch_start == 1:
                self.update_scheduler(epoch + 1, batch=1)

        self.save_training_metrics()
        self.evaluate()

        self.print_and_log_info("Training run is over")

    def single_train_batch_run(self, images, labels, masks=None):
        start_time = time.time()
        B = images.size(0)
        self.model.train()
        images = images.to(self.device)

        masks = masks.to(self.device) if masks != [] else None

        self.optimizer.zero_grad()
        loss, distances, probas = self.model(images, masks)
        loss.backward()
        self.optimizer.step()

        if self.has_sprite_optimizer:
            self.sprite_optimizer.step()

        with torch.no_grad():
            mask, argmin_idx, proportions = self.compute_cluster_assignments(distances, probas, B)

            if hasattr(self.model, "proba"):
                winners = probas * mask  # B, K
                probabilities = (
                    winners.sum(0).cpu().numpy() / mask.sum(0).cpu().numpy()
                )  # K
                isnan = np.isnan(probabilities)
                probabilities[isnan] = 0
                self.train_metrics.update(
                    {f"proba_clus{i}": p for i, p in enumerate(probabilities)}
                )

            self.train_metrics.update(
                {
                    "time/img": (time.time() - start_time) / B,
                    "loss": loss.item(),
                }
            )
            self.train_metrics.update(
                {f"prop_clus{i}": p for i, p in enumerate(proportions)}
            )

    @torch.no_grad()
    def get_cluster_assignments(self, images):
        distances = self.model(images)[1]
        dist_min_by_sample, argmin_idx = map(
            lambda t: t.cpu().numpy(), distances.min(1)
        )
        return dist_min_by_sample, argmin_idx

    ######################
    #   SAVING METHODS   #
    ######################

    def save_variances(self, cur_iter=None, path=None):
        self.save_pred(cur_iter, pred_name="variance", prefix="var", n_preds=self.n_prototypes, pred_path=path)

    def get_transformed_images(self, output):
        """Ensure image and output have matching channels, then concatenate them."""
        img_channels = self.images_to_tsf.shape[1]
        out_channels = output.shape[2] if output.ndim == 5 else output.shape[1]

        if img_channels != out_channels:
            target_channels = max(img_channels, out_channels)
            images = unify_channels(self.images_to_tsf, target_channels)
            outputs = unify_channels(output, target_channels)
        else:
            images = self.images_to_tsf
            outputs = output

        return torch.cat([images.unsqueeze(1), outputs], dim=1)

    @torch.no_grad()
    def save_transformed_images(self, cur_iter=None):
        self.model.eval()
        output = self.model.transform(self.images_to_tsf)

        transformed_imgs = self.get_transformed_images(output)
        for k in range(transformed_imgs.size(0)):
            for j, img in enumerate(transformed_imgs[k][1:]):
                if cur_iter is None:
                    convert_to_img(img).save(
                        self.transformation_path / f"img{k}" / f"tsf{j}.png"
                    )
                elif self.save_iter:
                    convert_to_img(img).save(
                        self.transformation_path / f"img{k}" / f"tsf{j}/{cur_iter}.jpg"
                    )

                # img_name = f"tsf{j}/{cur_iter}.jpg" if cur_iter is not None else f"tsf{j}.png"
                # convert_to_img(img).save(
                #     self.transformation_path / f"img{k}" / img_name
                # )

        return transformed_imgs

    def _save_additional_image_gifs(self, size):
        """Save additional image GIFs specific to KMeans trainer"""
        if self.is_gmm:
            self.save_variances()
            for k in range(self.n_prototypes):
                self.save_gif_to_path(self.variances_path / f"var{k}", f"variance{k}.gif", size=size)

    ######################
    #   LOGGING METHODS  #
    ######################

    def _log_model_specific_images(self, cur_iter):
        """Visualize GMM variances if applicable"""
        if self.is_gmm:
            self.save_variances(cur_iter)
            variances = self.model.variances
            # Normalize variances for visualization
            M = variances.flatten(1).max(1)[0][:, None, None, None]
            variances = (variances - self.model.var_min) / (M - self.model.var_min + 1e-7)
            self.update_visualizer_images(variances, "variances", nrow=5)

    ######################
    # VALIDATION METHODS #
    ######################

    def check_cluster(self, cur_iter, epoch, batch):
        proportions = [
            self.train_metrics[f"prop_clus{i}"].avg for i in range(self.n_prototypes)
        ]
        reassigned, idx = self.model.reassign_empty_clusters(proportions)
        if len(reassigned) != 0:
            msg = f"{self.progress_str(epoch, batch)}: Reassigned clusters {reassigned} from cluster {idx}"
            self.print_and_log_info(msg)
        self.train_metrics.reset(*[f"prop_clus{i}" for i in range(self.n_prototypes)])
        self.train_metrics.reset(*[f"proba_clus{i}" for i in range(self.n_prototypes)])

    @torch.no_grad()
    def run_val(self):
        self.model.eval()
        for images, labels, _, _ in self.val_loader:
            images = images.to(self.device)
            if hasattr(self.model, "proba"):
                _, out, probas = self.model(images)
                argmin_idx = probas.argmax(1)
                dist_min_by_sample = out[:, argmin_idx]
            else:
                distances = self.model(images)[1]
                dist_min_by_sample, argmin_idx = distances.min(1)
            loss_val = dist_min_by_sample.mean()

            self.val_metrics.update({"loss_val": loss_val.item()})
            self.val_scores.update(labels.long().numpy(), argmin_idx.cpu().numpy())

    ######################
    # EVALUATION METHODS #
    ######################

    def evaluate(self):
        self.model.eval()
        no_label = self.train_loader.dataset[0][1] == -1
        if no_label:
            self.qualitative_eval()
        else:
            # self.qualitative_eval()
            self.quantitative_eval()
        self.print_and_log_info("Evaluation is over")

    @torch.no_grad()
    def qualitative_eval(self):
        """Routine to save qualitative results"""
        loss = AverageMeter()
        scores_path = self.run_dir / FINAL_SCORES_FILE
        with open(scores_path, mode="w") as f:
            f.write("loss\n")

        # self.cluster_path = coerce_to_path_and_create_dir(self.run_dir / "clusters") # MARKER
        dataset = self.train_loader.dataset
        train_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.n_workers,
            shuffle=False,
        )
        # Compute results
        distances, cluster_idx = np.array([]), np.array([], dtype=np.int32)
        averages = {k: AverageTensorMeter() for k in range(self.n_prototypes)}
        cluster_by_path = []
        for images, _, _, path in train_loader:
            images = images.to(self.device)

            if hasattr(self.model, "proba"):
                _, out, probas = self.model(images)
                argmin_idx = probas.argmax(1)
                argmin_idx = argmin_idx.cpu().numpy()
                dist_min_by_sample = out[:, argmin_idx].cpu().numpy()
            else:
                out = self.model(images)[1]
                dist_min_by_sample, argmin_idx = map(
                    lambda t: t.cpu().numpy(), out.min(1)
                )

            loss.update(dist_min_by_sample.mean(), n=len(dist_min_by_sample))
            argmin_idx = argmin_idx.astype(np.int32)
            distances = np.hstack([distances, dist_min_by_sample])
            cluster_idx = np.hstack([cluster_idx, argmin_idx])
            if hasattr(train_loader.dataset, "data_path"):
                cluster_by_path += [
                    (os.path.relpath(p, train_loader.dataset.data_path), argmin_idx[i])
                    for i, p in enumerate(path)
                ]
            transformed_imgs = self.model.transform(images).cpu()
            for k in range(self.n_prototypes):
                imgs = transformed_imgs[argmin_idx == k, k]
                averages[k].update(imgs)
        if cluster_by_path:
            cluster_by_path = pd.DataFrame(
                cluster_by_path, columns=["path", "cluster_id"]
            ).set_index("path")
            cluster_by_path.to_csv(self.run_dir / "cluster_by_path.csv")

        self.print_and_log_info("final_loss: {:.5}".format(loss.avg))

        # Save results
        with open(self.cluster_path / "cluster_counts.tsv", mode="w") as f:
            f.write("\t".join([str(k) for k in range(self.n_prototypes)]) + "\n")
            f.write(
                "\t".join([str(averages[k].count) for k in range(self.n_prototypes)])
                + "\n"
            )
        for k in range(self.n_prototypes):
            indices = np.where(cluster_idx == k)[0]
            top_idx = np.argsort(distances[indices])[:N_CLUSTER_SAMPLES]
            for j, idx in enumerate(top_idx):
                inp = dataset[indices[idx]][0].unsqueeze(0).to(self.device)
                convert_to_img(inp).save(self.cluster_path / f"cluster{k}" / f"top{j}_raw.png")
                if not self.model.transformer.is_identity:
                    convert_to_img(self.model.transform(inp)[0, k]).save(
                        self.cluster_path / f"cluster{k}" / f"top{j}_tsf.png"
                    )
            if len(indices) <= N_CLUSTER_SAMPLES:
                random_idx = indices
            else:
                random_idx = np.random.choice(indices, N_CLUSTER_SAMPLES, replace=False)
            for j, idx in enumerate(random_idx):
                inp = dataset[idx][0].unsqueeze(0).to(self.device)
                convert_to_img(inp).save(self.cluster_path / f"cluster{k}" / f"random{j}_raw.png")
                if not self.model.transformer.is_identity:
                    convert_to_img(self.model.transform(inp)[0, k]).save(
                        self.cluster_path / f"cluster{k}" / f"random{j}_tsf.png"
                    )
            try:
                convert_to_img(averages[k].avg).save(self.cluster_path / f"cluster{k}" / "avg.png")
            except AssertionError:
                print_warning(f"no image found in cluster {k}")

    @torch.no_grad()
    def quantitative_eval(self):
        """Routine to save quantitative results: loss + scores"""
        loss = AverageMeter()
        scores_path = self.run_dir / FINAL_SCORES_FILE
        scores = Scores(self.n_classes, self.n_prototypes)
        with open(scores_path, mode="w") as f:
            f.write("loss\t" + "\t".join(scores.names) + "\n")

        dataset = get_dataset(self.dataset_name)(
            "train", eval_mode=True, **self.dataset_kwargs
        )
        loader = DataLoader(
            dataset, batch_size=self.batch_size, num_workers=self.n_workers
        )
        for images, labels, _, _ in loader:
            images = images.to(self.device)
            if hasattr(self.model, "proba"):
                _, out, probas = self.model(images)
                argmin_idx = probas.argmax(1)
                dist_min_by_sample = out[:, argmin_idx]
                hist, _ = np.histogram(probas.cpu().numpy(), bins=self.bin_edges)
                self.bin_counts += hist
            else:
                distances = self.model(images)[1]
                dist_min_by_sample, argmin_idx = distances.min(1)

            loss.update(dist_min_by_sample.mean(), n=len(dist_min_by_sample))
            scores.update(labels.long().numpy(), argmin_idx.cpu().numpy())

        scores = scores.compute()
        self.print_and_log_info("bin_counts: " + str(self.bin_counts))
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


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    run_trainer(
        cfg=cfg,
        trainer_class=Trainer,
    )

if __name__ == "__main__":
    main()
