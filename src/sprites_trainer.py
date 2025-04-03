import sys
import os
import shutil
import datetime
import time
import argparse
import traceback

import seaborn as sns
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from .abstract_trainer import AbstractTrainer, PRINT_CHECK_CLUSTERS_FMT

try:
    import visdom
except ModuleNotFoundError:
    pass

from .dataset import get_dataset
from .model import get_model
from .model.tools import safe_model_state_dict
from .optimizer import get_optimizer
from .utils import (
    use_seed,
    coerce_to_path_and_check_exist,
    coerce_to_path_and_create_dir,
)
from .utils.image import convert_to_img, save_gif
from .utils.logger import print_warning
from .utils.metrics import (
    AverageTensorMeter,
    AverageMeter,
    Scores,
    SegmentationScores,
    InstanceSegScores,
)
from .utils.path import RUNS_PATH
from .utils.consts import *

from PIL import ImageDraw

from fvcore.nn import FlopCountAnalysis


class Trainer(AbstractTrainer):
    """Pipeline to train a NN model using a certain dataset, both specified by an YML config."""

    model_name = "dti_sprites"
    interpolate_settings = {
        'mode': 'bilinear',
        'align_corners': False
    }

    n_backgrounds = None
    n_objects = None
    pred_class = None
    n_clusters = None
    learn_masks = None
    learn_backgrounds = None
    learn_proba = None

    masked_prototypes_path = None
    masks_path = None
    backgrounds_path = None

    eval_semantic = None
    eval_qualitative = None
    eval_with_bkg = None

    @use_seed()
    def __init__(self, cfg, run_dir, save=False):
        super().__init__(cfg, run_dir, save)

    ######################
    #   SETUP METHODS    #
    ######################

    def setup_dataset(self):
        """Set up dataset parameters and load dataset."""
        super().setup_dataset()
        self.seg_eval = getattr(self.train_dataset, "seg_eval", False)
        self.instance_eval = getattr(self.train_dataset, "instance_eval", False)

    def get_model(self):
        """Return model instance."""
        return get_model(self.model_name)(
            n_epochs=self.n_epochs,
            dataset=self.train_loader.dataset,
            **self.model_kwargs
        )

    def setup_model(self):
        """Initialize model architecture."""
        super().setup_model()
        self.n_backgrounds = getattr(self.model, "n_backgrounds", 0)
        self.n_objects = max(self.model.n_objects, 1)
        self.pred_class = getattr(self.model, "pred_class", False) or getattr(
            self.model, "estimate_minimum", False
        )

        # Calculate number of clusters
        if self.pred_class:
            self.n_clusters = self.n_prototypes * self.n_objects
        else:
            self.n_clusters = self.n_prototypes ** self.n_objects * max(
                self.n_backgrounds, 1
            )

        # Additional sprite model properties
        self.learn_masks = getattr(self.model, "learn_masks", False)
        self.learn_backgrounds = getattr(self.model, "learn_backgrounds", False)
        self.learn_proba = getattr(self.model, "proba", False)

    def setup_directories(self):
        super().setup_directories()

        if not self.save_img:
            return

        if self.learn_masks:
            self.masked_prototypes_path = coerce_to_path_and_create_dir(self.run_dir / "masked_prototypes")
            self.masks_path = coerce_to_path_and_create_dir(self.run_dir / "masks")
            for k in range(self.n_prototypes):
                coerce_to_path_and_create_dir(self.masked_prototypes_path / f"proto{k}")
                coerce_to_path_and_create_dir(self.masks_path / f"mask{k}")

                for j in range(self.images_to_tsf.size(0)):
                    img_path = self.transformation_path / f"img{j}"
                    coerce_to_path_and_create_dir(img_path / f"frg_tsf{k}")
                    coerce_to_path_and_create_dir(img_path / f"mask_tsf{k}")

        if self.learn_backgrounds:
            self.backgrounds_path = coerce_to_path_and_create_dir(self.run_dir / "backgrounds")
            for k in range(self.n_backgrounds):
                coerce_to_path_and_create_dir(self.backgrounds_path / f"bkg{k}")

                for j in range(self.images_to_tsf.size(0)):
                    img_path = self.transformation_path / f"img{j}"
                    coerce_to_path_and_create_dir(img_path / f"bkg_tsf{k}")

    def setup_optimizer(self):
        """Configure optimizer for Sprites model."""
        # Extract optimizer parameters
        opt_params = self.cfg["training"]["optimizer"] or {}
        optimizer_name = self.cfg["training"]["optimizer_name"]
        cluster_kwargs = self.cfg["training"].get("cluster_optimizer", {})
        tsf_kwargs = self.cfg["training"]["transformer_optimizer"] or {}

        # Create optimizer with multiple parameter groups
        self.optimizer = get_optimizer(optimizer_name)(
            [
                dict(params=self.model.cluster_parameters(), **cluster_kwargs),
                dict(params=self.model.transformer_parameters(), **tsf_kwargs),
            ],
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
        metric_names = ["time/img", "loss_rec", "loss_em", "loss_bin", "loss_freq"]
        metric_names += [f"prop_clus{i}" for i in range(self.n_clusters)]
        return metric_names

    def setup_val_scores(self):
        self.eval_semantic = self.cfg["training"].get("eval_semantic", False)
        self.eval_qualitative = self.cfg["training"].get("eval_qualitative", False)
        self.eval_with_bkg = self.cfg["training"].get("eval_with_bkg", False)

        # Create appropriate score tracker based on evaluation mode
        if hasattr(self, 'seg_eval') and self.seg_eval:
            self.val_scores = SegmentationScores(self.n_classes)
        elif hasattr(self, 'instance_eval') and self.instance_eval:
            self.val_scores = InstanceSegScores(
                self.n_objects + 1, with_bkg=self.eval_with_bkg
            )
        else:
            self.val_scores = Scores(self.n_classes, self.n_prototypes)

    def setup_prototypes(self):
        super().save_prototypes()
        self.check_cluster_interval = self.cfg["training"]["check_cluster_interval"]
        if not self.save_img:
            return
        for k in range(self.images_to_tsf.size(0)):
            convert_to_img(self.images_to_tsf[k]).save(self.transformation_path / f"img{k}" / "input.png")

    def setup_visualizer(self, *args, **kwargs):
        """Set up real-time visualization (e.g., Visdom)."""
        pass

    def setup_additional_components(self, *args, **kwargs):
        """Set up any additional trainer-specific components."""
        pass

    ######################
    #    MAIN METHODS    #
    ######################

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
            self.model.module.load_state_dict(state, dataset=self.train_loader.dataset)
        self.start_epoch, self.start_batch = 1, 1
        if resume:
            self.start_epoch, self.start_batch = (
                checkpoint["epoch"],
                checkpoint.get("batch", 0) + 1,
            )
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state"])
            self.cur_lr = self.scheduler.get_last_lr()[0]
        if hasattr(self.model, "cur_epoch"):
            self.model.cur_epoch = checkpoint["epoch"]
        self.print_and_log_info(
            "Checkpoint loaded at epoch {}, batch {}".format(
                self.start_epoch, self.start_batch - 1
            )
        )
        self.print_and_log_info("LR = {}".format(self.cur_lr))

    @use_seed()
    def run(self):
        cur_iter = (self.start_epoch - 1) * self.n_batches + self.start_batch - 1
        prev_train_stat_iter, prev_val_stat_iter = cur_iter, cur_iter
        prev_check_cluster_iter = cur_iter
        if self.start_epoch == self.n_epochs:
            self.print_and_log_info("No training, only evaluating")
            self.evaluate()
            self.save_metric_plots()
            self.print_and_log_info("Training run is over")
            return
        for epoch in range(self.start_epoch, self.n_epochs + 1):
            batch_start = self.start_batch if epoch == self.start_epoch else 1
            for batch, (images, _, img_masks, _) in enumerate(
                    self.train_loader, start=1
            ):
                if batch < batch_start:
                    continue
                cur_iter += 1
                if cur_iter > self.n_iterations:
                    break

                self.single_train_batch_run(images, img_masks)
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
                    self.save(epoch=epoch, batch=batch)
                    if self.save_img:
                        self.log_images(cur_iter)

            self.model.step()
            if self.scheduler_update_range == "epoch" and batch_start == 1:
                self.update_scheduler(epoch + 1, batch=1)

        self.save(epoch=epoch, batch=batch)
        self.save_metric_plots()
        self.evaluate()
        self.print_and_log_info("Training run is over")

    def single_train_batch_run(self, images, masks):
        start_time = time.time()
        B = images.size(0)
        self.model.train()
        images = images.to(self.device)
        if masks != []:
            masks = masks.to(self.device)
        else:
            masks = None
        self.optimizer.zero_grad()

        loss, distances, class_prob = self.model(images, masks)
        loss[0].backward()
        self.optimizer.step()

        with torch.no_grad():
            if self.learn_proba:
                class_oh = torch.zeros(class_prob.shape, device=class_prob.device).scatter_(
                    1, class_prob.argmax(1, keepdim=True), 1
                )
                one_hot = class_oh.permute(2, 0, 1).flatten(1)  # B(L*K)
                proportions = one_hot.mean(0)
            else:
                if self.pred_class:  # distances B(L*K), discovery
                    proportions = (1 - distances).mean(0)
                else:  # distances B(K**L*M), clustering
                    argmin_idx = distances.argmin(1)
                    one_hot = torch.zeros(
                        B, distances.size(1), device=self.device
                    ).scatter(1, argmin_idx[:, None], 1)
                    proportions = one_hot.sum(0) / B

        self.train_metrics.update(
            {
                "time/img": (time.time() - start_time) / B,
                "loss_rec": loss[1].item(),
                "loss_em": loss[4].item(),
                "loss_bin": loss[2].item(),
                "loss_freq": loss[3].item(),
            }
        )
        self.train_metrics.update(
            {f"prop_clus{i}": p.item() for i, p in enumerate(proportions)}
        )

    ######################
    #   SAVING METHODS   #
    ######################

    @torch.no_grad()
    def save_masked_prototypes(self, cur_iter=None):
        self.save_pred(
            cur_iter,
            pred_name="prototype",
            transform_fn=lambda proto, k: proto * self.model.masks[k],
            prefix="proto"
        )

    @torch.no_grad()
    def save_masks(self, cur_iter=None):
        self.save_pred(
            cur_iter=cur_iter,
            pred_name="mask",
            n_preds=self.n_prototypes
        )

    @torch.no_grad()
    def save_backgrounds(self, cur_iter=None):
        self.save_pred(cur_iter, pred_name="background", prefix="bkg")

    @torch.no_grad()
    def save_transformed_images(self, cur_iter=None):
        self.model.eval()
        if self.learn_masks:
            output, compositions, _ = self.model.transform(
                self.images_to_tsf, with_composition=True
            )
        else:
            output, compositions = self.model.transform(self.images_to_tsf), []

        transformed_imgs = torch.cat([self.images_to_tsf.unsqueeze(1), output], 1)
        N = self.get_n_clusters()
        transformed_imgs = transformed_imgs[:, : N + 1]
        for k in range(transformed_imgs.size(0)):
            for j, img in enumerate(transformed_imgs[k][1:]):
                tsf_path = self.transformation_path / f"img{k}"
                if cur_iter is not None:
                    self.save_img_to_path(img, tsf_path / f"tsf{j}", f"{cur_iter}.jpg")
                else:
                    self.save_img_to_path(img, tsf_path, f"tsf{j}.png")

        i = 0
        for name in ["frg", "mask", "bkg", "frg_aux", "mask_aux"]:
            if name == "bkg" and not self.learn_backgrounds:
                continue
            if i == len(compositions):
                break

            layer = compositions[i].expand(-1, -1, self.images_to_tsf.size(1), -1, -1)
            compositions[i] = torch.cat([self.images_to_tsf.unsqueeze(1), layer], 1)
            if name in ["frg", "mask", "bkg"]:
                for k in range(transformed_imgs.size(0)):
                    tmp_path = self.transformation_path / f"img{k}"
                    for j, img in enumerate(compositions[i][k][1:]):
                        if cur_iter is not None:
                            self.save_img_to_path(img, tmp_path / f"{name}_tsf{j}", f"{cur_iter}.jpg")
                        else:
                            self.save_img_to_path(img, tmp_path, f"{name}_tsf{j}.png")
            i += 1

        return transformed_imgs, compositions

    def _save_additional_image_gifs(self, size):
        """Save additional image GIFs specific to Sprites trainer."""
        if hasattr(self, 'learn_masks') and self.learn_masks:
            if hasattr(self, 'masked_prototypes_path') and os.path.exists(self.masked_prototypes_path):
                for k in range(self.n_prototypes):
                    self.save_gif_to_path(self.masked_prototypes_path / f"proto{k}", f"prototype{k}.gif", size=size)

            if hasattr(self, 'masks_path') and os.path.exists(self.masks_path):
                for k in range(self.n_prototypes):
                    self.save_gif_to_path(self.masks_path / f"mask{k}", f"mask{k}.gif", size=size)

            for i in range(self.images_to_tsf.size(0)):
                for k in range(self.n_prototypes):
                    for component in ["frg", "mask"]:
                        component_path = self.transformation_path / f"img{i}" / f"{component}_tsf{k}"
                        self.save_gif_to_path(component_path, f"{component}_tsf{k}.gif", size=size)

        if hasattr(self, 'learn_backgrounds') and self.learn_backgrounds:
            if hasattr(self, 'backgrounds_path') and os.path.exists(self.backgrounds_path):
                for k in range(self.n_backgrounds):
                    self.save_gif_to_path(self.backgrounds_path / f"bkg{k}", f"background{k}.gif", size=size)

            for i in range(self.images_to_tsf.size(0)):
                for k in range(self.n_backgrounds):
                    bkg_tsf_path = self.transformation_path / f"img{i}" / f"bkg_tsf{k}"
                    self.save_gif_to_path(bkg_tsf_path, f"bkg_tsf{k}.gif", size=size)

    ######################
    #   LOGGING METHODS  #
    ######################

    def _log_model_specific_images(self, cur_iter):
        """Visualize masks, masked prototypes, and backgrounds if applicable."""
        if self.learn_masks:
            self.save_masked_prototypes(cur_iter)
            self.update_visualizer_images(
                self.model.prototypes * self.model.masks, "masked_prototypes", nrow=5
            )
            self.save_masks(cur_iter)
            self.update_visualizer_images(self.model.masks, "masks", nrow=5)

        if self.learn_backgrounds:
            self.save_backgrounds(cur_iter)
            self.update_visualizer_images(self.model.backgrounds, "backgrounds", nrow=5)

    def _log_transformation_compositions(self, compositions, C, H, W):
        if len(compositions) > 0:
            k = 0
            # Visualize foreground and mask transformations
            for imgs, name in zip(compositions[:2], ["frg_tsf", "mask_tsf"]):
                self.update_visualizer_images(
                    imgs.view(-1, imgs.size(2), H, W), name, nrow=self.n_prototypes + 1
                )
                k += 1

            # Visualize background transformations
            if self.learn_backgrounds:
                imgs = compositions[k]
                self.update_visualizer_images(
                    imgs.view(-1, imgs.size(2), H, W),
                    "bkg_tsf",
                    nrow=self.n_backgrounds + 1,
                )
                k += 1

            # Visualize auxiliary transformations for multi-object models
            if self.n_objects > 1:
                for name in ["frg_tsf_aux", "mask_tsf_aux"]:
                    imgs = compositions[k]
                    self.update_visualizer_images(
                        imgs.view(-1, imgs.size(2), H, W),
                        name,
                        nrow=self.n_prototypes + 1,
                    )
                    k += 1

    def get_transformation_nrow(self, tsf_imgs):
        """Custom row count for transformation visualization."""
        return tsf_imgs.size(1)

    ######################
    # VALIDATION METHODS #
    ######################

    def check_cluster(self, cur_iter, epoch, batch):
        if hasattr(self.model, "_diff_selections") and self.visualizer is not None:
            diff = self.model._diff_selections
            x, y = [[cur_iter] * len(diff[0])], [diff[1]]
            self.visualizer.line(
                y,
                x,
                win="diff selection",
                update="append",
                opts=dict(
                    title="diff selection",
                    legend=diff[0],
                    width=VIZ_WIDTH,
                    height=VIZ_HEIGHT,
                ),
            )

        proportions = torch.Tensor(
            [self.train_metrics[f"prop_clus{i}"].avg for i in range(self.n_clusters)]
        )
        if self.n_backgrounds > 1:
            proportions = proportions.view(self.n_prototypes, self.n_backgrounds)
            for axis, is_bkg in zip([1, 0], [False, True]):
                prop = proportions.sum(axis)
                reassigned, idx = self.model.reassign_empty_clusters(
                    prop, is_background=is_bkg
                )
                msg = PRINT_CHECK_CLUSTERS_FMT(
                    epoch, self.n_epochs, batch, self.n_batches, reassigned, idx
                )
                if is_bkg:
                    msg += " for backgrounds"
                self.print_and_log_info(msg)
                self.print_and_log_info(
                    ", ".join(
                        ["prop_{}={:.4f}".format(k, prop[k]) for k in range(len(prop))]
                    )
                )
        elif self.n_objects > 1:
            k = np.random.randint(0, self.n_objects)
            if self.n_clusters == self.n_prototypes ** self.n_objects:
                prop = (
                    proportions.view((self.n_prototypes,) * self.n_objects)
                    .transpose(0, k)
                    .flatten(1)
                    .sum(1)
                )
            else:
                prop = proportions.view(self.n_objects, self.n_prototypes)[k]
            reassigned, idx = self.model.reassign_empty_clusters(prop)
            msg = PRINT_CHECK_CLUSTERS_FMT(
                epoch, self.n_epochs, batch, self.n_batches, reassigned, idx
            )
            msg += f" for object layer {k}"
            self.print_and_log_info(msg)
            self.print_and_log_info(
                ", ".join(
                    ["prop_{}={:.4f}".format(k, prop[k]) for k in range(len(prop))]
                )
            )
        else:
            reassigned, idx = self.model.reassign_empty_clusters(proportions)
            msg = PRINT_CHECK_CLUSTERS_FMT(
                epoch, self.n_epochs, batch, self.n_batches, reassigned, idx
            )
            self.print_and_log_info(msg)
        self.train_metrics.reset(*[f"prop_clus{i}" for i in range(self.n_clusters)])

    @torch.no_grad()
    def run_val(self):
        """Run validation step for current model."""
        self.model.eval()

        for images, labels, _, _ in self.val_loader:
            B, C, H, W = images.shape
            images = images.to(self.device)

            # Get model outputs
            loss_val, distances, class_prob = self.model(images)
            self.val_metrics.update({"loss_val": loss_val[0].item()})

            # Clustering
            if self.n_backgrounds > 1 and not self.pred_class:
                assert class_prob is None
                distances, _ = distances.view(B, self.n_prototypes, self.n_backgrounds).min(2)

            # Multi-object
            other_idxs = []
            if self.n_objects > 1 and not self.pred_class:
                assert class_prob is None
                distances = distances.view(B, *(self.n_prototypes,) * self.n_objects)
                for k in range(self.n_objects, 1, -1):
                    distances, idx = distances.min(k)
                    other_idxs.insert(0, idx)

            if self.learn_proba and class_prob is not None:
                class_oh = class_prob.permute(2, 0, 1).flatten(1)
                argmin_idx = class_oh.argmax(1)
            else:
                argmin_idx = distances.argmin(1)

            if self.seg_eval:
                self.evaluate_segmentation(images, labels, argmin_idx, other_idxs, B, H, W)
            elif self.instance_eval:
                self.evaluate_instance(images, labels, argmin_idx, other_idxs, B)
            else:
                assert self.n_objects == 1
                self.val_scores.update(labels.long().numpy(), argmin_idx.cpu().numpy())

    ######################
    # EVALUATION METHODS #
    ######################

    def evaluate(self):
        print(f"TAU: {self.model.tau}")
        self.model.eval()
        label = self.train_loader.dataset[0][1]
        empty_label = isinstance(label, (int, np.integer)) and label == -1
        if empty_label:
            if self.seg_eval:
                self.segmentation_qualitative_eval()
            elif self.instance_eval:
                self.instance_seg_qualitative_eval()
            else:
                self.qualitative_eval()
        elif self.seg_eval or self.instance_eval:
            if (self.seg_eval and self.learn_masks) or self.eval_semantic:
                self.print_and_log_info("Semantic segmentation evaluation")
                self.segmentation_quantitative_eval()
                self.segmentation_qualitative_eval()
            elif self.instance_eval and self.learn_masks:  # NOTE: evaluate either semantic or instance
                self.print_and_log_info("Instance segmentation evaluation")
                self.instance_seg_quantitative_eval()
                self.instance_seg_qualitative_eval()
        else:
            self.quantitative_eval()
            if self.eval_qualitative:
                self.qualitative_eval()

        self.print_and_log_info("Evaluation is over")

    def evaluate_segmentation(self, images, labels, argmin_idx, other_idxs, B, H, W):
        """Handle semantic segmentation evaluation."""
        if self.n_objects == 1:
            masks = self.model.transform(images, with_composition=True)[1][1]
            masks = masks[torch.arange(B), argmin_idx]
            self.val_scores.update(labels.long().numpy(), (masks > 0.5).long().cpu().numpy())
            return

        target = self.model.transform(images, pred_semantic_labels=True).cpu()

        if not self.pred_class:
            target = target.view(B, *(self.n_prototypes,) * self.n_objects, H, W)
            real_idxs = []
            for idx in [argmin_idx] + other_idxs:
                for i in real_idxs:
                    idx = idx[torch.arange(B), i]
                real_idxs.insert(0, idx)
                target = target[torch.arange(B), idx]

        self.val_scores.update(labels.long().numpy(), target.long().cpu().numpy())

    def evaluate_instance(self, images, labels, argmin_idx, other_idxs, B):
        """Handle instance segmentation evaluation."""
        if self.n_objects == 1:
            # Single object case
            masks = self.model.transform(images, with_composition=True)[1][1]
            self.val_scores.update(labels.long().numpy(), (masks > 0.5).long().cpu().numpy())
            return

        target = self.model.transform(images, pred_instance_labels=True, with_bkg=self.eval_with_bkg).cpu()

        if not self.pred_class:
            target = target.view(B, *(self.n_prototypes,) * self.n_objects, images.size(2), images.size(3))
            real_idxs = []
            for idx in [argmin_idx] + other_idxs:
                for i in real_idxs:
                    idx = idx[torch.arange(B), i]
                real_idxs.insert(0, idx)
                target = target[torch.arange(B), idx]

            if not self.eval_with_bkg:
                bkg_idx = target == 0
                tsf_layers = self.model.predict(images)[0]
                new_target = ((tsf_layers - images) ** 2).sum(3).min(1)[0].argmin(0).long() + 1
                target[bkg_idx] = new_target[bkg_idx]

        self.val_scores.update(labels.long().numpy(), target.long().numpy())

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
        for images, _, _, path in train_loader:
            images = images.to(self.device)
            dist = self.model(images)[1]
            dist_min_by_sample, argmin_idx = map(lambda t: t.cpu().numpy(), dist.min(1))
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

        # Compute results
        distances, cluster_idx = np.array([]), np.array([], dtype=np.int32)
        averages = {k: AverageTensorMeter() for k in range(self.n_prototypes)}
        cluster_by_path = []
        for images, _, _, path in train_loader:
            images = images.to(self.device)
            _, dist, class_prob = self.model(images)
            if self.n_backgrounds > 1:
                assert class_prob is None
                dist = dist.view(
                    images.size(0), self.n_prototypes, self.n_backgrounds
                ).min(2)[0]

            dist_min_by_sample, argmin_idx = map(lambda t: t.cpu().numpy(), dist.min(1))
            if self.learn_proba:
                class_oh = class_prob.permute(2, 0, 1).flatten(1)
                argmin_idx = class_oh.argmax(1).cpu().numpy()
                hist, _ = np.histogram(class_prob.cpu().numpy(), bins=self.bin_edges)
                self.bin_counts += hist

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

        # Save cluster_by_path as csv
        if cluster_by_path:
            cluster_by_path = pd.DataFrame(
                cluster_by_path, columns=["path", "cluster_id"]
            ).set_index("path")
            cluster_by_path.to_csv(self.run_dir / "cluster_by_path.csv")

        self.print_and_log_info("bin_counts: " + str(self.bin_counts))
        self.print_and_log_info("final_loss: {:.5}".format(float(loss.avg)))

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
                convert_to_img(self.model.transform(inp)[0, k]).save(
                    path / f"top{j}_tsf.png"
                )
            if len(indices) <= N_CLUSTER_SAMPLES:
                random_idx = indices
            else:
                random_idx = np.random.choice(indices, N_CLUSTER_SAMPLES, replace=False)
            for j, idx in enumerate(random_idx):
                inp = dataset[idx][0].unsqueeze(0).to(self.device)
            try:
                convert_to_img(averages[k].avg).save(path / "avg.png")
            except AssertionError:
                print_warning(f"no image found in cluster {k}")

    @torch.no_grad()
    def segmentation_quantitative_eval(self):
        """Run and save evaluation for semantic segmentation"""
        dataset = get_dataset(self.dataset_name)(
            "train", eval_mode=True, eval_semantic=True, **self.dataset_kwargs
        )
        train_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.n_workers,
            shuffle=False,
        )
        loss = AverageMeter()
        scores_path = self.run_dir / FINAL_SEMANTIC_SCORES_FILE
        scores = SegmentationScores(self.n_classes)
        with open(scores_path, mode="w") as f:
            f.write("loss\t" + "\t".join(scores.names) + "\n")

        for images, labels, _, _ in train_loader:
            images = images.to(self.device)
            loss_val, distances, class_prob = self.model(images)
            hist, _ = np.histogram(class_prob.cpu().numpy(), bins=self.bin_edges)
            self.bin_counts += hist
            B, C, H, W = images.shape
            if self.n_objects == 1:
                masks = self.model.transform(images, with_composition=True)[1][1]
                if masks.size(1) > 1:
                    if self.learn_proba:
                        class_oh = class_prob.permute(2, 0, 1).flatten(1)
                        argmin_idx = class_oh.argmax(1)
                    else:
                        argmin_idx = distances.argmin(1)
                    masks = masks[torch.arange(B), argmin_idx]
                scores.update(labels.long().numpy(), (masks > 0.5).long().cpu().numpy())
            else:
                if self.pred_class:
                    target = self.model.transform(
                        images, pred_semantic_labels=True
                    ).cpu()
                    scores.update(labels.long().numpy(), target.long().numpy())
                else:
                    assert class_prob is None
                    distances = distances.view(
                        B, *(self.n_prototypes,) * self.n_objects
                    )
                    other_idxs = []
                    for k in range(self.n_objects, 1, -1):
                        distances, idx = distances.min(k)
                        other_idxs.insert(0, idx)
                    argmin_idx = distances.argmin(1)

                    target = self.model.transform(
                        images, pred_semantic_labels=True
                    ).cpu()
                    target = target.view(
                        B, *(self.n_prototypes,) * self.n_objects, H, W
                    )
                    real_idxs = []
                    for idx in [argmin_idx] + other_idxs:
                        for i in real_idxs:
                            idx = idx[torch.arange(B), i]
                        real_idxs.insert(0, idx)
                        target = target[torch.arange(B), idx]
                    scores.update(labels.long().numpy(), target.long().cpu().numpy())

            loss.update(loss_val[0].item(), n=images.size(0))

        flops = FlopCountAnalysis(self.model, images)
        self.print_and_log_info("flops:" + str(flops.total()))

        scores = scores.compute()
        self.print_and_log_info("bin_counts: " + str(self.bin_counts))
        self.print_and_log_info("final_loss: {:.4f}".format(float(loss.avg)))
        self.print_and_log_info(
            "final_scores: "
            + ", ".join(["{}={:.4f}".format(k, v) for k, v in scores.items()])
        )
        with open(scores_path, mode="a") as f:
            f.write(
                "{:.6}\t".format(float(loss.avg))
                + "\t".join(map("{:.6f}".format, scores.values()))
                + "\n"
            )

    @torch.no_grad()
    def segmentation_qualitative_eval(self):
        """Run and save qualitative evaluation for semantic segmentation"""
        out = coerce_to_path_and_create_dir(self.run_dir / "semantic_seg")
        K = self.n_prototypes if self.model.add_empty_sprite else self.n_prototypes + 1
        colors = sns.color_palette("hls", K)
        colors[0] = tuple((np.asarray(colors[0]) / colors[0][0]) * 0.5)
        dataset = self.train_loader.dataset
        if 32 % self.batch_size == 0:
            N, B = 32 // self.batch_size, self.batch_size
        else:
            N, B = 8, 4
        C, H, W = dataset[0][0].shape
        train_loader = DataLoader(
            dataset, batch_size=B, num_workers=self.n_workers, shuffle=False
        )

        iterator = iter(train_loader)
        for j in range(N):
            images, _, _, _ = next(iterator)
            images = images.to(self.device)
            if self.pred_class:
                recons, composition, class_prob = self.model.transform(images, hard_occ_grid=False,
                                                                       with_composition=True)
                class_prob = class_prob.permute(2, 0, 1)
                if len(composition) % 2 != 0:
                    print("Omitting background from compositions")
                    composition = composition[:2] + composition[3:]
                n_layers = int(len(composition) / 2)
                for b in range(B):
                    for l in range(0, n_layers):
                        tsf_layers, tsf_masks = composition[l * 2], composition[l * 2 + 1]
                        _, K, _, _, _ = tsf_layers.shape
                        for k in range(K):
                            name = f"image_{b}_layer_{l}_sprite_{k}"
                            convert_to_img(tsf_layers[b, k, ...]).save(out / f"frg_rec_{name}.png")
                            weight = class_prob[b, l, k].item()
                            img_temp = convert_to_img(tsf_masks[b, k, ...] * weight)
                            draw = ImageDraw.Draw(img_temp)
                            draw.text((0, 0), "%.3f" % weight, (255, 255, 255))
                            img_temp.save(out / f"mask_rec_{name}.png")
                        construction = torch.sum(
                            tsf_layers[b] * tsf_masks[b] * class_prob[b, l, :][:, None, None, None], dim=0)
                        convert_to_img(construction).save(out / f"rec_image_{b}_layer_{l}.png")

                recons = recons.cpu()
                if self.model.return_map_out:
                    (infer_seg, bboxes, class_ids) = self.model.transform(
                        images, pred_semantic_labels=True
                    )
                    infer_seg = infer_seg.cpu()
                else:
                    infer_seg = self.model.transform(
                        images, pred_semantic_labels=True
                    ).cpu()

            else:
                # TODO: proba segmentation
                exit("Not implemented")
                _, distances, class_prob = self.model(images)
                distances = distances.view(B, *(self.n_prototypes,) * self.n_objects)
                other_idxs = []
                for k in range(self.n_objects, 1, -1):
                    distances, idx = distances.min(k)
                    other_idxs.insert(0, idx)
                dist_min_by_sample, argmin_idx = distances.min(1)

                recons = self.model.transform(images).cpu()
                recons = recons.view(B, *(self.n_prototypes,) * self.n_objects, C, H, W)
                infer_seg = self.model.transform(
                    images, pred_semantic_labels=True
                ).cpu()
                infer_seg = infer_seg.view(
                    B, *(self.n_prototypes,) * self.n_objects, H, W
                )
                real_idxs = []
                for idx in [argmin_idx] + other_idxs:
                    for i in real_idxs:
                        idx = idx[torch.arange(B), i]
                    real_idxs.insert(0, idx)
                    infer_seg = infer_seg[torch.arange(B), idx]
                    recons = recons[torch.arange(B), idx]

            infer_seg = infer_seg.unsqueeze(1).expand(-1, C, H, W)
            color_seg = torch.zeros(infer_seg.shape).float()
            masks = []
            for k, col in enumerate(colors):
                masks.append(infer_seg == k)
                color_seg[masks[-1]] = (
                    torch.Tensor(col)[None, :, None, None]
                    .to("cpu")
                    .expand(B, C, H, W)[masks[-1]]
                )

            images = images.cpu()
            for k in range(B):
                name = f"{k + j * B}".zfill(2)
                convert_to_img(images[k]).save(out / f"{name}.png")
                convert_to_img(recons[k]).save(out / f"{name}_recons.png")
                convert_to_img(color_seg[k]).save(out / f"{name}_seg_full.png")

    @torch.no_grad()
    def instance_seg_quantitative_eval(self):
        """Run and save quantitative evaluation for instance segmentation"""
        dataset = get_dataset(self.dataset_name)(
            "train", eval_mode=True, **self.dataset_kwargs
        )
        if 320 % self.batch_size == 0:
            N, B = 320 // self.batch_size, self.batch_size
        else:
            N, B = 80, 4
        train_loader = DataLoader(
            dataset, batch_size=B, num_workers=self.n_workers, shuffle=False
        )
        loss = AverageMeter()
        scores_path = self.run_dir / FINAL_SEG_SCORES_FILE
        scores = InstanceSegScores(self.n_objects + 1, with_bkg=self.eval_with_bkg)
        with open(scores_path, mode="w") as f:
            f.write("loss\t" + "\t".join(scores.names) + "\n")

        iterator = iter(train_loader)
        for k in range(N):
            images, labels, _, _ = next(iterator)
            images = images.to(self.device)
            loss_val, distances, class_prob = self.model(images)
            if self.learn_proba:
                class_oh = class_prob.permute(2, 0, 1).flatten(1)
                argmin_idx = class_oh.argmax(1).cpu().numpy()
                hist, _ = np.histogram(class_prob.cpu().numpy(), bins=self.bin_edges)
                self.bin_counts += hist
            if self.n_objects == 1:
                masks = self.model.transform(images, with_composition=True)[1][1]
                scores.update(labels.long().numpy(), (masks > 0.5).long().cpu().numpy())
            else:
                if self.pred_class:
                    target = self.model.transform(
                        images, pred_instance_labels=True, with_bkg=self.eval_with_bkg
                    ).cpu()
                    scores.update(labels.long().numpy(), target.long().numpy())

                else:
                    # TODO: proba segmentation
                    exit("Not implemented")
                    distances = distances.view(
                        B, *(self.n_prototypes,) * self.n_objects
                    )
                    other_idxs = []
                    for k in range(self.n_objects, 1, -1):
                        distances, idx = distances.min(k)
                        other_idxs.insert(0, idx)
                    dist_min_by_sample, argmin_idx = distances.min(1)

                    target = self.model.transform(
                        images, pred_instance_labels=True, with_bkg=self.eval_with_bkg
                    ).cpu()
                    target = target.view(
                        B,
                        *(self.n_prototypes,) * self.n_objects,
                        images.size(2),
                        images.size(3),
                    )
                    real_idxs = []
                    for idx in [argmin_idx] + other_idxs:
                        for i in real_idxs:
                            idx = idx[torch.arange(B), i]
                        real_idxs.insert(0, idx)
                        target = target[torch.arange(B), idx]
                    if not self.eval_with_bkg:
                        bkg_idx = target == 0
                        tsf_layers = self.model.predict(images)[0]
                        new_target = (
                                ((tsf_layers - images) ** 2)
                                .sum(3)
                                .min(1)[0]
                                .argmin(0)
                                .long()
                                + 1
                        ).cpu()
                        target[bkg_idx] = new_target[bkg_idx]
                    scores.update(labels.long().numpy(), target.long().numpy())

            loss.update(loss_val[0].item(), n=images.size(0))

        scores = scores.compute()
        self.print_and_log_info("bin_counts: " + str(self.bin_counts))
        self.print_and_log_info("final_loss: {:.4f}".format(float(loss.avg)))
        self.print_and_log_info(
            "final_scores: "
            + ", ".join(["{}={:.4f}".format(k, v) for k, v in scores.items()])
        )
        with open(scores_path, mode="a") as f:
            f.write(
                "{:.6}\t".format(loss.avg)
                + "\t".join(map("{:.6f}".format, scores.values()))
                + "\n"
            )

    @torch.no_grad()
    def instance_seg_qualitative_eval(self):
        """Run and save qualitative evaluation for instance segmentation"""
        out = coerce_to_path_and_create_dir(self.run_dir / "instance_seg")
        colors = sns.color_palette("tab10", self.n_objects + 1)
        dataset = self.train_loader.dataset
        if 32 % self.batch_size == 0:
            N, B = 32 // self.batch_size, self.batch_size
        else:
            N, B = 8, 4
        C, H, W = dataset[0][0].shape
        train_loader = DataLoader(
            dataset, batch_size=B, num_workers=self.n_workers, shuffle=False
        )

        iterator = iter(train_loader)
        for j in range(N):
            images, labels, _, _ = next(iterator)
            images = images.to(self.device)
            if self.pred_class:
                recons = self.model.transform(images, hard_occ_grid=True).cpu()
                infer_seg = self.model.transform(
                    images, pred_instance_labels=True, with_bkg=self.eval_with_bkg
                ).cpu()
            else:
                # TODO: proba segmentation
                exit("Not implemented")
                _, distances, class_prob = self.model(images)
                distances = distances.view(B, *(self.n_prototypes,) * self.n_objects)
                other_idxs = []
                for k in range(self.n_objects, 1, -1):
                    distances, idx = distances.min(k)
                    other_idxs.insert(0, idx)
                dist_min_by_sample, argmin_idx = distances.min(1)

                recons = self.model.transform(images).cpu()
                recons = recons.view(B, *(self.n_prototypes,) * self.n_objects, C, H, W)
                infer_seg = self.model.transform(
                    images, pred_instance_labels=True, with_bkg=self.eval_with_bkg
                ).cpu()
                infer_seg = infer_seg.view(
                    B, *(self.n_prototypes,) * self.n_objects, H, W
                )
                real_idxs = []
                for idx in [argmin_idx] + other_idxs:
                    for i in real_idxs:
                        idx = idx[torch.arange(B), i]
                    real_idxs.insert(0, idx)
                    infer_seg = infer_seg[torch.arange(B), idx]
                    recons = recons[torch.arange(B), idx]

                if not self.eval_with_bkg:
                    bkg_idx = infer_seg == 0
                    tsf_layers = self.model.predict(images)[0]
                    new_target = (
                            ((tsf_layers - images) ** 2).sum(3).min(1)[0].argmin(0).long()
                            + 1
                    ).cpu()
                    infer_seg[bkg_idx] = new_target[bkg_idx]

            infer_seg = infer_seg.unsqueeze(1).expand(-1, C, H, W)
            color_seg = torch.zeros(infer_seg.shape).float()
            masks = []
            for k, col in enumerate(colors):
                masks.append(infer_seg == k)
                color_seg[masks[-1]] = (
                    torch.Tensor(col)[None, :, None, None]
                    .to("cpu")
                    .expand(B, C, H, W)[masks[-1]]
                )

            images = images.cpu()
            for k in range(B):
                name = f"{k + j * B}".zfill(2)
                convert_to_img(images[k]).save(out / f"{name}.png")
                convert_to_img(recons[k]).save(out / f"{name}_recons.png")
                convert_to_img(color_seg[k]).save(out / f"{name}_seg_full.png")
                for l in range(self.n_objects + 1):
                    convert_to_img((images[k] * masks[l][k])).save(
                        out / f"{name}_seg_obj{l}.png"
                    )

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
            B = images.size(0)
            images = images.to(self.device)
            _, distances, class_prob = self.model(images)
            if self.n_backgrounds > 1:
                assert class_prob is None
                distances, bkg_idx = distances.view(
                    B, self.n_prototypes, self.n_backgrounds
                ).min(2)
            if self.n_objects > 1:
                distances = distances.view(B, *(self.n_prototypes,) * self.n_objects)
                other_idxs = []
                for k in range(self.n_objects, 1, -1):
                    distances, idx = distances.min(k)
                    other_idxs.insert(0, idx)

            dist_min_by_sample, argmin_idx = distances.min(1)
            if self.learn_proba:
                class_oh = class_prob.permute(2, 0, 1).flatten(1)
                argmin_idx = class_oh.argmax(1)

            loss.update(dist_min_by_sample.mean(), n=len(dist_min_by_sample))
            assert self.n_objects == 1
            scores.update(labels.long().numpy(), argmin_idx.cpu().numpy())

        scores = scores.compute()
        self.print_and_log_info("final_loss: {:.4f}".format(float(loss.avg)))
        self.print_and_log_info(
            "final_scores: "
            + ", ".join(["{}={:.4f}".format(k, v) for k, v in scores.items()])
        )
        with open(scores_path, mode="a") as f:
            f.write(
                "{:.6}\t".format(float(loss.avg))
                + "\t".join(map("{:.6f}".format, scores.values()))
                + "\n"
            )

    ######################
    #  VISUALIZE METHODS #
    ######################

    def win_loss(self, split):
        return f"{split}_losses"

    def get_n_clusters(self):
        """Return number of clusters for visualization."""
        if hasattr(self, 'n_clusters'):
            return self.n_clusters if self.n_clusters <= 40 else 2 * self.n_prototypes
        return super().get_n_clusters()

    def should_visualize_cls_scores(self):
        """Determine if class scores should be visualized."""
        return not hasattr(self, 'instance_eval') or not self.instance_eval

    def cls_score_name(self):
        """Return score name based on evaluation type."""
        if hasattr(self, 'seg_eval') and self.seg_eval:
            return "iou"
        return "acc"


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    parser = argparse.ArgumentParser(
        description="Pipeline to train a NN model specified by a YML config"
    )

    torch.backends.cudnn.enabled = False
    print(OmegaConf.to_yaml(cfg))

    dataset = cfg["dataset"]["name"]
    seed = cfg["training"]["seed"]
    save = cfg["training"]["save"]
    now = datetime.datetime.now().isoformat()
    job_name = HydraConfig.get().job.name

    tag = f"{dataset}_{job_name}_{now}"

    if cfg["training"]["cont"] == True:
        cfg["training"]["resume"] = tag

    run_dir = RUNS_PATH / dataset / tag
    run_dir = str(run_dir)
    trainer = Trainer(cfg, run_dir, seed=seed, save=save)
    try:
        trainer.run(seed=seed)
    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
