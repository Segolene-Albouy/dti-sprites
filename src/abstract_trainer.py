import datetime
import os
import shutil
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, DictConfig
from torch.utils.data import DataLoader

from .dataset import get_dataset
from .model import get_model
from .model.tools import count_parameters
from .scheduler import get_scheduler
from .utils import coerce_to_path_and_create_dir
from .utils.metrics import Metrics
from .utils.path import RUNS_PATH
from .utils.plot import plot_lines, plot_bar

try:
    import visdom
except ModuleNotFoundError:
    pass

from .utils.image import convert_to_img, save_gif, normalize_values
from .utils.logger import print_info, get_logger
from .utils.consts import *


from abc import ABC, abstractmethod

class AbstractTrainer(ABC):
    """Abstract base class for all Trainer implementations"""

    run_dir = None
    logger = None
    save_img = False
    device = None
    cfg = None

    # Dataset attributes
    dataset_kwargs = None
    dataset_name = None
    n_classes = None
    is_val_empty = None
    img_size = None
    train_dataset = None
    val_dataset = None

    # Data loaders
    batch_size = None
    n_workers = None
    train_loader = None
    val_loader = None
    n_batches = None

    # Training parameters
    n_iterations = None
    n_epochs = None
    start_epoch = None
    start_batch = None

    # Model configuration
    model_name = None
    model_kwargs = None
    is_gmm = None
    model = None
    n_prototypes = None

    # Optimizer and scheduler
    optimizer = None
    scheduler = None
    cur_lr = None
    scheduler_update_range = None

    # Metrics and logging
    train_stat_interval = None
    train_metrics = None
    train_metrics_path = None
    val_stat_interval = None
    val_metrics = None
    val_metrics_path = None
    val_scores = None
    val_scores_path = None
    bin_edges = None
    bin_counts = None

    # Cluster checking
    check_cluster_interval = None

    # Visualization
    visualizer = None

    # Image transformation
    images_to_tsf = None
    prototypes_path = None
    transformation_path = None

    # Evaluation modes
    seg_eval = None
    instance_eval = None

    interpolate_settings = {
        'mode': 'bilinear',
        'align_corners': False
    }

    def __init__(self, cfg, run_dir, save=False, *args, **kwargs):
        self.run_dir = coerce_to_path_and_create_dir(run_dir)
        self.setup_logging()
        self.setup_config(cfg)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.print_device_info()

        self.save_img = save
        self.setup_dataset()
        self.setup_dataloaders()

        self.setup_model()
        self.setup_directories()

        self.setup_optimizer()
        self.setup_scheduler()

        self.setup_checkpoint_loading()
        self.setup_metrics()

        self.setup_prototypes()
        self.setup_visualizer()

    @property
    def has_sprite_optimizer(self):
        return False

    ######################
    #   SETUP METHODS    #
    ######################

    def setup_logging(self):
        self.logger = get_logger(self.run_dir, name="trainer")
        self.print_and_log_info(
            f"Trainer initialisation: run directory is {self.run_dir}"
        )

    def setup_config(self, cfg):
        """Load and process configuration"""
        OmegaConf.save(cfg, self.run_dir / "config.yaml")
        self.cfg = cfg
        self.print_and_log_info(f"Current config saved to {self.run_dir}")

    def setup_dataset(self):
        """Set up dataset parameters and load dataset"""
        self.dataset_kwargs = self.cfg["dataset"].copy()
        self.dataset_name = self.dataset_kwargs["name"]

        self.train_dataset = get_dataset(self.dataset_name)("train", **self.dataset_kwargs)
        self.val_dataset = get_dataset(self.dataset_name)("val", **self.dataset_kwargs)

        self.n_classes = self.train_dataset.n_classes
        self.is_val_empty = len(self.val_dataset) == 0
        self.img_size = self.train_dataset.img_size

        self.print_and_log_info(
            f"Dataset {self.dataset_name} instantiated with img_size={self.train_dataset}, n_channels={self.train_dataset.n_channels}, "
        )
        self.print_and_log_info(
            f"Found {len(self.train_dataset)} train samples / {len(self.val_dataset)} val samples"
        )

    def setup_dataloaders(self):
        """Create data loaders from datasets"""
        if not hasattr(self, 'train_dataset') or not hasattr(self, 'val_dataset'):
            raise AttributeError("train_dataset and val_dataset must be set before calling setup_dataloaders")

        self.batch_size = (
            self.cfg["training"]["batch_size"]
            if self.cfg["training"]["batch_size"] < len(self.train_dataset)
            else len(self.train_dataset)
        )
        self.n_workers = self.cfg["training"].get("n_workers", 4)

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.n_workers,
            shuffle=True,
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.n_workers
        )

        self.print_and_log_info(
            f"Dataloaders instantiated with batch_size={self.batch_size} and n_workers={self.n_workers}"
        )
        self.n_batches = len(self.train_loader)
        self._setup_iteration_counts()

    def _setup_iteration_counts(self):
        """Setup iteration and epoch counts"""
        self.n_iterations = self.cfg["training"].get("n_iterations")
        self.n_epochs = self.cfg["training"].get("n_epochs")

        assert not (self.n_iterations is not None and self.n_epochs is not None)

        if self.n_iterations is not None:
            self.n_epochs = max(self.n_iterations // self.n_batches, 1)
        else:
            self.n_iterations = self.n_epochs * self.n_batches


    @abstractmethod
    def get_model(self):
        raise NotImplementedError

    def setup_model(self):
        """Initialize model architecture"""
        self.model_kwargs = self.cfg["model"].copy()

        self.model = self.get_model().to(self.device)
        self.n_prototypes = self.model.n_prototypes
        self.is_gmm = "gmm" in self.model_name

        self.print_and_log_info(
            f"Using model {self.model_name} with kwargs {self.model_kwargs}"
        )
        self.print_and_log_info(
            f"Number of trainable parameters: {count_parameters(self.model):,}"
        )

    def setup_directories(self):
        if not self.save_img:
            return

        self.prototypes_path = coerce_to_path_and_create_dir(self.run_dir / "prototypes")
        for k in range(self.n_prototypes):
            coerce_to_path_and_create_dir(self.prototypes_path / f"proto{k}")

        self.transformation_path = coerce_to_path_and_create_dir(self.run_dir / "transformations")
        self.images_to_tsf = next(iter(self.train_loader))[0][:N_TRANSFORMATION_PREDICTIONS].to(self.device)

        for k in range(self.images_to_tsf.size(0)):
            out = coerce_to_path_and_create_dir(self.transformation_path / f"img{k}")
            convert_to_img(self.images_to_tsf[k]).save(out / "input.png")

    @abstractmethod
    def setup_optimizer(self):
        """Configure optimizer for training"""
        raise NotImplementedError

    def setup_scheduler(self):
        scheduler_params = self.cfg["training"].get("scheduler", {}) or {}
        scheduler_name = self.get_scheduler_name(scheduler_params)
        self.scheduler_update_range = self.cfg["training"].get("scheduler_update_range", "epoch")

        assert self.scheduler_update_range in ["epoch", "batch"]

        milestones = scheduler_params.get("milestones", [])
        if scheduler_name == "multi_step" and (len(milestones) > 0 and isinstance(milestones[0], float)):
            n_tot = (
                self.n_epochs
                if self.scheduler_update_range == "epoch"
                else self.n_iterations
            )
            scheduler_params["milestones"] = [
                round(m * n_tot) for m in scheduler_params["milestones"]
            ]

        self.scheduler = get_scheduler(scheduler_name)(
            self.optimizer, **scheduler_params
        )
        self.cur_lr = self.scheduler.get_last_lr()[0]

        self.print_and_log_info(
            f"Using scheduler {scheduler_name} with parameters {scheduler_params}"
        )

    def get_scheduler_name(self, scheduler_params):
        """Extract scheduler name from config.
        Subclasses can override this to customize name extraction"""
        name = self.cfg["training"].get("scheduler_name")
        if name is None and "name" in scheduler_params:
            name = scheduler_params.pop("name", None)
        return name

    def setup_checkpoint_loading(self):
        """Handle loading from pretrained models or resuming training"""
        checkpoint_path = self.cfg["training"].get("pretrained")
        checkpoint_path_resume = self.cfg["training"].get("resume")
        assert not (checkpoint_path is not None and checkpoint_path_resume is not None)
        if checkpoint_path is not None:
            self.load_from_tag(checkpoint_path)
        elif checkpoint_path_resume is not None:
            self.load_from_tag(checkpoint_path_resume, resume=True)
        else:
            self.start_epoch, self.start_batch = 1, 1


    def setup_metrics(self):
        """Initialize metrics for training and evaluation"""
        self.setup_train_metrics()
        self.setup_val_metrics()
        self.check_cluster_interval = self.cfg["training"]["check_cluster_interval"]

    @property
    def train_metric_names(self):
        # overriden in child trainers
        return ["time/img", "loss"]

    def setup_train_metrics(self):
        self.bin_edges = np.arange(0, 1.1, 0.1)
        self.bin_counts = np.zeros(len(self.bin_edges) - 1)

        self.train_stat_interval = self.cfg["training"]["train_stat_interval"]
        self.train_metrics = Metrics(*self.train_metric_names)

        self.train_metrics_path = self.run_dir / TRAIN_METRICS_FILE
        self.save_metrics_file(self.train_metrics_path, self.train_metrics.names)

    @abstractmethod
    def setup_val_scores(self):
        raise NotImplementedError

    def setup_val_metrics(self):
        self.val_stat_interval = self.cfg["training"]["val_stat_interval"]

        self.val_metrics = Metrics("loss_val")

        self.val_metrics_path = self.run_dir / VAL_METRICS_FILE
        self.save_metrics_file(self.val_metrics_path, self.val_metrics.names)

        self.setup_val_scores()

        self.val_scores_path = self.run_dir / VAL_SCORES_FILE
        self.save_metrics_file(self.val_scores_path, self.val_scores.names)

    def setup_prototypes(self):
        # self.images_to_tsf = next(iter(self.train_loader))[0][:N_TRANSFORMATION_PREDICTIONS].to(self.device)
        pass

    @abstractmethod
    def setup_visualizer(self):
        viz_port = self.cfg["training"].get("visualizer_port")

        if viz_port is not None:
            try:
                import visdom
                os.environ["http_proxy"] = ""
                self.visualizer = visdom.Visdom(port=viz_port, env=f"{self.run_dir.parent.name}_{self.run_dir.name}")
                self.visualizer.delete_env(self.visualizer.env) # Clean env before plotting

                self.print_and_log_info(f"Visualizer initialized at port {viz_port}")
            except (ImportError, Exception) as e:
                self.visualizer = None
                self.print_and_log_info(f"Visdom initialization failed: {e}")
        else:
            self.print_and_log_info("No visualizer initialized (no port specified)")

    ######################
    #    MAIN METHODS    #
    ######################

    @abstractmethod
    def run(self):
        # """Main training loop with defined sequence of operations"""
        # self._setup()
        #
        # for epoch in range(self.start_epoch, self.n_epochs + 1):
        #     self._run_epoch(epoch)
        #
        # self._finalize()
        # return self.results
        pass

    def _run_epoch(self, epoch):
        """Run a single epoch"""
        # self._before_epoch(epoch)
        #
        # for batch_idx, batch_data in enumerate(self.train_loader):
        #     self._process_batch(batch_idx, batch_data, is_training=True)
        #
        #     if self._should_validate(batch_idx):
        #         self._validate()
        #
        # self._after_epoch(epoch)
        pass

    def _before_epoch(self, epoch):
        """Hook called before each epoch"""
        # self.model.train()
        # self._log_info(f"Starting epoch {epoch}/{self.n_epochs}")
        pass

    def _process_batch(self, batch_idx, batch_data, is_training):
        # raise NotImplementedError
        pass

    def _after_epoch(self, epoch):
        # """Hook called after each epoch"""
        # if self.scheduler and self.scheduler_update_range == "epoch":
        #     self.scheduler.step()
        #
        # # Default implementation can be overridden
        # self._save_checkpoint(epoch)
        pass

    def _finalize(self):
        # """Finalize training and save model"""
        # self._save_checkpoint(self.n_epochs)
        # self.logger.info("Training finished.")
        pass

    @abstractmethod
    def load_from_tag(self, tag, resume=False):
        """Load model from a previous run tag"""
        raise NotImplementedError

    def update_scheduler(self, epoch, batch):
        """Update the learning rate scheduler"""
        self.scheduler.step()
        lr = self.scheduler.get_last_lr()[0]
        if lr != self.cur_lr:
            self.cur_lr = lr
            self.print_and_log_info(
                f"{self.progress_str(epoch, batch)} | LR update: lr = {lr}"
            )

    @abstractmethod
    def single_train_batch_run(self, *args, **kwargs):
        """Process a single training batch"""
        raise NotImplementedError

    @abstractmethod
    def run_val(self):
        """Run validation process"""
        raise NotImplementedError

    ######################
    #   SAVING METHODS   #
    ######################

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

        if self.has_sprite_optimizer: # NOTE condition only present in kmeans_trainer
            state["sprite_optimizer_state"] = self.sprite_optimizer.state_dict()

        save_path = self.run_dir / MODEL_FILE
        torch.save(state, save_path)
        self.print_and_log_info(f"Model saved at {save_path}")

    @torch.no_grad()
    def save_pred(self, cur_iter=None, pred_name="prototype", prefix=None, transform_fn=None, n_preds=None):
        prefix = prefix or pred_name
        pred_names = f"{pred_name}s"

        try:
            preds = getattr(self.model, pred_names)
            n_preds = n_preds or getattr(self, f"n_{pred_names}", None) or self.n_prototypes
            pred_path = getattr(self, f"{pred_names}_path")

            for k in range(n_preds):
                data = preds[k]
                if transform_fn:
                    data = transform_fn(data, k)

                img = convert_to_img(data)

                if cur_iter is not None:
                    img.save(pred_path / f"{prefix}{k}" / f"{cur_iter}.jpg")
                else:
                    img.save(pred_path / f"{prefix}{k}.png")

        except AttributeError as e:
            self.print_and_log_info(f"Warning: Could not save {pred_names}: {e}")

    def save_prototypes(self, cur_iter=None, normalize_contrast=True):
        tsf = (lambda proto, k: normalize_values(proto)) if normalize_contrast else None
        self.save_pred(cur_iter, pred_name="prototype", prefix="proto", transform_fn=tsf)

    @abstractmethod
    def save_transformed_images(self, cur_iter=None):
        """Save transformed images"""
        raise NotImplementedError

    def save_training_metrics(self):
        """Save training metrics, plots, and visualizations"""
        self.model.eval()

        # Load metrics data
        df_train = pd.read_csv(self.train_metrics_path, sep="\t", index_col=0)
        df_val = pd.read_csv(self.val_metrics_path, sep="\t", index_col=0)
        df_scores = pd.read_csv(self.val_scores_path, sep="\t", index_col=0)

        if len(df_train) == 0:
            self.print_and_log_info("No metrics or plots to save")
            return

        self._save_loss_plots(df_train, df_val)
        self._save_cluster_plots(df_train)
        self._save_additional_metric_plots(df_train)

        if not self.is_val_empty:
            self._save_validation_plots(df_scores)

        if self.save_img:
            self._save_image_gifs()

        self.print_and_log_info("Training metrics and visualizations saved")

    def _save_loss_plots(self, df_train, df_val):
        """Save plots for loss metrics"""
        losses = self.get_metrics_names(prefix="loss")

        # Handle validation loss if available
        if not self.is_val_empty and "loss_val" in df_val.columns:
            df = df_train.join(df_val[["loss_val"]], how="outer")
            fig = plot_lines(df, losses + ["loss_val"], title="Loss")
        else:
            df = df_train
            fig = plot_lines(df, losses, title="Loss")

        fig.savefig(self.run_dir / "loss.pdf")

    def _save_cluster_plots(self, df_train):
        """Save plots for cluster proportions"""
        # Get cluster proportion metrics
        n_clusters = self.get_n_clusters()
        names = self.get_metrics_names(prefix="prop_")
        if n_clusters < len(names):
            names = names[:n_clusters]

        # Plot cluster proportions over time
        fig = plot_lines(df_train, names, title="Cluster proportions")
        fig.savefig(self.run_dir / "cluster_proportions.pdf")

        # Plot final cluster proportions as bar chart
        s = df_train[names].iloc[-1]
        s.index = list(map(lambda n: n.replace("prop_clus", ""), names))
        fig = plot_bar(s, title="Final cluster proportions")
        fig.savefig(self.run_dir / "cluster_proportions_final.pdf")

    def _save_additional_metric_plots(self, df_train):
        """Save additional metric plots.

        This is a hook for subclasses to override to save additional metrics.
        Default implementation saves cluster probabilities if available.
        """
        # Save cluster probabilities if available
        proba_names = self.get_metrics_names(prefix="proba_")
        if proba_names:
            fig = plot_lines(df_train, proba_names, title="Cluster Probabilities")
            fig.savefig(self.run_dir / "cluster_probabilities.pdf")

    def _save_validation_plots(self, df_scores):
        """Save plots for validation scores"""
        # Save global scores
        names = list(filter(lambda name: "cls" not in name, self.val_scores.names))
        fig = plot_lines(df_scores, names, title="Global scores", unit_yaxis=True)
        fig.savefig(self.run_dir / "global_scores.pdf")

        # Save class-specific scores
        if self.should_visualize_cls_scores():
            name = self.cls_score_name()
            N = self.n_classes
            fig = plot_lines(
                df_scores,
                [f"{name}_cls{i}" for i in range(N)],
                title="Scores by cls",
                unit_yaxis=True,
            )
            fig.savefig(self.run_dir / "scores_by_cls.pdf")

    def _save_image_gifs(self):
        """Save image artifacts as GIFs"""
        size = MAX_GIF_SIZE if MAX_GIF_SIZE < max(self.img_size) else self.img_size

        with torch.no_grad():
            self.save_prototypes()

        for k in range(self.n_prototypes):
            self.save_gif_to_path(self.prototypes_path / f"proto{k}", f"prototype{k}.gif", size=size)

        save_tsf = True
        if hasattr(self.model, 'transformer'): # DTIKmeans has a single transformer
            save_tsf = not hasattr(self.model.transformer, 'is_identity') and self.model.transformer.is_identity
        elif hasattr(self.model, 'transformer_is_identity'):
            save_tsf = not self.model.transformer_is_identity
        elif hasattr(self.model, 'is_layer_tsf_id'):
            save_tsf = not self.model.is_layer_tsf_id

        if save_tsf:
            self.save_transformed_images()
            for i in range(self.images_to_tsf.size(0)):
                for k in range(min(self.n_prototypes, self.get_n_clusters())):
                    tsf_path = self.transformation_path / f"img{i}" / f"tsf{k}"
                    self.save_gif_to_path(tsf_path, f"tsf{k}.gif", size=size)
        else:
            shutil.rmtree(str(self.transformation_path))
            coerce_to_path_and_create_dir(self.transformation_path)

        self._save_additional_image_gifs(size)

    @abstractmethod
    def _save_additional_image_gifs(self, size):
        """Save additional model-specific image GIFs, overriden by model-specific methods"""
        raise NotImplementedError

    @staticmethod
    def save_metrics_file(path, column_names):
        """Create or check a metrics output file"""
        if not path.exists():
            with open(path, mode="w") as f:
                f.write("\t".join(["iteration", "epoch", "batch"] + column_names) + "\n")

    def save_img_to_path(self, img, path: Path, filename):
        """Save an image to the specified path"""
        if not self.save_img:
            return
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        try:
            convert_to_img(img).save(path / filename)
        except Exception as e:
            self.print_and_log_info(f"Could not save image {filename}: {e}")

    def save_gif_to_path(self, path: Path, filename, size):
        """Save a GIF to the specified path"""
        if not self.save_img:
            return
        if not os.path.exists(path):
            # self.print_and_log_info(f"Could not save GIF: {path} does not exist")
            os.makedirs(path, exist_ok=True)
        try:
            save_gif(path, filename, size=size)
            shutil.rmtree(str(path))
        except Exception as e:
            self.print_and_log_info(f"Could not save GIF {filename}: {e}")

    ######################
    #   LOGGING METHODS  #
    ######################

    def progress_str(self, epoch, batch):
        return  f"Epoch [{epoch}/{self.n_epochs}], Iter [{batch}/{self.n_batches}]"

    def print_and_log_info(self, string):
        print_info(string)
        self.logger.info(string)

    def _log_model_specific_images(self, cur_iter):
        pass

    def _log_transformation_compositions(self, compositions, C, H, W):
        pass

    def get_transformation_nrow(self, tsf_imgs):
        return self.n_prototypes + 1

    def print_device_info(self):
        """Print information about the device configuration"""
        nb_device = torch.cuda.device_count() if self.device.type == "cuda" else None
        self.print_and_log_info(
            f"Using {self.device.type} device, nb_device is {nb_device}"
        )

    @torch.no_grad()
    def log_images(self, cur_iter):
        """Log images for visualization"""
        self.model.eval()
        self.save_prototypes(cur_iter)
        self.update_visualizer_images(self.model.prototypes, "prototypes", nrow=5)

        self._log_model_specific_images(cur_iter)

        tsf_result = self.save_transformed_images(cur_iter)

        if isinstance(tsf_result, tuple):
            tsf_imgs, compositions = tsf_result
        else:
            tsf_imgs, compositions = tsf_result, []

        c, h, w = tsf_imgs.shape[2:]
        nrow = self.get_transformation_nrow(tsf_imgs)
        self.update_visualizer_images(
            tsf_imgs.reshape(-1, c, h, w), "transformations", nrow=nrow
        )

        self._log_transformation_compositions(compositions, c, h, w)

    def print_memory_usage(self, prefix):
        """Print GPU memory usage information with the given prefix"""
        stats = [
            "memory_allocated",
            "max_memory_allocated",
            "memory_cached",
            "max_memory_cached"
        ]
        memory_stats = {stat: getattr(torch.cuda, stat)() / 1e6 for stat in stats}
        stats_text = " / ".join(
            f"{name}: {value:.0f}MiB" for name, value in memory_stats.items()
        )
        self.print_and_log_info(f"{prefix}:\t{stats_text}")

    def log_val_metrics(self, cur_iter, epoch, batch, precision=5):
        stat = f"{self.progress_str(epoch, batch)}: val_metrics: {self.val_metrics}"
        fmt = f"{{:.{precision}f}}".format
        self.print_and_log_info(stat)
        with open(self.val_metrics_path, mode="a") as f:
            f.write(
                f"{cur_iter}\t{epoch}\t{batch}\t"
                + "\t".join(map(fmt, self.val_metrics.avg_values))
                + "\n"
            )

        scores = self.val_scores.compute()
        self.print_and_log_info(
            "val_scores: "
            + ", ".join([f"{k}={fmt(v)}" for k, v in scores.items()])
        )
        with open(self.val_scores_path, mode="a") as f:
            f.write(
                f"{cur_iter}\t{epoch}\t{batch}\t"
                + "\t".join(map(fmt, scores.values()))
                + "\n"
            )

        self.update_visualizer_metrics(cur_iter, train=False)
        self.val_scores.reset()
        self.val_metrics.reset()

    def log_train_metrics(self, cur_iter, epoch, batch, precision=5):
        stat = f"{self.progress_str(epoch, batch)}: train_metrics: {self.train_metrics}"
        fmt = f"{{:.{precision}f}}".format

        self.print_and_log_info(stat[:1000])
        with open(self.train_metrics_path, mode="a") as f:
            f.write(
                f"{cur_iter}\t{epoch}\t{batch}\t"
                + "\t".join(map(fmt, self.train_metrics.avg_values))
                + "\n"
            )

        self.update_visualizer_metrics(cur_iter, train=True)
        self.train_metrics.reset("time/img", "loss", "loss_em", "loss_bin", "loss_rec", "loss_freq")

    ######################
    # VALIDATION METHODS #
    ######################

    @abstractmethod
    def check_cluster(self, cur_iter, epoch, batch):
        """Check cluster assignments and reassign empty clusters"""
        raise NotImplementedError

    ######################
    # EVALUATION METHODS #
    ######################

    @abstractmethod
    def evaluate(self):
        """Run evaluation process"""
        raise NotImplementedError

    @abstractmethod
    def qualitative_eval(self):
        """Run qualitative evaluation"""
        raise NotImplementedError

    @abstractmethod
    def quantitative_eval(self):
        """Run quantitative evaluation"""
        raise NotImplementedError

    ######################
    #  VISUALIZE METHODS #
    ######################

    def win_loss(self, split):
        return f"{split}_loss"

    @torch.no_grad()
    def update_visualizer_images(self, images, title, nrow):
        if self.visualizer is None:
            return None

        if max(images.shape[2:]) > VIZ_MAX_IMG_SIZE:
            # H, W = images.shape[2], images.shape[3]
            # if H > W:
            #     new_size = (VIZ_MAX_IMG_SIZE, int(W * VIZ_MAX_IMG_SIZE / H))
            # else:
            #     new_size = (int(H * VIZ_MAX_IMG_SIZE / W), VIZ_MAX_IMG_SIZE)

            H, W = images.shape[2], images.shape[3]
            scale = VIZ_MAX_IMG_SIZE / max(H, W)
            new_size = (int(H * scale), int(W * scale))
            images = torch.nn.functional.interpolate(
                images,
                size=new_size,
                **self.interpolate_settings
            )

        self.visualizer.images(
            images.clamp(0, 1),
            win=title,
            nrow=nrow,
            opts=dict(
                title=title,
                store_history=True,
                width=VIZ_WIDTH,
                height=VIZ_HEIGHT
            ),
        )
        return None

    def get_metrics_names(self, metrics=None, prefix="loss"):
        if not metrics:
            metrics = self.train_metrics
        # NOTE kmeans => if "loss" in n and not n.startswith("loss")
        # return [n for n in metrics.names if n.startswith(prefix)]
        return list(filter(lambda s: s.startswith(prefix), metrics.names))

    def _visualize_losses(self, cur_iter, split, metrics):
        """Visualize loss metrics"""
        losses = self.get_metrics_names(metrics)
        y, x = [[metrics[n].avg for n in losses]], [[cur_iter] * len(losses)]

        self.visualizer.line(
            y, x,
            win=self.win_loss(split),
            update="append",
            opts=dict(
                title=self.win_loss(split),
                legend=losses,
                width=VIZ_WIDTH,
                height=VIZ_HEIGHT,
            ),
        )

    def update_visualizer_metrics(self, cur_iter, train):
        """Update visualizer with metrics"""
        if self.visualizer is None:
            return None

        split, metrics = ("train", self.train_metrics) if train else ("val", self.val_metrics)
        self._visualize_losses(cur_iter, split, metrics)

        if train:
            if self.n_prototypes > 1:
                self._visualize_cluster_proportions(metrics)
        else:
            self._visualize_global_scores(cur_iter)
            self._visualize_class_scores(cur_iter)

    def get_n_clusters(self):
        """Return number of clusters to display in visualizations"""
        return self.n_prototypes

    def _visualize_cluster_proportions(self, metrics):
        """Visualize cluster proportions"""
        n_clusters = self.get_n_clusters()
        proportions = [metrics[f"prop_clus{i}"].avg for i in range(n_clusters)]

        self.visualizer.bar(
            proportions,
            win="train_cluster_prop",
            opts=dict(
                title="train_cluster_proportions",
                width=VIZ_HEIGHT,
                height=VIZ_HEIGHT,
            ),
        )

    def _visualize_global_scores(self, cur_iter):
        """Visualize global validation scores"""
        names = list(filter(lambda name: "cls" not in name, self.val_scores.names))
        y, x = [[self.val_scores[n] for n in names]], [[cur_iter] * len(names)]

        self.visualizer.line(
            y, x,
            win="global_scores",
            update="append",
            opts=dict(
                title="global_scores",
                legend=names,
                width=VIZ_WIDTH,
                height=VIZ_HEIGHT,
            ),
        )

    def should_visualize_cls_scores(self):
        """Determine if class scores should be visualized"""
        return True

    def cls_score_name(self):
        return "acc"

    def _visualize_class_scores(self, cur_iter):
        """Visualize class-specific scores"""
        if not self.should_visualize_cls_scores():
            return

        name = self.cls_score_name()
        N = self.n_classes
        y = [[self.val_scores[f"{name}_cls{i}"] for i in range(N)]]
        x = [[cur_iter] * N]

        self.visualizer.line(
            y, x,
            win=f"{name}_by_cls",
            update="append",
            opts=dict(
                title=f"{name}_by_cls",
                legend=[f"cls{i}" for i in range(N)],
                width=VIZ_WIDTH,
                height=VIZ_HEIGHT,
            ),
        )


def run_trainer(cfg: DictConfig, trainer_class, run_dir=None, enable_cuda_deterministic=False, *args, **kwargs):
    """
    Common function to run a trainer with appropriate configuration.

    Args:
        cfg (DictConfig): Hydra configuration
        trainer_class (class): The trainer class to instantiate
        run_dir (Path): Directory to save the results
        enable_cuda_deterministic (bool): Whether to enable CUDA deterministic behavior
    """
    if not enable_cuda_deterministic:
        torch.backends.cudnn.enabled = False

    print(OmegaConf.to_yaml(cfg))

    training_config = cfg.get("training", {})
    dataset = cfg.get("dataset", {}).get("name", "default")
    seed = training_config.get("seed", 777)
    save = training_config.get("save", True)

    try:
        job_name = HydraConfig.get().job.name
    except Exception:
        job_name = "default"
    now = datetime.datetime.now().isoformat()
    tag = f"{dataset}_{job_name}_{now}"

    if training_config.get("cont", False):
        training_config["resume"] = tag

    if not run_dir:
        run_dir = RUNS_PATH / dataset / tag
    trainer = trainer_class(cfg=cfg, run_dir=str(run_dir), seed=seed, save=save, *args, **kwargs)

    try:
        trainer.run(seed=seed)
        return trainer
    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise
