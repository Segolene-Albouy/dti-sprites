import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F

from .utils.image import convert_to_img
from .utils.logger import print_info
from .utils.consts import *

PRINT_TRAIN_STAT_FMT = "Epoch [{}/{}], Iter [{}/{}], train_metrics: {}".format
PRINT_VAL_STAT_FMT = "Epoch [{}/{}], Iter [{}/{}], val_metrics: {}".format
PRINT_CHECK_CLUSTERS_FMT = (
    "Epoch [{}/{}], Iter [{}/{}]: Reassigned clusters {} from cluster {}".format
)
PRINT_LR_UPD_FMT = "Epoch [{}/{}], Iter [{}/{}], LR update: lr = {}".format



from abc import ABC, abstractmethod
import torch
import numpy as np
from pathlib import Path

class AbstractTrainer(ABC):
    """Abstract base class for all Trainer implementations."""

    @abstractmethod
    def __init__(self, *args, **kwargs):
        self.run_dir = None
        self.logger = None

        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        nb_device = torch.cuda.device_count() if self.device.type == "cuda" else 0
        self.print_and_log_info(
            f"Using {self.device} device, nb_device is {nb_device}"
        )

        # Dataset attributes
        self.dataset_kwargs = None
        self.dataset_name = None
        self.n_classes = None
        self.is_val_empty = None
        self.img_size = None

        # Data loaders
        self.batch_size = None
        self.n_workers = None
        self.train_loader = None
        self.val_loader = None
        self.n_batches = None

        # Training parameters
        self.n_iterations = None
        self.n_epochs = None
        self.start_epoch = None
        self.start_batch = None

        # Model configuration
        self.model_kwargs = None
        self.model_name = None
        self.is_gmm = None
        self.model = None
        self.n_prototypes = None

        # Optimizer and scheduler
        self.optimizer = None
        self.scheduler = None
        self.cur_lr = None
        self.scheduler_update_range = None

        # Metrics and logging
        self.train_stat_interval = None
        self.train_metrics = None
        self.train_metrics_path = None
        self.val_stat_interval = None
        self.val_metrics = None
        self.val_metrics_path = None
        self.val_scores = None
        self.val_scores_path = None

        # Cluster checking
        self.check_cluster_interval = None

        # Visualization
        self.visualizer = None
        self.save_img = None

        # Image transformation
        self.images_to_tsf = None
        self.prototypes_path = None
        self.transformation_path = None

        # Evaluation modes
        self.seg_eval = None
        self.instance_eval = None

        self.interpolate_settings = {'mode': 'bilinear'}

        raise NotImplementedError

    def run(self):
        # """Main training loop with defined sequence of operations."""
        # self._setup()
        #
        # for epoch in range(self.start_epoch, self.n_epochs + 1):
        #     self._run_epoch(epoch)
        #
        # self._finalize()
        # return self.results
        pass

    @abstractmethod
    def _setup(self):
        # """Setup training environment, metrics, visualization."""
        # self.logger = self._setup_logger()
        #
        # # Initialize common components
        # self.device = self._setup_device()
        # self.train_loader, self.val_loader = self._setup_data()
        # self.model = self._setup_model()
        # self.optimizer = self._setup_optimizer()
        # self.scheduler = self._setup_scheduler()
        #
        # # Metrics and visualization
        # self.metrics = self._setup_metrics()
        # self.visualizer = self._setup_visualizer()
        pass


    def _run_epoch(self, epoch):
        """Run a single epoch."""
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
        """Hook called before each epoch."""
        # self.model.train()
        # self._log_info(f"Starting epoch {epoch}/{self.n_epochs}")
        pass

    def _process_batch(self, batch_idx, batch_data, is_training):
        # raise NotImplementedError
        pass

    def _after_epoch(self, epoch):
        # """Hook called after each epoch."""
        # if self.scheduler and self.scheduler_update_range == "epoch":
        #     self.scheduler.step()
        #
        # # Default implementation can be overridden
        # self._save_checkpoint(epoch)
        pass

    def _finalize(self):
        # """Finalize training and save model."""
        # self._save_checkpoint(self.n_epochs)
        # self.logger.info("Training finished.")
        pass

    @abstractmethod
    def print_and_log_info(self, string):
        print_info(string)
        self.logger.info(string)

    @abstractmethod
    def load_from_tag(self, tag, resume=False):
        """Load model from a previous run tag."""
        raise NotImplementedError

    @property
    def score_name(self):
        return self.val_scores.score_name

    @abstractmethod
    def print_memory_usage(self, prefix):
        """Print GPU memory usage information with the given prefix."""
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

    @abstractmethod
    def update_scheduler(self, epoch, batch):
        """Update the learning rate scheduler."""
        self.scheduler.step()
        lr = self.scheduler.get_last_lr()[0]
        if lr != self.cur_lr:
            self.cur_lr = lr
            self.print_and_log_info(
                PRINT_LR_UPD_FMT(epoch, self.n_epochs, batch, self.n_batches, lr)
            )

    @abstractmethod
    def single_train_batch_run(self, *args, **kwargs):
        """Process a single training batch."""
        raise NotImplementedError

    @abstractmethod
    def run_val(self):
        """Run validation process."""
        raise NotImplementedError

    @abstractmethod
    def log_images(self, cur_iter):
        """Log images for visualization."""
        raise NotImplementedError

    @torch.no_grad()
    def save_pred(self, cur_iter=None, pred_name="prototype", prefix=None, transform_fn=None, n_preds=None):
        prefix = prefix or pred_name
        pred_names = f"{pred_name}s"

        try:
            preds = getattr(self.model, pred_names)
            n_preds = n_preds or getattr(self, f"n_{pred_names}")
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

    def save_prototypes(self, cur_iter=None):
        self.save_pred(cur_iter, pred_name="prototype", prefix="proto")

    # def save_masks(self, cur_iter=None):
    #     self.save_pred(cur_iter, pred_name="mask")
    #
    # def save_backgrounds(self, cur_iter=None):
    #     self.save_pred(cur_iter, pred_name="background", prefix="bkg")
    #
    # def save_masked_prototypes(self, cur_iter=None):
    #     self.save_pred(
    #         cur_iter,
    #         pred_name="prototype",
    #         transform_fn=lambda proto, k: proto * self.model.masks[k],
    #         prefix="proto"
    #     )
    #
    # def save_variance(self, cur_iter=None):
    #     self.save_pred(cur_iter, pred_name="variance", prefix="var", n_preds=self.n_prototypes)

    @abstractmethod
    def save_transformed_images(self, cur_iter=None):
        """Save transformed images."""
        raise NotImplementedError


    @torch.no_grad()
    def update_visualizer_images(self, images, title, nrow):
        if self.visualizer is None:
            return None

        if max(images.shape[1:]) > VIZ_MAX_IMG_SIZE:
            images = torch.nn.functional.interpolate(
                images,
                size=VIZ_MAX_IMG_SIZE,
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

    @abstractmethod
    def check_cluster(self, cur_iter, epoch, batch):
        """Check cluster assignments and reassign empty clusters."""
        raise NotImplementedError

    @abstractmethod
    def log_train_metrics(self, cur_iter, epoch, batch):
        stat = PRINT_TRAIN_STAT_FMT(
            epoch, self.n_epochs, batch, self.n_batches, self.train_metrics
        )
        self.print_and_log_info(stat[:1000])
        with open(self.train_metrics_path, mode="a") as f:
            f.write(
                "{}\t{}\t{}\t".format(cur_iter, epoch, batch)
                + "\t".join(map("{:.5f}".format, self.train_metrics.avg_values))
                + "\n"
            )

        self.update_visualizer_metrics(cur_iter, train=True)
        self.train_metrics.reset("time/img", "loss", "loss_em", "loss_bin", "loss_rec", "loss_freq")

    def win_loss(self, split):
        return f"{split}_loss"

    def update_visualizer_metrics(self, cur_iter, train):
        """Update visualizer with metrics."""
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

    def _visualize_losses(self, cur_iter, split, metrics):
        """Visualize loss metrics."""
        losses = [n for n in metrics.names if n.startswith("loss")]  # NOTE kmeans => if "loss" in n
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

    def get_n_clusters(self):
        """Return number of clusters to display in visualizations."""
        return self.n_prototypes

    def _visualize_cluster_proportions(self, metrics):
        """Visualize cluster proportions."""
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
        """Visualize global validation scores."""
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
        """Determine if class scores should be visualized."""
        return True

    def cls_score_name(self):
        return "acc"

    def _visualize_class_scores(self, cur_iter):
        """Visualize class-specific scores."""
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

    @abstractmethod
    def log_val_metrics(self, cur_iter, epoch, batch, precision=5):
        stat = PRINT_VAL_STAT_FMT(
            epoch, self.n_epochs, batch, self.n_batches, self.val_metrics
        )

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

    @abstractmethod
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
        if hasattr(self, "sprite_optimizer"): # NOTE only for kmeans trainer
            state["sprite_optimizer_state"] = self.sprite_optimizer.state_dict()
        save_path = self.run_dir / MODEL_FILE
        torch.save(state, save_path)
        self.print_and_log_info("Model saved at {}".format(save_path))

    @abstractmethod
    def evaluate(self):
        """Run evaluation process."""
        raise NotImplementedError

    @abstractmethod
    def qualitative_eval(self):
        """Run qualitative evaluation."""
        raise NotImplementedError

    @abstractmethod
    def quantitative_eval(self):
        """Run quantitative evaluation."""
        raise NotImplementedError
