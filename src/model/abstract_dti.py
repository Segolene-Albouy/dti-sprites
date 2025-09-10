import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class AbstractDTI(nn.Module, ABC):
    """
    Abstract base class for DTI models
    """

    def __init__(self):
        super().__init__()
        self._criterion = nn.MSELoss(reduction="none")

    @abstractmethod
    def forward(self, x, masks=None):
        """
        Abstract forward method that subclasses must implement

        Args:
            x: Input images
            masks: Optional alpha masks for transparency

        Returns:
            Tuple of (loss, distances, probabilities/class_info)
        """
        pass

    @staticmethod
    def match_dist_shape(dist, w):
        if w.shape != dist.shape:
            # dist.shape = (B, n_proto, C, H, W) // mask.shape = (B, 1, H, W)
            while w.ndim < dist.ndim:
                w = w.unsqueeze(-3)
            w = w.expand_as(dist)
        return w


    def criterion(self, inp, target, alpha_masks=None, weights=None, reduction="mean"):
        # dist.shape = (B, n_proto, C, H, W)
        dist = self._criterion(inp, target)
        normalizer = torch.ones_like(dist)

        if alpha_masks is not None:
            # mask.shape = (B, 1, H, W)
            alpha_masks = self.match_dist_shape(dist, alpha_masks.clamp(0, 1))
            dist *= alpha_masks
            normalizer *= alpha_masks

        if weights is not None:
            weights = self.match_dist_shape(dist, weights)
            dist *= weights
            normalizer *= weights

        if reduction == "mean":
            valid_weights = normalizer.sum(dim=(-2, -1), keepdim=True).clamp(min=1)
            return (dist.sum(dim=(-2, -1), keepdim=True) / valid_weights).squeeze(-1).squeeze(-1)
        elif reduction == "sum":
            return dist.sum(dim=(-2, -1))
        elif reduction == "none":
            return dist
        raise NotImplementedError(f"Reduction {reduction} not supported")

    def reassign_empty_clusters(self, proportions):
        if not self._reassign_cluster  or getattr(self, "are_sprite_frozen", False):
            return [], 0

        if self.n_prototypes == 1:
            return [], 0

        if getattr(self, "add_empty_sprite", False):
            proportions = proportions[:-1] / max(proportions[:-1])

        if not isinstance(proportions, torch.Tensor):
            proportions = torch.tensor(proportions, device=next(self.parameters()).device)

        n_proto = len(proportions) # self.n_prototypes

        idx = torch.argmax(proportions).item()
        reassigned = []
        for i in range(n_proto):
            if proportions[i] < self.empty_cluster_threshold:
                self.restart_branch_from(i, idx)
                reassigned.append(i)

        if len(reassigned) > 0:
            self.restart_branch_from(idx, idx)
        return reassigned, idx
