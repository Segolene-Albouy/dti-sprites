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

    def criterion(self, inp, target, alpha_masks=None, reduction="mean"):
        dist = self._criterion(inp, target)

        if alpha_masks is not None:
            masks = alpha_masks.clamp(0, 1)
            dist = dist * masks

            if reduction == "mean":
                visible_pixels = masks.sum(dim=(-2, -1), keepdim=True).clamp(min=1)
                return (dist.sum(dim=(-2, -1), keepdim=True) / visible_pixels).squeeze(-1).squeeze(-1)
            elif reduction == "sum":
                return dist.sum(dim=(-2, -1))
            elif reduction == "none":
                return dist

        if reduction == "mean":
            return dist.flatten(-2).mean(-1)
        elif reduction == "sum":
            return dist.flatten(-2).sum(-1)
        elif reduction == "none":
            return dist

        raise NotImplementedError(f"Reduction {reduction} not supported")
