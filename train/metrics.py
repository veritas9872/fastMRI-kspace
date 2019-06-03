import torch.nn as nn
import torch.nn.functional as F


class CustomL1Loss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, image_recons, targets):
        """

        Args:
            image_recons (list):
            targets (list):

        Returns:

        """
        assert len(image_recons) == len(targets)

        return sum(F.l1_loss(recon, target, reduction=self.reduction) for recon, target in zip(image_recons, targets))

