import torch
from torch import nn

from metrics.custom_losses import CSSIM


class L1CSSIM7(nn.Module):
    """
    Implementation of L1 and complementary SSIM (1 - SSIM) loss with weighting according to the API specified in
    Model Trainer K2CI. Alpha dictates the proportion that L1 is given in the total loss.
    The kernel size of SSIM is fixed at 7, the size used for the fast-MRI challenge.
    """
    def __init__(self, reduction='mean', alpha=0.5):
        super().__init__()
        assert 0 <= alpha <= 1, f'Invalid value of alpha: {alpha}'
        self.l1_loss = nn.L1Loss(reduction=reduction)
        self.cssim = CSSIM(filter_size=7, reduction=reduction)
        self.alpha = alpha  # l1 * alpha + (1 - ssim) * (1 - alpha)

    def forward(self, tensor, target):
        l1_loss = self.l1_loss(tensor, target)
        cssim = self.cssim(tensor, target)
        img_loss = l1_loss * self.alpha + cssim * (1 - self.alpha)
        return img_loss, {'l1_loss': l1_loss, 'cssim': cssim}
