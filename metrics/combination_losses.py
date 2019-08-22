import torch
from torch import nn
import torch.nn.functional as F

from metrics.custom_losses import CSSIM
from metrics.new_1d_ssim import _fspecial_gauss_1d, ssim


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


class L1SSIMLoss(nn.Module):
    def __init__(self, filter_size=11, sigma=1.5, max_val=None, l1_ratio=0.5, reduction='mean'):
        r""" class for ssim
        Args:
            filter_size: (int, optional): the size of gauss kernel
            sigma: (float, optional): sigma of normal distribution
            max_val (float or int, optional): value range of input images. (usually 1.0 or 255)
        """

        super().__init__()
        self.register_buffer('kernel', _fspecial_gauss_1d(filter_size, sigma))
        self.register_buffer('one', torch.tensor(1., dtype=torch.float32))
        self.register_buffer('l1_ratio', torch.tensor(l1_ratio, dtype=torch.float32))
        self.max_val = max_val
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> (torch.Tensor, dict):
        ssim_loss = self.one - ssim(input, target, max_val=self.max_val, kernel=self.kernel, reduction=self.reduction)
        l1_loss = F.l1_loss(input, target, reduction=self.reduction)
        return l1_loss * self.l1_ratio + (1 - self.l1_ratio) * ssim_loss, {'l1_loss': l1_loss, 'ssim_loss': ssim_loss}
