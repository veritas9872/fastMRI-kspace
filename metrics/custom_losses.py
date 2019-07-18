import torch
from torch import nn
import torch.nn.functional as F

from metrics.my_ssim import ssim_loss


class CSSIM(nn.Module):  # Complementary SSIM
    def __init__(self, filter_size=7, k1=0.01, k2=0.03, sigma=1.5, reduction='mean'):
        super().__init__()
        # self.max_val = default_range
        self.filter_size = filter_size
        self.k1 = k1
        self.k2 = k2
        self.sigma = sigma
        self.reduction = reduction

    def forward(self, input, target, max_val=None):
        assert input.shape == target.shape, 'Input and target sizes do not match.'
        assert input.device == target.device, 'Input and target are on different devices.'

        true_range = target.max() - target.min()

        if max_val is None:
            max_val = true_range
        elif max_val < true_range:
            raise RuntimeWarning('Given value range is smaller than actual range of values.')

        return 1 - ssim_loss(input, target, max_val=max_val, filter_size=self.filter_size, k1=self.k1, k2=self.k2,
                             sigma=self.sigma, reduction=self.reduction)


class MySSIM(nn.Module):
    def __init__(self, filter_size=7, k1=0.01, k2=0.03, sigma=1.5, reduction='mean'):
        super().__init__()
        self.filter_size = filter_size
        self.k1 = k1
        self.k2 = k2
        self.sigma = sigma
        self.reduction = reduction

    def forward(self, input, target, max_val=None):
        assert input.shape == target.shape, 'Input and target sizes do not match.'
        assert input.device == target.device, 'Input and target are on different devices.'

        true_range = target.max() - target.min()

        if max_val is None:
            max_val = true_range
        elif max_val < true_range:
            raise RuntimeWarning('Given value range is smaller than actual range of values.')

        return ssim_loss(input, target, max_val=max_val, filter_size=self.filter_size,
                         k1=self.k1, k2=self.k2, sigma=self.sigma, reduction=self.reduction)


class L1CSSIM(nn.Module):  # Replace this with a system of summing losses in Model Trainer later on.
    def __init__(self, l1_weight, default_range=1, filter_size=11, k1=0.01, k2=0.03, sigma=1.5, reduction='mean'):
        super().__init__()
        self.l1_weight = l1_weight
        self.max_val = default_range
        self.filter_size = filter_size
        self.k1 = k1
        self.k2 = k2
        self.sigma = sigma
        self.reduction = reduction

    def forward(self, input, target, max_val=None):
        max_val = self.max_val if max_val is None else max_val

        cssim = 1 - ssim_loss(input, target, max_val=max_val, filter_size=self.filter_size, k1=self.k1, k2=self.k2,
                              sigma=self.sigma, reduction=self.reduction)

        l1_loss = F.l1_loss(input, target, reduction=self.reduction)

        return cssim + self.l1_weight * l1_loss


def psnr_loss(img_comp, img_orig, data_range):
    assert img_comp.size() == img_orig.size()

    # true_range = img_orig.max() - img_orig.min()
    #
    # if data_range is None:
    #     data_range = true_range
    #
    # if true_range > data_range:
    #     raise ValueError('True data range is greater than given value range.')

    err = F.mse_loss(img_comp, img_orig, reduction='mean')

    return 10 * torch.log10((data_range * data_range) / err)


def nmse_loss(img_comp, img_orig):
    return F.mse_loss(img_comp, img_orig, reduction='sum') / torch.sum(img_orig ** 2)


