import torch
from torch import nn
import torch.nn.functional as F


def _fspecial_gauss_1d(size: int, sigma: float) -> torch.Tensor:
    r"""Create 1-D gauss kernel
    Args:
        size (int): the size of gaussian kernel
        sigma (float): sigma of normal distribution
    Returns:
        torch.Tensor: 1D kernel
    """
    coords = torch.arange(size, dtype=torch.float32)
    coords -= (size - 1) / 2.

    grid = -(coords**2) / (2 * sigma**2)
    grid = F.softmax(grid, dim=-1)

    return grid.view(1, 1, -1)


def gaussian_filter(input: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    r""" Blur input with 1-D kernel
    Args:
        input (torch.Tensor): a batch of tensors to be blurred
        kernel (torch.Tensor): 1-D gaussian kernel
    Returns:
        torch.Tensor: blurred tensors
    """

    ch = input.size(1)  # The kernel is expected to have been expanded by the number of channels already.
    # No padding is used, thus removing the edges from consideration.
    out = F.conv2d(input, kernel, stride=1, padding=0, groups=ch)
    out = F.conv2d(out, kernel.transpose(-2, -1), stride=1, padding=0, groups=ch)
    return out


def _ssim(input, target, kernel, max_val, k1=0.01, k2=0.03, biased_cov=True):
    r""" Calculate ssim index for X and Y
    Args:
        input (torch.Tensor): images
        target (torch.Tensor): images
        kernel (torch.Tensor): 1-D gauss kernel. Expected to be on the same device as input and target.
        max_val (float or int, optional): value range of input images. (usually 1.0 or 255)
    Returns:
        torch.Tensor: ssim results
    """

    if biased_cov:
        compensation = 1.
    else:
        raise NotImplementedError('I have not understood compensation well enough yet.')

    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2

    mu1 = gaussian_filter(input, kernel)
    mu2 = gaussian_filter(target, kernel)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = compensation * (gaussian_filter(input * input, kernel) - mu1_sq)
    sigma2_sq = compensation * (gaussian_filter(target * target, kernel) - mu2_sq)
    sigma12 = compensation * (gaussian_filter(input * target, kernel) - mu1_mu2)

    cs_map = (2 * sigma12 + c2) / (sigma1_sq + sigma2_sq + c2)
    ssim_map = ((2 * mu1_mu2 + c1) / (mu1_sq + mu2_sq + c1)) * cs_map

    # reduce along CHW
    ssim_val = ssim_map.mean(dim=(-1, -2, -3))
    cs = cs_map.mean(dim=(-1, -2, -3))

    return ssim_val, cs


def ssim(input, target, max_val=None, filter_size=11, sigma=1.5, kernel=None, reduction='mean'):
    r""" interface of ssim
    Args:
        input (torch.Tensor): a batch of images, (N,C,H,W)
        target (torch.Tensor): a batch of images, (N,C,H,W)
        max_val (float or int, optional): value range of input images. (usually 1.0 or 255)
        filter_size: (int, optional): the size of gaussian kernel.
        sigma: (float, optional): sigma of the normal distribution.
        kernel (torch.Tensor, optional): 1-D gauss kernel.
            If None, a new kernel will be created according to filter_size and sigma.
        reduction (str): reduction method for outputs. One of 'mean' or 'sum'.
    Returns:
        torch.Tensor: ssim results
    """
    if not input.shape == target.shape:
        raise ValueError('Input images must have the same shapes and dimensions.')

    if not input.type() == target.type():
        raise ValueError('Input and target images must have the same data type and be on the same device.')

    dim = input.dim()
    if dim == 2:
        input = input.expand(1, 1, -1, -1)
        target = target.expand(1, 1, -1, -1)
    elif dim == 3:
        input = input.expand(1, -1, -1, -1)
        target = target.expand(1, -1, -1, -1)
    elif dim != 4:
        raise ValueError(f'Input images must be 2D, 3D, or 4D tensors, got {dim}D tensors.')

    if max_val is None:
        max_val = target.max() - target.min()  # May cause problems if value is 0.

    if kernel is None:
        kernel = _fspecial_gauss_1d(filter_size, sigma).to(device=input.device)

    ch = input.size(1)
    kernel = kernel.expand(ch, 1, 1, -1)  # Dynamic window expansion. expand() does not copy memory.

    # Not exposing k1, k2, etc. SSIM should have unbiased covariance but this isn't ready yet.
    ssim_val, _ = _ssim(input, target, kernel=kernel, max_val=max_val, biased_cov=True)

    if reduction == 'mean':
        ssim_val = ssim_val.mean()
    elif reduction == 'sum':
        ssim_val = ssim_val.sum()
    elif reduction == 'none':
        pass
    else:
        raise NotImplementedError(f'Invalid reduction method `{reduction}`.')

    return ssim_val


def ms_ssim(input, target, filter_size=11, sigma=1.5, kernel=None, max_val=None, weights=None, reduction='mean'):
    r""" interface of ms-ssim
    Args:
        input (torch.Tensor): a batch of images, (N,C,H,W)
        target (torch.Tensor): a batch of images, (N,C,H,W)
        filter_size: (int, optional): the size of gauss kernel
        sigma: (float, optional): sigma of normal distribution
        kernel (torch.Tensor, optional): 1-D gauss kernel.
            If None, a new kernel will be created according to filter_size and sigma
        max_val (float or int, optional): value range of input images. (usually 1.0 or 255)
        weights (list, optional): weights for different levels
        reduction (str): reduction method for outputs.
    Returns:
        torch.Tensor: ms-ssim results
    """

    if not input.shape == target.shape:
        raise ValueError('Input images must have the same shapes and dimensions.')

    dim = input.dim()
    if dim == 2:
        input = input.expand(1, 1, -1, -1)
        target = target.expand(1, 1, -1, -1)
    elif dim == 3:
        input = input.expand(1, -1, -1, -1)
        target = target.expand(1, -1, -1, -1)
    elif dim != 4:
        raise ValueError(f'Input images must be 2D, 3D, or 4D tensors, got {dim}D tensors.')

    if not input.type() == target.type():
        raise ValueError('Input and target images must have the same data type and be on the same device.')

    if max_val is None:
        max_val = target.max() - target.min()  # May cause problems if value is 0.

    if weights is None:
        weights = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], dtype=input.dtype, device=input.device)

    if kernel is None:
        kernel = _fspecial_gauss_1d(filter_size, sigma).to(device=input.device)

    ch = input.size(1)
    kernel = kernel.expand(ch, 1, 1, -1)  # Dynamic window expansion. expand() does not copy memory.

    levels = len(weights)
    mcs = list()
    ssim_val = float('nan')  # Just in case there are no weights. Also keeps pylint happy.
    for _ in range(levels):
        ssim_val, cs = _ssim(input, target, kernel=kernel, max_val=max_val, biased_cov=True)
        mcs.append(cs)

        padding = (input.size(-2) % 2, input.size(-1) % 2)
        input = F.avg_pool2d(input, kernel_size=2, padding=padding)
        target = F.avg_pool2d(target, kernel_size=2, padding=padding)

    mcs = torch.stack(mcs, dim=0)  # mcs, (level, batch)  # weights, (level)
    ms_ssim_val = torch.prod((mcs[:-1] ** weights[:-1].unsqueeze(1)) * (ssim_val ** weights[-1]), dim=0)  # (batch, )

    if reduction == 'mean':
        ms_ssim_val = ms_ssim_val.mean()
    elif reduction == 'sum':
        ms_ssim_val = ms_ssim_val.sum()
    elif reduction == 'none':
        pass
    else:
        raise NotImplementedError(f'Invalid reduction method `{reduction}`.')
    return ms_ssim_val


# Classes to re-use kernel
class SSIM(nn.Module):
    def __init__(self, filter_size=11, sigma=1.5, max_val=None, reduction='mean'):
        r""" class for ssim
        Args:
            filter_size: (int, optional): the size of gauss kernel
            sigma: (float, optional): sigma of normal distribution
            max_val (float or int, optional): value range of input images. (usually 1.0 or 255)
        """

        super().__init__()
        self.register_buffer('kernel', _fspecial_gauss_1d(filter_size, sigma))
        self.max_val = max_val
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return ssim(input, target, max_val=self.max_val, kernel=self.kernel, reduction=self.reduction)


class MSSSIM(nn.Module):
    def __init__(self, filter_size=11, sigma=1.5, max_val=None, weights=None, reduction='mean'):
        r""" class for ms-ssim
        Args:
            filter_size: (int, optional): the size of gauss kernel
            sigma: (float, optional): sigma of normal distribution
            max_val (float or int, optional): value range of input images. (usually 1.0 or 255)
            weights (list, optional): weights for different levels
        """

        super().__init__()
        self.register_buffer('kernel', _fspecial_gauss_1d(filter_size, sigma))
        if weights is None:
            weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
        self.register_buffer('weights', torch.tensor(weights))
        self.max_val = max_val
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return ms_ssim(input, target, kernel=self.kernel, max_val=self.max_val, weights=self.weights)


class SSIMLoss(nn.Module):
    def __init__(self, filter_size=11, sigma=1.5, max_val=None, reduction='mean'):
        r""" class for ssim
        Args:
            filter_size: (int, optional): the size of gauss kernel
            sigma: (float, optional): sigma of normal distribution
            max_val (float or int, optional): value range of input images. (usually 1.0 or 255)
        """

        super().__init__()
        self.register_buffer('kernel', _fspecial_gauss_1d(filter_size, sigma))
        self.register_buffer('one', torch.tensor(1., dtype=torch.float32))
        self.max_val = max_val
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.one - ssim(input, target, max_val=self.max_val, kernel=self.kernel, reduction=self.reduction)


class MSSSIMLoss(nn.Module):
    def __init__(self, filter_size=11, sigma=1.5, max_val=None, weights=None, reduction='mean'):
        r""" class for ms-ssim
        Args:
            filter_size: (int, optional): the size of gauss kernel
            sigma: (float, optional): sigma of normal distribution
            max_val (float or int, optional): value range of input images. (usually 1.0 or 255)
            weights (list, optional): weights for different levels
        """

        super().__init__()
        self.register_buffer('kernel', _fspecial_gauss_1d(filter_size, sigma))
        self.register_buffer('weights', torch.tensor(weights))
        self.register_buffer('one', torch.tensor(1., dtype=torch.float32))
        self.max_val = max_val
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.one - ms_ssim(input, target, kernel=self.kernel, max_val=self.max_val, weights=self.weights)


class LogSSIMLoss(nn.Module):
    r""" Implementation of ln(1 - SSIM) loss function. Includes epsilon value for identical images.
    Args:
        filter_size: (int, optional): the size of gauss kernel
        sigma: (float, optional): sigma of normal distribution
        max_val (float or int, optional): value range of input images. (usually 1.0 or 255)
    """

    def __init__(self, filter_size=11, sigma=1.5, max_val=None, epsilon=0., reduction='mean'):
        super().__init__()
        self.register_buffer('kernel', _fspecial_gauss_1d(filter_size, sigma))
        self.max_val = max_val
        self.reduction = reduction
        self.register_buffer('epsilon', torch.tensor(epsilon, dtype=torch.float32))

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Implement ln(1 - SSIM) loss function.
        return torch.log1p(
            self.epsilon - ssim(input, target, max_val=self.max_val, kernel=self.kernel, reduction=self.reduction))
