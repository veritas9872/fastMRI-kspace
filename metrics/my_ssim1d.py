import torch
from torch import nn
import torch.nn.functional as F


def _fspecial_gauss_1d(size, sigma):
    r"""Create 1-D gaussian kernel
    Args:
        size (int): the size of gauss kernel
        sigma (float): sigma of normal distribution
    Returns:
        torch.Tensor: 1D kernel
    """
    coords = torch.arange(size, dtype=torch.float)
    coords -= size//2

    g = torch.exp(-(coords**2) / (2*sigma**2))
    g /= g.sum()

    return g.expand(1, 1, -1)


def gaussian_filter(input, window):
    r""" Blur input with 1-D kernel
    Args:
        input (torch.Tensor): a batch of tensors to be blurred
        window (torch.Tensor): 1-D gauss kernel
    Returns:
        torch.Tensor: blurred tensors
    """
    channels = input.size(1)
    # Input window has the shape (1, 1, win_size)
    window = window.expand(channels, 1, 1, -1)  # Dynamic channel expansion.
    out = F.conv2d(input, window, stride=1, padding=0, groups=channels)
    out = F.conv2d(out, window.transpose(2, 3), stride=1, padding=0, groups=channels)
    return out


def _ssim(input, target, window, data_range=None, k1=0.01, k2=0.03, size_average=True, full=False):
    r""" Calculate ssim index for X and Y
    Args:
        input (torch.Tensor): images
        target (torch.Tensor): images
        window (torch.Tensor): 1-D gauss kernel
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        full (bool, optional): return sc or not
    Returns:
        torch.Tensor: ssim results
    """
    compensation = 1.0

    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    # window = window.to(input.device, dtype=input.dtype)

    mu1 = gaussian_filter(input, window)
    mu2 = gaussian_filter(target, window)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = compensation * (gaussian_filter(input * input, window) - mu1_sq)
    sigma2_sq = compensation * (gaussian_filter(target * target, window) - mu2_sq)
    sigma12 = compensation * (gaussian_filter(input * target, window) - mu1_mu2)

    cs_map = (2 * sigma12 + c2) / (sigma1_sq + sigma2_sq + c2)
    ssim_map = ((2 * mu1_mu2 + c1) / (mu1_sq + mu2_sq + c1)) * cs_map

    if size_average:
        ssim_val = ssim_map.mean()
        cs = cs_map.mean()
    else:
        # ssim_val = ssim_map.mean(-1).mean(-1).mean(-1)  # reduce along CHW
        ssim_val = torch.mean(ssim_map, dim=(-1, -2, -3))
        # cs = cs_map.mean(-1).mean(-1).mean(-1)
        cs = torch.mean(cs_map, dim=(-1, -2, -3))

    if full:
        return ssim_val, cs
    else:
        return ssim_val


def ssim(input, target, win_size=11, win_sigma=1.5, win=None, data_range=None, size_average=True, full=False):
    r""" interface of ssim
    Args:
        input (torch.Tensor): a batch of images, (N,C,H,W)
        target (torch.Tensor): a batch of images, (N,C,H,W)
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        full (bool, optional): return sc or not
    Returns:
        torch.Tensor: ssim results
    """

    if not input.type() == target.type():
        raise ValueError('Input images must have the same dtype.')

    if not input.shape == target.shape:
        print(input.shape, target.shape)
        raise ValueError('Input images must have the same dimensions.')

    dim = input.dim()
    if dim == 2:
        input = input.expand(1, 1, -1, -1)
        target = target.expand(1, 1, -1, -1)
    elif dim == 3:
        input = input.expand(1, -1, -1, -1)
        target = target.expand(1, -1, -1, -1)
    elif dim != 4:
        raise ValueError('Input images are expected to be 2, 3, or 4D tensors.')

    if not (win_size % 2 == 1):
        raise ValueError('Window size must be odd.')

    if data_range is None:
        data_range = target.max() - target.min()

    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma).to(dtype=input.dtype, device=input.device)
        win = win.expand(-1, 1, 1, 1)
    else:
        win_size = win.size(-1)

    ssim_val, cs = _ssim(input, target, window=win, data_range=data_range, size_average=False, full=True)
    if size_average:
        ssim_val = ssim_val.mean()
        cs = cs.mean()

    if full:
        return ssim_val, cs
    else:
        return ssim_val


def ms_ssim(input, target, win_size=11, win_sigma=1.5, win=None, data_range=None, size_average=True, weights=None):
    r""" interface of ms-ssim
    Args:
        input (torch.Tensor): a batch of images, (N,C,H,W)
        target (torch.Tensor): a batch of images, (N,C,H,W)
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        full (bool, optional): return sc or not
        weights (list, optional): weights for different levels
    Returns:
        torch.Tensor: ms-ssim results
    """

    if not input.type() == target.type():
        raise ValueError('Input images must have the same dtype.')

    if not input.shape == target.shape:
        raise ValueError('Input images must have the same dimensions.')

    dim = input.dim()
    if dim == 2:
        input = input.expand(1, 1, -1, -1)
        target = target.expand(1, 1, -1, -1)
    elif dim == 3:
        input = input.expand(1, -1, -1, -1)
        target = target.expand(1, -1, -1, -1)
    elif dim != 4:
        raise ValueError('Input images must 2, 3, or 4D tensors.')

    if not (win_size % 2 == 1):
        raise ValueError('Window size must be odd.')

    if weights is None:
        values = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
        weights = torch.tensor(values, dtype=input.dtype, device=input.device)

    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma).to(dtype=input.dtype, device=input.device)
        win = win.expand(-1, 1, 1, 1)
    else:
        win_size = win.size(-1)

    if data_range is None:
        data_range = target.max() - target.min()

    levels = weights.size(0)
    mcs = list()
    for _ in range(levels):
        ssim_val, cs = _ssim(input, target, window=win, data_range=data_range, size_average=False, full=True)
        mcs.append(cs)

        padding = (input.shape[2] % 2, input.shape[3] % 2)
        input = F.avg_pool2d(input, kernel_size=2, padding=padding)
        target = F.avg_pool2d(target, kernel_size=2, padding=padding)

    mcs = torch.stack(mcs, dim=0)  # mcs, (level, batch)
    # weights, (level)
    msssim_val = torch.prod((mcs[:-1] ** weights[:-1].unsqueeze(1)) * (ssim_val ** weights[-1]), dim=0)  # (batch, )

    if size_average:
        msssim_val = msssim_val.mean()
    return msssim_val


# Classes to re-use window
class SSIM(nn.Module):
    def __init__(self, win_size=11, win_sigma=1.5, data_range=None, size_average=True):
        r""" class for ssim
        Args:
            win_size: (int, optional): the size of gauss kernel
            win_sigma: (float, optional): sigma of normal distribution
            data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
            size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        """

        super(SSIM, self).__init__()
        self.win = _fspecial_gauss_1d(win_size, win_sigma)

        self.register_buffer('window', _fspecial_gauss_1d(win_size, win_sigma))
        self.size_average = size_average
        self.data_range = data_range

    def forward(self, input, target):
        assert isinstance(input, torch.Tensor)
        # This will be unnecessary if the named buffer is used. The loss will have to be sent to GPU though.
        self.win = self.win.to(dtype=input.dtype, device=input.device)  # This is a non-op if already the same.
        return ssim(input, target, win=self.win, data_range=self.data_range, size_average=self.size_average)


class MSSSIM(nn.Module):
    def __init__(self, win_size=11, win_sigma=1.5, data_range=None, size_average=True, channel=3, weights=None):
        r""" class for ms-ssim
        Args:
            win_size: (int, optional): the size of gauss kernel
            win_sigma: (float, optional): sigma of normal distribution
            data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
            size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
            channel (int, optional): input channels (default: 3)
            weights (list, optional): weights for different levels
        """

        super(MSSSIM, self).__init__()
        self.win = _fspecial_gauss_1d(win_size, win_sigma)
        self.size_average = size_average
        self.data_range = data_range
        self.weights = torch.tensor(weights)

        self.register_buffer('window', _fspecial_gauss_1d(win_size, win_sigma))
        self.register_buffer('weights', torch.tensor(weights))

    def forward(self, input, target):
        assert isinstance(input, torch.Tensor)
        self.win = self.win.to(dtype=input.dtype, device=input.device)  # This is a non-op if already the same.
        self.weights = self.weights.to(dtype=input.dtype, device=input.device)
        return ms_ssim(input, target, win=self.win, size_average=self.size_average, data_range=self.data_range, weights=self.weights)
