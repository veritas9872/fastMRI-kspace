import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_msssim


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

        a = sum(F.l1_loss(recon, target, reduction=self.reduction) for recon, target in zip(image_recons, targets))
        b = sum([ssim_loss(recon.unsqueeze(dim=0)*1E5, target.unsqueeze(dim=0)*1E5, torch.max(target)*1E5, reduction=self.reduction) for recon, target in zip(image_recons, targets)])
        # b = 0
        # return sum(F.l1_loss(recon, target, reduction=self.reduction) for recon, target in zip(image_recons, targets))\
        #     + sum(ssim_loss(recon.unsqueeze(dim=0), target.unsqueeze(dim=0), torch.max(target), reduction=self.reduction) for recon, target in zip(image_recons, targets))
        return 1E5 * a + b


class CustomL2Loss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, recons, targets):
        """

        Args:
            recons (list):
            targets (list):

        Returns:

        """
        assert len(recons) == len(targets)

        return sum(F.mse_loss(recon, target, reduction=self.reduction) for recon, target in zip(recons, targets))


def _fspecial_gaussian(_size, channel, sigma):
    coords = torch.tensor([(x - (_size - 1.) / 2.) for x in range(_size)])
    coords = -coords ** 2 / (2. * sigma ** 2)
    grid = coords.view(1, -1) + coords.view(-1, 1)
    grid = grid.view(1, -1)
    grid = grid.softmax(-1)
    kernel = grid.view(1, 1, _size, _size)
    kernel = kernel.expand(channel, 1, _size, _size).contiguous()
    return kernel


def _ssim(_input, target, max_val, k1, k2, channel, kernel):
    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2

    mu1 = F.conv2d(_input, kernel, groups=channel)
    mu2 = F.conv2d(target, kernel, groups=channel)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(_input * _input, kernel, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(target * target, kernel, groups=channel) - mu2_sq
    sigma12 = F.conv2d(_input * target, kernel, groups=channel) - mu1_mu2

    v1 = 2 * sigma12 + c2
    v2 = sigma1_sq + sigma2_sq + c2

    ssim = ((2 * mu1_mu2 + c1) * v1) / ((mu1_sq + mu2_sq + c1) * v2)
    return ssim, v1 / v2


# @weak_script
def ssim_loss(_input, target, max_val, filter_size=11, k1=0.01, k2=0.03,
              sigma=1.5, size_average=None, reduce=None, reduction='mean'):
    # type: (Tensor, Tensor, float, int, float, float, float, Optional[bool], Optional[bool], str) -> Tensor
    r"""ssim_loss(input, target, max_val, filter_size, k1, k2,
                  sigma, size_average=None, reduce=None, reduction='mean') -> Tensor
    Measures the structural similarity index (SSIM) error.
    See :class:`~torch.nn.SSIMLoss` for details.
    """

    dim = _input.dim()
    if dim != 4:
        raise ValueError('Expected 4 dimensions (got {})'.format(dim))

    if _input.size() != target.size():
        raise ValueError('Expected input size ({}) to match target size ({}).'
                         .format(_input.size(0), target.size(0)))

    if size_average is not None or reduce is not None:
        reduction = F._Reduction.legacy_get_string(size_average, reduce)

    _, channel, _, _ = _input.size()
    kernel = _fspecial_gaussian(filter_size, channel, sigma).to('cuda')
    ret, _ = _ssim(_input, target, max_val, k1, k2, channel, kernel)
    _input.squeeze(dim=0)
    target.squeeze(dim=0)

    if reduction != 'none':
        ret = torch.mean(ret) if reduction == 'mean' else torch.sum(ret)
    return ret


def ms_ssim_loss(_input, target, max_val, filter_size=11, k1=0.01, k2=0.03,
                 sigma=1.5, size_average=None, reduce=None, reduction='mean'):
    # type: (Tensor, Tensor, float, int, float, float, float, Optional[bool], Optional[bool], str) -> Tensor
    r"""ms_ssim_loss(input, target, max_val, filter_size, k1, k2,
                     sigma, size_average=None, reduce=None, reduction='mean') -> Tensor
    Measures the multi-scale structural similarity index (MS-SSIM) error.
    See :class:`~torch.nn.MSSSIMLoss` for details.
    """

    dim = _input.dim()
    if dim != 4:
        raise ValueError('Expected 4 dimensions (got {}) from input'.format(dim))

    if _input.size() != target.size():
        raise ValueError('Expected input size ({}) to match target size ({}).'
                         .format(_input.size(0), target.size(0)))

    if size_average is not None or reduce is not None:
        reduction = F._Reduction.legacy_get_string(size_average, reduce)

    _, channel, _, _ = _input.size()
    kernel = _fspecial_gaussian(filter_size, channel, sigma)

    weights = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    weights = weights.unsqueeze(-1).unsqueeze(-1)
    levels = weights.size(0)
    mssim = []
    mcs = []
    for _ in range(levels):
        ssim, cs = _ssim(_input, target, max_val, k1, k2, channel, kernel)
        ssim = ssim.mean((2, 3))
        cs = cs.mean((2, 3))
        mssim.append(ssim)
        mcs.append(cs)

        _input = F.avg_pool2d(_input, (2, 2))
        target = F.avg_pool2d(target, (2, 2))

    mssim = torch.stack(mssim)
    mcs = torch.stack(mcs)
    p1 = mcs ** weights
    p2 = mssim ** weights

    ret = torch.prod(p1[:-1], 0) * p2[-1]

    if reduction != 'none':
        ret = torch.mean(ret) if reduction == 'mean' else torch.sum(ret)
    return ret
