import torch
from torch.nn.functional import conv2d, _Reduction, avg_pool2d
from torch.nn.modules.loss import _Loss


def _fspecial_gaussian(size, channel, sigma):
    coords = torch.tensor([(x - (size - 1.) / 2.) for x in range(size)])
    coords = -coords ** 2 / (2. * sigma ** 2)
    grid = coords.view(1, -1) + coords.view(-1, 1)
    grid = grid.view(1, -1)
    grid = grid.softmax(-1)
    kernel = grid.view(1, 1, size, size)
    kernel = kernel.expand(channel, 1, size, size).contiguous()
    return kernel


def _ssim(input, target, max_val, k1, k2, channel, kernel):
    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2

    mu1 = conv2d(input, kernel, groups=channel)
    mu2 = conv2d(target, kernel, groups=channel)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = conv2d(input * input, kernel, groups=channel) - mu1_sq
    sigma2_sq = conv2d(target * target, kernel, groups=channel) - mu2_sq
    sigma12 = conv2d(input * target, kernel, groups=channel) - mu1_mu2

    v1 = 2 * sigma12 + c2
    v2 = sigma1_sq + sigma2_sq + c2

    ssim = ((2 * mu1_mu2 + c1) * v1) / ((mu1_sq + mu2_sq + c1) * v2)
    return ssim, v1 / v2


def ssim_loss(input, target, max_val, filter_size=11, k1=0.01, k2=0.03,
              sigma=1.5, kernel=None, size_average=None, reduce=None, reduction='mean'):
    # type: (Tensor, Tensor, float, int, float, float, float, Tensor, Optional[bool], Optional[bool], str) -> Tensor
    r"""ssim_loss(input, target, max_val, filter_size, k1, k2,
                  sigma, kernel=None, size_average=None, reduce=None, reduction='mean') -> Tensor
    Measures the structural similarity index (SSIM) error.
    See :class:`~torch.nn.SSIMLoss` for details.
    """

    if input.size() != target.size():
        raise ValueError('Expected input size ({}) to match target size ({}).'
                         .format(input.size(0), target.size(0)))

    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_string(size_average, reduce)

    dim = input.dim()
    if dim == 2:
        input = input.expand(1, 1, input.dim(-2), input.dim(-1))
        target = target.expand(1, 1, target.dim(-2), target.dim(-1))
    elif dim == 3:
        input = input.expand(1, input.dim(-3), input.dim(-2), input.dim(-1))
        target = target.expand(1, target.dim(-3), target.dim(-2), target.dim(-1))
    elif dim != 4:
        raise ValueError('Expected 2, 3, or 4 dimensions (got {})'.format(dim))

    _, channel, _, _ = input.size()

    if kernel is None:
        kernel = _fspecial_gaussian(filter_size, channel, sigma)
    kernel = kernel.to(device=input.device)

    ret, _ = _ssim(input, target, max_val, k1, k2, channel, kernel)

    if reduction != 'none':
        ret = torch.mean(ret) if reduction == 'mean' else torch.sum(ret)
    return ret


def ms_ssim_loss(input, target, max_val, filter_size=11, k1=0.01, k2=0.03,
                 sigma=1.5, kernel=None, weights=None, size_average=None, reduce=None, reduction='mean'):
    # type: (Tensor, Tensor, float, int, float, float, float, Tensor, list, Optional[bool], Optional[bool], str) -> Tensor
    r"""ms_ssim_loss(input, target, max_val, filter_size, k1, k2,
                     sigma, kernel=None, size_average=None, reduce=None, reduction='mean') -> Tensor
    Measures the multi-scale structural similarity index (MS-SSIM) error.
    See :class:`~torch.nn.MSSSIMLoss` for details.
    """

    if input.size() != target.size():
        raise ValueError('Expected input size ({}) to match target size ({}).'
                         .format(input.size(0), target.size(0)))

    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_string(size_average, reduce)

    dim = input.dim()
    if dim == 2:
        input = input.expand(1, 1, input.dim(-2), input.dim(-1))
        target = target.expand(1, 1, target.dim(-2), target.dim(-1))
    elif dim == 3:
        input = input.expand(1, input.dim(-3), input.dim(-2), input.dim(-1))
        target = target.expand(1, target.dim(-3), target.dim(-2), target.dim(-1))
    elif dim != 4:
        raise ValueError('Expected 2, 3, or 4 dimensions (got {})'.format(dim))

    _, channel, _, _ = input.size()

    if kernel is None:
        kernel = _fspecial_gaussian(filter_size, channel, sigma)
    kernel = kernel.to(device=input.device)

    if weights is None:
        weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    weights = torch.tensor(weights, device=input.device)
    weights = weights.unsqueeze(-1).unsqueeze(-1)
    levels = weights.size(0)
    mssim = []
    mcs = []
    for _ in range(levels):
        ssim, cs = _ssim(input, target, max_val, k1, k2, channel, kernel)
        ssim = ssim.mean((2, 3))
        cs = cs.mean((2, 3))
        mssim.append(ssim)
        mcs.append(cs)

        input = avg_pool2d(input, (2, 2))
        target = avg_pool2d(target, (2, 2))

    mssim = torch.stack(mssim)
    mcs = torch.stack(mcs)
    p1 = mcs ** weights
    p2 = mssim ** weights

    ret = torch.prod(p1[:-1], 0) * p2[-1]

    if reduction != 'none':
        ret = torch.mean(ret) if reduction == 'mean' else torch.sum(ret)
    return ret


class SSIMLoss(_Loss):
    r"""Creates a criterion that measures the structural similarity index error between
    each element in the input :math:`x` and target :math:`y`.
    The unreduced (i.e. with :attr:`reduction` set to ``'none'``) loss can be described as:
    .. math::
        SSIM = \{ssim_1,\dots,ssim_{N \times C}\}, \quad
        ssim_{l}(x, y) = \frac{(2 \mu_x \mu_y + c_1) (2 \sigma_{xy} + c_2)}
        {(\mu_x^2 +\mu_y^2 + c_1)(\sigma_x^2 +\sigma_y^2 + c_2)},
    where :math:`N` is the batch size, `C` is the channel size. If :attr:`reduction` is not ``'none'``
    (default ``'mean'``), then:
    .. math::
        SSIMLoss(x, y) =
        \begin{cases}
            \operatorname{mean}(SSIM), &  \text{if reduction} = \text{'mean';}\\
            \operatorname{sum}(SSIM),  &  \text{if reduction} = \text{'sum'.}
        \end{cases}
    :math:`x` and :math:`y` are tensors of arbitrary shapes with a total
    of :math:`n` elements each.
    The sum operation still operates over all the elements, and divides by :math:`n`.
    The division by :math:`n` can be avoided if one sets ``reduction = 'sum'``.
    Args:
        channel (int, optional): The channel size of elements. Default: 3
        max_val (float, optional): The difference between the maximum and minimum of the pixel value,
            i.e., if for image x it holds min(x) = 0 and max(x) = 1, then max_val = 1.
            The pixel value interval of both input and output should remain the same. Default: 1.
        filter_size (int, optional): By default, the mean and covariance of a pixel is obtained
            by convolution with given filter_size. Default: 11
        k1 (float, optional): Coefficient related to c1 in the above equation. Default: 0.01
        k2 (float, optional): Coefficient related to c2 in the above equation. Default: 0.03
        sigma (float, optional): Standard deviation for Gaussian kernel. Default: 1.5
        kernel (Tensor, optional): The kernel used in sliding gaussian window. Default: ``None``
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there are multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when reduce is ``False``. Default: ``True``
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``
    Shape:
        - Input: :math:`(N, *)` where :math:`*` means, any number of additional dimensions
        - Target: :math:`(N, *)`, same shape as the input
    Examples::
        >>> loss = nn.SSIMLoss()
        >>> input = torch.rand(3, 3, 256, 256, requires_grad=True)
        >>> target = torch.rand(3, 3, 256, 256)
        >>> output = loss(input, target, max_val=1.)
        >>> output.backward()
    """
    __constants__ = ['filter_size', 'k1', 'k2', 'sigma', 'kernel', 'reduction']

    def __init__(self, channel=3, filter_size=11, k1=0.01, k2=0.03, sigma=1.5, size_average=None, reduce=None, reduction='mean'):
        super(SSIMLoss, self).__init__(size_average, reduce, reduction)
        self.filter_size = filter_size
        self.k1 = k1
        self.k2 = k2
        self.sigma = sigma
        self.kernel = _fspecial_gaussian(filter_size, channel, sigma)

    def forward(self, input, target, max_val=1.):
        return ssim_loss(input, target, max_val=max_val, filter_size=self.filter_size, k1=self.k1, k2=self.k2,
                         sigma=self.sigma, reduction=self.reduction, kernel=self.kernel)


class MultiScaleSSIMLoss(_Loss):
    r"""Creates a criterion that measures the multi-scale structural similarity index error between
    each element in the input :math:`x` and target :math:`y`.
    The unreduced (i.e. with :attr:`reduction` set to ``'none'``) loss can be described as:
    .. math::
        MSSIM = \{mssim_1,\dots,mssim_{N \times C}\}, \quad
        mssim_{l}(x, y) = \frac{(2 \mu_{x,m} \mu_{y,m} + c_1) }
        {(\mu_{x,m}^2 +\mu_{y,m}^2 + c_1)} \prod_{j=1}^{m - 1}
        \frac{(2 \sigma_{xy,j} + c_2)}{(\sigma_{x,j}^2 +\sigma_{y,j}^2 + c_2)}
    where :math:`N` is the batch size, `C` is the channel size, `m` is the scale level (Default: 5).
    If :attr:`reduction` is not ``'none'``(default ``'mean'``), then:
    .. math::
        MultiscaleSSIMLoss(x, y) =
        \begin{cases}
            \operatorname{mean}(MSSIM), &  \text{if reduction} = \text{'mean';}\\
            \operatorname{sum}(MSSIM),  &  \text{if reduction} = \text{'sum'.}
        \end{cases}
    :math:`x` and :math:`y` are tensors of arbitrary shapes with a total
    of :math:`n` elements each.
    The sum operation still operates over all the elements, and divides by :math:`n`.
    The division by :math:`n` can be avoided if one sets ``reduction = 'sum'``.
    Args:
        channel (int, optional): The channel size of elements. Default: 3
        max_val (float, optional): The difference between the maximum and minimum of the pixel value,
            i.e., if for image x it holds min(x) = 0 and max(x) = 1, then max_val = 1.
            The pixel value interval of both input and output should remain the same. Default: 1.
        filter_size (int, optional): By default, the mean and covariance of a pixel is obtained
            by convolution with given filter_size. Default: 11
        k1 (float, optional): Coefficient related to c1 in the above equation. Default: 0.01
        k2 (float, optional): Coefficient related to c2 in the above equation. Default: 0.03
        sigma (float, optional): Standard deviation for Gaussian kernel. Default: 1.5
        kernel (Tensor, optional): The kernel used in sliding gaussian window. Default: ``None``
        weights (list, optional): The list that weight the relative importance between different scales.
            Deafault: ``[0.0448, 0.2856, 0.3001, 0.2363, 0.1333]``
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there are multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when reduce is ``False``. Default: ``True``
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``
    Shape:
        - Input: :math:`(N, *)` where :math:`*` means, any number of additional dimensions
        - Target: :math:`(N, *)`, same shape as the input
    Examples::
        >>> loss = nn.MultiScaleSSIMLoss()
        >>> input = torch.rand(3, 3, 256, 256, requires_grad=True)
        >>> target = torch.rand(3, 3, 256, 256)
        >>> output = loss(input, target, max_val=1.)
        >>> output.backward()
    """
    __constants__ = ['filter_size', 'k1', 'k2', 'sigma', 'kernel', 'reduction']

    def __init__(self, channel=3, filter_size=11, k1=0.01, k2=0.03, sigma=1.5, size_average=None, reduce=None, reduction='mean'):
        super(MultiScaleSSIMLoss, self).__init__(size_average, reduce, reduction)
        self.filter_size = filter_size
        self.k1 = k1
        self.k2 = k2
        self.sigma = sigma
        self.kernel = _fspecial_gaussian(filter_size, channel, sigma)

    def forward(self, input, target, weights=[0.0448, 0.2856, 0.3001, 0.2363, 0.1333], max_val=1.):
        return ms_ssim_loss(input, target, max_val=max_val, k1=self.k1, k2=self.k2, sigma=self.sigma, kernel=self.kernel,
                            weights=weights, filter_size=self.filter_size, reduction=self.reduction)


