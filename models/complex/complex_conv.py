import torch
from torch import nn, Tensor
from torch.nn.init import _calculate_fan_in_and_fan_out

import numpy as np


class ComplexInitializer:
    """
    Implementation according to github page for Deep Complex Networks.
    I am not certain whether the mode is correct, the paper seems to suggest that it should be
    1 / sqrt(fan_in + fan_out) and 1 / sqrt(fan_in).
    However, I will follow my instincts and diverge from the code.
    """
    def __init__(self, method='kaiming'):
        assert method in ('kaiming', 'xavier'), 'Invalid initialization method.'
        self.method = method

    def get_weight_inits(self, weight_shape):
        fan_in, fan_out = _calculate_fan_in_and_fan_out(torch.zeros(size=weight_shape))
        if self.method == 'xavier':
            mode = 1 / np.sqrt(fan_in + fan_out)
        elif self.method == 'kaiming':
            mode = 1 / np.sqrt(fan_in)
        else:
            raise NotImplementedError('Invalid initialization method.')

        magnitude = np.random.rayleigh(scale=mode, size=weight_shape)
        phase = np.random.uniform(low=-np.pi, high=np.pi, size=weight_shape)
        weight_real = torch.from_numpy((magnitude * np.cos(phase)).astype(np.float32))
        weight_imag = torch.from_numpy((magnitude * np.sin(phase)).astype(np.float32))
        return weight_real, weight_imag


class ComplexConv2d(nn.Module):
    """
    Complex convolution in 2D.
    Expects the real and imaginary data to be in the second (dim=1) dimension.
    Thus, the input and output tensors are all 5D.

    Please note that this layer does not implement gradient clipping at norm of 1, as was the original implementation
    set out in DEEP COMPLEX NETWORKS (Trabelsi et al.).
    This is just an imitation with the bare minimum necessary to get complex convolutions functioning.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super().__init__()
        kwargs = dict(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)

        self.conv_real = nn.Conv2d(**kwargs)
        self.conv_imag = nn.Conv2d(**kwargs)

        # Weight initialization. Somewhat inelegant in style but it works.
        init = ComplexInitializer(method='kaiming')
        weight_real, weight_imag = init.get_weight_inits(weight_shape=self.conv_real.weight.shape)
        new_weights = {'conv_real.weight': weight_real, 'conv_imag.weight': weight_imag}
        self.load_state_dict(state_dict=new_weights, strict=False)

    def forward(self, tensor: Tensor) -> Tensor:
        assert tensor.dim() == 5, 'Expected (N,2,C,H,W) format.'
        assert tensor.size(1) == 2, 'Expected real/imag to be represented in the second dimension, dim=1.'
        # Separating out the real and imaginary parts without copying memory by using narrow() and squeeze().
        # This increases the speed significantly by removing unnecessary memory copies as in a naive implementation.
        r = tensor.narrow(dim=1, start=0, length=1).squeeze(dim=1)
        i = tensor.narrow(dim=1, start=1, length=1).squeeze(dim=1)

        real = self.conv_real(r) - self.conv_imag(i)
        imag = self.conv_real(i) + self.conv_imag(r)
        return torch.stack([real, imag], dim=1)


class ComplexSpatialDropout2d(nn.Module):
    """
    Spatial dropout for 2D complex convolution.
    Using ordinary dropout2d with merged batch and complex dimensions results in separation of real/complex values.
    Ordinary dropout2d randomly drops out channels with different channels dropped in different slices of the batch.
    To preserve the complex values as one unit, 3D dropout is used, where the real/complex values are always dropped
    together by being in the depth dimension.
    This is efficient since transposing a tensor does not copy memory.
    """
    def __init__(self, p=0.):
        super().__init__()
        self.drop = nn.Dropout3d(p=p)

    def forward(self, tensor: Tensor) -> Tensor:
        assert tensor.dim() == 5, 'Expected (N,2,C,H,W) format.'
        assert tensor.size(1) == 2, 'Expected real/imag to be represented in the second dimension, dim=1.'
        output = tensor.transpose(1, 2)  # Put the channels on dim=1. The 2 is on the depth dimension to stay together.
        output = self.drop(output)  # 3D conv keeps the complex values together.
        output = output.transpose(2, 1)  # Return to original shape. Transpose does not copy memory.
        return output
