import torch
from torch import nn, Tensor
from torch.nn.init import _calculate_fan_in_and_fan_out

import numpy as np


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

        # Weight initialization. Somewhat inelegant in style but it works (I think...).
        weight_real, weight_imag = self._get_weight_inits(weight_shape=self.conv_real.weight.shape, method='kaiming')
        new_weights = {'conv_real.weight': weight_real, 'conv_imag.weight': weight_imag}
        self.load_state_dict(state_dict=new_weights, strict=False)

    def _get_weight_inits(self, weight_shape: tuple, method='kaiming'):
        fan_in, fan_out = _calculate_fan_in_and_fan_out(torch.zeros(size=weight_shape))
        if method == 'xavier':
            mode = 1 / np.sqrt(fan_in + fan_out)
        elif method == 'kaiming':
            mode = 1 / np.sqrt(fan_in)
        else:
            raise NotImplementedError('Invalid initialization method.')

        magnitude = np.random.rayleigh(scale=mode, size=weight_shape)
        phase = np.random.uniform(low=-np.pi, high=np.pi, size=weight_shape)
        weight_real = torch.from_numpy((magnitude * np.cos(phase)).astype(np.float32))
        weight_imag = torch.from_numpy((magnitude * np.sin(phase)).astype(np.float32))
        return weight_real, weight_imag

    def forward(self, tensor: Tensor) -> Tensor:
        assert tensor.dim() == 5, 'Expected (N,2,C,H,W) format.'
        assert tensor.size(1) == 2, 'Expected real/imag to be represented in the second dimension.'
        real = self.conv_real(tensor[:, 0]) - self.conv_imag(tensor[:, 1])
        imag = self.conv_real(tensor[:, 1]) - self.conv_imag(tensor[:, 0])
        return torch.stack([real, imag], dim=1)
