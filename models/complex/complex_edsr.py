import torch
from torch import nn, Tensor

from ..complex.complex_conv import ComplexConv2d


class ComplexConvLayer(nn.Module):
    def __init__(self, in_chans: int, out_chans: int, negative_slope=0.01):
        super().__init__()
        self.layer = nn.Sequential(
            ComplexConv2d(in_channels=in_chans, out_channels=out_chans, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=negative_slope)
        )

    def forward(self, tensor: Tensor) -> Tensor:
        # assert tensor.dim() == 5 and tensor.size(1) == 2, 'Invalid shape!'
        return self.layer(tensor)


class ComplexResBlock(nn.Module):
    def __init__(self, num_chans: int, negative_slope=0.01, res_scale=1.):
        super().__init__()
        self.res_scale = res_scale
        self.layer = nn.Sequential(
            ComplexConv2d(in_channels=num_chans, out_channels=num_chans, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=negative_slope),  # Equivalent to Complex Leaky ReLU.
            ComplexConv2d(in_channels=num_chans, out_channels=num_chans, kernel_size=3, padding=1)
        )

    def forward(self, tensor: Tensor) -> Tensor:
        # assert tensor.dim() == 5 and tensor.size(1) == 2, 'Invalid shape!'
        return tensor + self.res_scale * self.layer(tensor)


class ComplexEDSR(nn.Module):
    def __init__(self, in_chans, out_chans, chans, num_blocks, negative_slope, res_scale):
        super().__init__()
        self.head = ComplexConvLayer(in_chans=in_chans, out_chans=chans, negative_slope=negative_slope)
        body = list()
        for _ in range(num_blocks):
            res = ComplexResBlock(num_chans=chans, negative_slope=negative_slope, res_scale=res_scale)
            body.append(res)
        else:
            self.body = nn.Sequential(*body)

        # Not quite identical to EDSR here. Using 1x1 convolutions at the end.
        self.tail = ComplexConv2d(in_channels=chans, out_channels=out_chans, kernel_size=1)

    def forward(self, tensor):
        assert tensor.dim() == 5 and tensor.size(1) == 2, 'Invalid shape!'
        output = self.head(tensor)
        output = output + self.body(output)
        return tensor + self.tail(output)  # Residual Network.
