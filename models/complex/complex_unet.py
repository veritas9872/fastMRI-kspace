import torch
from torch import nn, Tensor
import torch.nn.functional as F

from ..complex.complex_layers import ComplexConv2d


class ComplexConvLayer(nn.Module):
    def __init__(self, in_chans: int, out_chans: int, stride=1):
        super().__init__()
        self.layer = nn.Sequential(
            ComplexConv2d(in_channels=in_chans, out_channels=out_chans, kernel_size=3, stride=stride, padding=1),
            nn.ReLU()
        )

    def forward(self, tensor: Tensor) -> Tensor:
        assert tensor.dim() == 5 and tensor.size(1) == 2, 'Invalid shape!'
        return self.layer(tensor)


class ComplexConvBlock(nn.Module):
    def __init__(self, in_chans: int, out_chans: int, stride=1):
        super().__init__()
        self.layer = nn.Sequential(
            ComplexConv2d(in_channels=in_chans, out_channels=out_chans, kernel_size=3, stride=stride, padding=1),
            nn.ReLU(),
            ComplexConv2d(in_channels=out_chans, out_channels=out_chans, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, tensor: Tensor) -> Tensor:
        assert tensor.dim() == 5 and tensor.size(1) == 2, 'Invalid shape!'
        return self.layer(tensor)


class ComplexResizeConv(nn.Module):
    def __init__(self, in_chans: int, out_chans: int, scale_factor=2.):
        super().__init__()
        self.layers = nn.Sequential(
            ComplexConv2d(in_channels=in_chans, out_channels=out_chans, kernel_size=3, padding=1),
            nn.ReLU()
        )
        # Interpolate needs to expand only the last two dimensions.
        self.scale_factor = (1, scale_factor, scale_factor)

    def forward(self, tensor: Tensor) -> Tensor:
        assert tensor.dim() == 5 and tensor.size(1) == 2, 'Invalid shape!'
        output = F.interpolate(tensor, scale_factor=self.scale_factor, mode='nearest')
        return self.layers(output)


class ComplexUNet(nn.Module):
    def __init__(self, in_chans: int, out_chans: int, chans: int, num_pool_layers: int):
        super().__init__()
        self.down_sample_layers = nn.ModuleList([ComplexConvBlock(in_chans=in_chans, out_chans=chans, stride=1)])
        ch = chans
        for _ in range(num_pool_layers - 1):
            block = ComplexConvBlock(in_chans=ch, out_chans=ch * 2, stride=2)
            ch *= 2
            self.down_sample_layers += [block]
