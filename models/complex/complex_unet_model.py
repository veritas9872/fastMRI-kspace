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
        self.num_pool_layers = num_pool_layers

        block = ComplexConvBlock(in_chans=in_chans, out_chans=chans, stride=1)
        self.down_layers = nn.ModuleList([block])

        ch = chans
        for _ in range(num_pool_layers - 1):
            block = ComplexConvBlock(in_chans=ch, out_chans=ch * 2, stride=2)
            self.down_layers += [block]
            ch *= 2

        self.mid_conv = ComplexConvBlock(in_chans=ch, out_chans=ch, stride=2)

        self.upscale_layers = nn.ModuleList()
        self.up_layers = nn.ModuleList()

        for _ in range(num_pool_layers - 1):
            deconv = ComplexResizeConv(in_chans=ch, out_chans=ch, scale_factor=2)
            block = ComplexConvBlock(in_chans=ch * 2, out_chans=ch // 2)
            self.upscale_layers += [deconv]
            self.up_layers += [block]
            ch //= 2
        else:
            deconv = ComplexResizeConv(in_chans=ch, out_chans=ch, scale_factor=2)
            block = ComplexConvBlock(in_chans=ch * 2, out_chans=ch)
            self.upscale_layers += [deconv]
            self.up_layers += [block]
            assert chans == ch, 'Channel indexing error!'

        self.final_layers = ComplexConv2d(in_channels=chans, out_channels=out_chans, kernel_size=1)
        assert len(self.down_layers) == len(self.upscale_layers) == len(self.up_layers) == self.num_pool_layers, \
            'Layer number error!'

    def forward(self, tensor: Tensor) -> Tensor:
        assert tensor.dim() == 5 and tensor.size(1) == 2, 'Invalid shape!'
        stack = list()
        output = tensor

        for layer in self.down_layers:
            output = layer(output)
            stack.append(output)

        output = self.mid_conv(output)

        for idx in range(self.num_pool_layers):
            output = self.upscale_layers[idx](output)
            output = torch.cat([output, stack.pop()], dim=2)  # Channels are on dim=2 for complex data.
            output = self.up_layers[idx](output)

        return tensor + self.final_layers(output)
