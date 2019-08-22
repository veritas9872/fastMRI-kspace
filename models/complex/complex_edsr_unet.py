import torch
from torch import nn, Tensor
import torch.nn.functional as F

from ..complex.complex_conv import ComplexConv2d


class ComplexConvLayer(nn.Module):
    def __init__(self, in_chans: int, out_chans: int, stride: int, negative_slope=0.01):
        super().__init__()
        self.layer = nn.Sequential(
            ComplexConv2d(in_channels=in_chans, out_channels=out_chans, kernel_size=3, stride=stride, padding=1),
            nn.LeakyReLU(negative_slope=negative_slope)
        )

    def forward(self, tensor: Tensor) -> Tensor:
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
        assert tensor.dim() == 5 and tensor.size(1) == 2, 'Invalid shape!'
        return tensor + self.res_scale * self.layer(tensor)


class ComplexResizeConv(nn.Module):
    def __init__(self, in_chans: int, out_chans: int, scale_factor=2., negative_slope=0.01, res_scale=1.):
        super().__init__()
        self.layers = nn.Sequential(
            ComplexConv2d(in_channels=in_chans, out_channels=out_chans, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=negative_slope)
        )
        # Interpolate needs to expand only the last two dimensions.
        self.scale_factor = (1, scale_factor, scale_factor)

    def forward(self, tensor: Tensor) -> Tensor:
        assert tensor.dim() == 5 and tensor.size(1) == 2, 'Invalid shape!'
        output = F.interpolate(tensor, scale_factor=self.scale_factor, mode='nearest')
        return self.layers(output)


class ComplexEDSRUNet(nn.Module):
    def __init__(self, in_chans: int, out_chans: int, chans: int, num_pool_layers: int,
                 num_depth_blocks: int, negative_slope=0.01, res_scale=0.1):
        super().__init__()
        self.num_pool_layers = num_pool_layers
        conv = ComplexConvLayer(in_chans=in_chans, out_chans=chans, stride=1, negative_slope=negative_slope)
        res = ComplexResBlock(num_chans=chans, negative_slope=negative_slope, res_scale=res_scale)
        self.down_reshape_layers = nn.ModuleList([conv])
        self.down_res_blocks = nn.ModuleList([res])

        ch = chans
        for _ in range(num_pool_layers - 1):
            conv = ComplexConvLayer(in_chans=ch, out_chans=ch * 2, stride=2, negative_slope=negative_slope)
            res = ComplexResBlock(num_chans=ch * 2, negative_slope=negative_slope, res_scale=res_scale)
            self.down_reshape_layers.append(conv)
            self.down_res_blocks.append(res)
            ch *= 2

        self.mid_conv = ComplexConvLayer(in_chans=ch, out_chans=ch, stride=2, negative_slope=negative_slope)

        mid_res_blocks = list()
        for _ in range(num_depth_blocks):
            res = ComplexResBlock(num_chans=ch, negative_slope=negative_slope, res_scale=res_scale)
            mid_res_blocks.append(res)
        self.mid_res_blocks = nn.Sequential(*mid_res_blocks)

        self.upscale_layers = nn.ModuleList()
        self.up_reshape_layers = nn.ModuleList()
        self.up_res_blocks = nn.ModuleList()
        for _ in range(num_pool_layers - 1):
            deconv = ComplexResizeConv(in_chans=ch, out_chans=ch, scale_factor=2,
                                       negative_slope=negative_slope, res_scale=res_scale)
            conv = ComplexConvLayer(in_chans=ch * 2, out_chans=ch // 2, stride=1, negative_slope=negative_slope)
            res = ComplexResBlock(num_chans=ch // 2, negative_slope=negative_slope, res_scale=res_scale)
            self.upscale_layers.append(deconv)
            self.up_reshape_layers.append(conv)
            self.up_res_blocks.append(res)
            ch //= 2
        else:
            deconv = ComplexResizeConv(in_chans=ch, out_chans=ch, scale_factor=2,
                                       negative_slope=negative_slope, res_scale=res_scale)
            conv = ComplexConvLayer(in_chans=ch * 2, out_chans=ch, stride=1, negative_slope=negative_slope)
            res = ComplexResBlock(num_chans=ch, negative_slope=negative_slope, res_scale=res_scale)
            self.upscale_layers.append(deconv)
            self.up_reshape_layers.append(conv)
            self.up_res_blocks.append(res)
            assert chans == ch, 'Channel indexing error!'

        self.final_layers = ComplexConv2d(in_channels=chans, out_channels=out_chans, kernel_size=1)
        assert len(self.down_reshape_layers) == len(self.down_res_blocks) == len(self.upscale_layers) \
            == len(self.up_reshape_layers) == len(self.up_res_blocks) == self.num_pool_layers, 'Layer number error!'

    def forward(self, tensor: Tensor) -> Tensor:
        assert tensor.dim() == 5 and tensor.size(1) == 2, 'Invalid shape!'
        stack = list()
        output = tensor

        for idx in range(self.num_pool_layers):
            output = self.down_reshape_layers[idx](output)
            output = self.down_res_blocks[idx](output)
            stack.append(output)

        output = self.mid_conv(output)
        output = output + self.mid_res_blocks(output)  # Residual of middle blocks. Same as EDSR.

        for idx in range(self.num_pool_layers):
            output = self.upscale_layers[idx](output)
            output = torch.cat([output, stack.pop()], dim=2)  # Channels are on dim=2 for complex data.
            output = self.up_reshape_layers[idx](output)
            output = self.up_res_blocks[idx](output)

        return tensor + self.final_layers(output)

