import torch
from torch import nn
import torch.nn.functional as F

from models.attention import ChannelAttention


class ConvBlock(nn.Module):
    def __init__(self, in_chans, out_chans, stride=2,
                 use_ca=True, reduction=16, use_gap=True, use_gmp=True):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.use_ca = use_ca

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, stride=stride, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, stride=1, padding=1, bias=True)
        )
        self.ca = ChannelAttention(num_chans=out_chans, reduction=reduction, use_gap=use_gap, use_gmp=use_gmp)

    def forward(self, tensor):
        output = self.layers(tensor)
        return self.ca(output) if self.use_ca else output


class UpShuffle(nn.Module):  # Use this later on.
    def __init__(self, num_chans, scale_factor=2):
        super().__init__()
        out_chans = num_chans * scale_factor ** 2
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=num_chans, out_channels=out_chans, kernel_size=3, padding=1, bias=True),
            nn.PixelShuffle(upscale_factor=scale_factor),
            nn.ReLU()
        )

    def forward(self, tensor):
        return self.layer(tensor)


class Interpolate(nn.Module):
    def __init__(self, scale_factor=2, mode='bilinear'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        if mode in ('linear', 'bilinear', 'trilinear'):
            self.align_corners = False
        elif mode == 'bicubic':
            self.align_corners = True
        else:
            self.align_corners = None

    def forward(self, tensor):
        return F.interpolate(tensor, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)


class UNet(nn.Module):
    def __init__(self, in_chans, out_chans, chans, num_pool_layers, num_depth_blocks,  # interp_mode='bilinear',
                 use_residual=True, use_ca=True, reduction=16, use_gap=True, use_gmp=True):

        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.num_depth_blocks = num_depth_blocks  # This must be a positive integer.
        self.use_residual = use_residual  # Residual for the entire model, not for each block.
        kwargs = dict(use_ca=use_ca, reduction=reduction, use_gap=use_gap, use_gmp=use_gmp)

        # self.interpolate = Interpolate(scale_factor=2, mode=interp_mode)

        # First block should have no reduction in feature map size.
        conv = ConvBlock(in_chans=in_chans, out_chans=chans, stride=1, **kwargs)
        self.down_sample_layers = nn.ModuleList([conv])

        ch = chans
        for _ in range(num_pool_layers - 1):
            conv = ConvBlock(in_chans=ch, out_chans=ch * 2, stride=2, **kwargs)
            self.down_sample_layers.append(conv)
            ch *= 2

        # Size reduction happens at the beginning of a block, hence the need for stride here.
        self.mid_conv = ConvBlock(in_chans=ch, out_chans=ch, stride=2, **kwargs)
        self.middle_layers = nn.ModuleList()
        for _ in range(num_depth_blocks - 1):
            self.middle_layers.append(ConvBlock(in_chans=ch, out_chans=ch, stride=1, **kwargs))

        self.upscale_layers = nn.ModuleList()
        self.up_sample_layers = nn.ModuleList()
        for _ in range(num_pool_layers - 1):
            shuffle = UpShuffle(num_chans=ch, scale_factor=2)
            conv = ConvBlock(in_chans=ch * 2, out_chans=ch // 2, stride=1, **kwargs)
            self.upscale_layers.append(shuffle)
            self.up_sample_layers.append(conv)
            ch //= 2
        else:  # Last block of up-sampling.
            shuffle = UpShuffle(num_chans=ch, scale_factor=2)
            conv = ConvBlock(in_chans=ch * 2, out_chans=ch, stride=1, **kwargs)
            self.upscale_layers.append(shuffle)
            self.up_sample_layers.append(conv)
            assert chans == ch, 'Channel indexing error!'

        self.final_layers = nn.Sequential(
            nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=ch, out_channels=out_chans, kernel_size=1)
        )

    def forward(self, tensor):
        stack = list()
        output = tensor

        # Down-Sampling
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)

        # Middle blocks
        output = self.mid_conv(output)
        for layer in self.middle_layers:
            output = output + layer(output)  # Residual layers in the middle.

        # Up-Sampling.
        for upscale, layer in zip(self.upscale_layers, self.up_sample_layers):
            output = upscale(output)
            output = torch.cat([output, stack.pop()], dim=1)
            output = layer(output)

        output = self.final_layers(output)
        return (tensor + output) if self.use_residual else output
