import torch
from torch import nn
import torch.nn.functional as F

from models.attention import ChannelAttention


class ConvBlock(nn.Module):
    def __init__(self, in_chans, out_chans, num_groups, negative_slope=0.01, stride=2,
                 use_ca=True, reduction=16, use_gap=True, use_gmp=True):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.use_ca = use_ca

        self.layers = nn.Sequential(
            # Use bias in conv since group-norm batches along many channels.
            nn.Conv2d(in_chans, out_chans, kernel_size=3, stride=stride, padding=1, bias=True),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_chans),
            nn.LeakyReLU(negative_slope=negative_slope),

            # Down-sampling using stride 2 convolution.
            nn.Conv2d(out_chans, out_chans, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_chans),
            nn.LeakyReLU(negative_slope=negative_slope)
        )
        self.ca = ChannelAttention(num_chans=out_chans, reduction=reduction, use_gap=use_gap, use_gmp=use_gmp)

    def forward(self, tensor):
        output = self.layers(tensor)
        return self.ca(output) if self.use_ca else output


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
    def __init__(self, in_chans, out_chans, chans, num_pool_layers, num_groups, negative_slope=0.01,
                 use_residual=True, interp_mode='bilinear', use_ca=True, reduction=16, use_gap=True, use_gmp=True):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.use_residual = use_residual
        kwargs = dict(num_groups=num_groups, negative_slope=negative_slope,
                      use_ca=use_ca, reduction=reduction, use_gap=use_gap, use_gmp=use_gmp)

        self.interpolate = Interpolate(scale_factor=2, mode=interp_mode)

        # First block should have no reduction in feature map size.
        conv = ConvBlock(in_chans=in_chans, out_chans=chans, stride=1, **kwargs)
        self.down_sample_layers = nn.ModuleList([conv])

        ch = chans
        for _ in range(num_pool_layers - 1):
            conv = ConvBlock(in_chans=ch, out_chans=ch * 2, stride=2, **kwargs)
            self.down_sample_layers.append(conv)
            ch *= 2

        # Size reduction happens at the beginning of a block, hence the need for stride here.
        self.conv_mid = ConvBlock(in_chans=ch, out_chans=ch, stride=2, **kwargs)

        self.up_sample_layers = nn.ModuleList()
        for _ in range(num_pool_layers - 1):
            conv = ConvBlock(in_chans=ch * 2, out_chans=ch // 2, stride=1, **kwargs)
            self.up_sample_layers.append(conv)
            ch //= 2
        else:  # Last block of up-sampling.
            conv = ConvBlock(in_chans=ch * 2, out_chans=ch, stride=1, **kwargs)
            self.up_sample_layers.append(conv)
            assert chans == ch, 'Channel indexing error!'

        self.conv_last = nn.Conv2d(ch, out_chans, kernel_size=1)

    def forward(self, tensor):
        stack = list()
        output = tensor

        # Down-Sampling
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)

        # Bottom Block
        output = self.conv_mid(output)

        # Up-Sampling.
        for layer in self.up_sample_layers:
            output = self.interpolate(output)
            output = torch.cat([output, stack.pop()], dim=1)
            output = layer(output)

        output = self.conv_last(output)
        return (tensor + output) if self.use_residual else output
