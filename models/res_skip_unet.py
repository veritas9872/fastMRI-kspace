import torch
from torch import nn
import torch.nn.functional as F

from models.attention import ChannelAttention, SpatialAttention


class AttConvBlockGN(nn.Module):
    def __init__(self, in_chans, out_chans, num_groups,
                 use_ca=True, reduction=16, use_gap=True, use_gmp=True,
                 use_sa=True, sa_kernel_size=7, sa_dilation=1, use_cap=True, use_cmp=True):

        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans

        self.use_ca = use_ca
        self.use_sa = use_sa

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_chans),
            nn.LeakyReLU()
        )

        self.ca = ChannelAttention(num_chans=out_chans, reduction=reduction, use_gap=use_gap, use_gmp=use_gmp)
        self.sa = SpatialAttention(kernel_size=sa_kernel_size, dilation=sa_dilation, use_cap=use_cap, use_cmp=use_cmp)

    def forward(self, tensor):
        output = self.layers(tensor)

        # Using fixed ordering of channel and spatial attention.
        # Not sure if this is best but this is how it is done in CBAM.
        if self.use_ca:
            output = self.ca(output)

        if self.use_sa:
            output = self.sa(output)

        return output


class Bilinear(nn.Module):
    def __init__(self, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, tensor):
        return F.interpolate(tensor, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)


class UNetModel(nn.Module):
    """
    UNet model with attention (channel and spatial), skip connections in blocks, and residual final outputs.
    All mentioned capabilities are optional and tunable.
    Normalization is performed by Group Normalization because of small expected mini-batch size.
    The down-sampling is performed either by max-pooling or avg-pooling.
    Up-sampling is done by bilinear interpolation.
    Frankly, I think that there are far too many parameters to control. Removing some might be beneficial.
    """
    def __init__(self, in_chans, out_chans, chans, num_pool_layers, num_groups, use_residual=True,
                 pool_type='avg', use_skip=False, use_ca=True, reduction=16, use_gap=True, use_gmp=True,
                 use_sa=True, sa_kernel_size=7, sa_dilation=1, use_cap=True, use_cmp=True):

        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.use_residual = use_residual
        self.use_skip = use_skip

        pool_type = pool_type.lower()
        if pool_type == 'avg':
            self.pool = nn.AvgPool2d(2)
        elif pool_type == 'max':
            self.pool = nn.MaxPool2d(2)
        else:
            raise ValueError('`pool` must be either `avg` or `max`.')

        self.interpolate = Bilinear(scale_factor=2)

        kwargs = dict(
            num_groups=num_groups, use_ca=use_ca, reduction=reduction, use_gap=use_gap, use_gmp=use_gmp,
            use_sa=use_sa, sa_kernel_size=sa_kernel_size, sa_dilation=sa_dilation, use_cap=use_cap, use_cmp=use_cmp)

        conv = AttConvBlockGN(in_chans=in_chans, out_chans=chans, **kwargs)
        self.down_sample_layers = nn.ModuleList([conv])

        ch = chans
        for _ in range(num_pool_layers - 1):
            conv = AttConvBlockGN(in_chans=ch, out_chans=ch * 2, **kwargs)
            self.down_sample_layers.append(conv)
            ch *= 2

        self.conv_mid = AttConvBlockGN(in_chans=ch, out_chans=ch, **kwargs)

        self.up_sample_layers = nn.ModuleList()
        for _ in range(num_pool_layers - 1):
            conv = AttConvBlockGN(in_chans=ch * 2, out_chans=ch // 2, **kwargs)
            self.up_sample_layers.append(conv)
            ch //= 2
        else:
            conv = AttConvBlockGN(in_chans=ch * 2, out_chans=ch, **kwargs)
            self.up_sample_layers.append(conv)
            assert chans == ch, 'Channel indexing error!'

        # Maybe this output structure makes it hard to learn? I have no idea...
        self.conv_last = nn.Conv2d(ch, out_chans, kernel_size=1)

    def forward(self, tensor):
        stack = list()
        skips = list()

        # Maybe add signal extractor later.
        output = tensor

        # Down-Sampling
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = self.pool(output)

            if self.use_skip:
                skips.append(output)

        # Bottom Block
        output = self.conv_mid(output)

        # Up-Sampling.
        for layer in self.up_sample_layers:

            if self.use_skip:
                output = output + skips.pop()

            output = self.interpolate(output)
            output = torch.cat([output, stack.pop()], dim=1)
            output = layer(output)

        if self.use_residual:
            output = tensor + self.conv_last(output)
        else:
            output = self.conv_last(output)

        return output



