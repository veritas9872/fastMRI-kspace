import torch
from torch import nn
import torch.nn.functional as F

from models.attention import ChannelAttention


class AttConvBlockGN(nn.Module):
    def __init__(self, in_chans, out_chans, num_groups, use_att=True, reduction=16, use_gap=True, use_gmp=True):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_chans),
            nn.ReLU()
        )

        self.use_att = use_att if (use_gap or use_gmp) else False
        self.att = ChannelAttention(num_chans=out_chans, reduction=reduction, use_gap=use_gap, use_gmp=use_gmp)

    def forward(self, tensor):
        return self.att(self.layers(tensor)) if self.use_att else self.layers(tensor)


class Bilinear(nn.Module):
    def __init__(self, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, tensor):
        return F.interpolate(tensor, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)


class UNetSkipGN(nn.Module):
    def __init__(self, in_chans, out_chans, chans, num_pool_layers, num_groups, pool_type='avg', use_skip=True,
                 use_att=True, reduction=16, use_gap=True, use_gmp=True):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.use_skip = use_skip

        pool_type = pool_type.lower()
        if pool_type == 'avg':
            self.pool = nn.AvgPool2d(2)
        elif pool_type == 'max':
            self.pool = nn.MaxPool2d(2)
        else:
            raise ValueError('`pool` must be either `avg` or `max`.')

        self.interpolate = Bilinear(scale_factor=2)

        kwargs = dict(num_groups=num_groups, use_att=use_att, reduction=reduction, use_gap=use_gap, use_gmp=use_gmp)
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

        self.conv_last = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=1),
            nn.Conv2d(ch, out_chans, kernel_size=1),
        )

    def forward(self, tensor):
        stack1 = list()
        stack2 = list()

        # Maybe add signal extractor later.
        output = tensor

        # Down-Sampling
        for layer in self.down_sample_layers:
            output = layer(output)
            stack1.append(output)
            output = self.pool(output)
            stack2.append(output)

        # Bottom Block
        output = self.conv_mid(output)

        # Up-Sampling.
        for layer in self.up_sample_layers:
            # Residual. Different from Dual-Frame UNet.
            output = output + stack2.pop() if self.use_skip else output
            output = self.interpolate(output)
            output = torch.cat([output, stack1.pop()], dim=1)
            output = layer(output)

        return self.conv_last(output)


