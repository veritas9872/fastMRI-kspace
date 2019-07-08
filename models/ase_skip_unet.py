import torch
from torch import nn
import torch.nn.functional as F

from models.attention import ChannelAttention


class AttConvBlock(nn.Module):
    def __init__(self, in_chans, out_chans, use_att=True, reduction=16, use_gap=True, use_gmp=True):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_chans),
            nn.ReLU()
        )
        self.use_att = use_att
        self.att = ChannelAttention(num_chans=out_chans, reduction=reduction, use_gap=use_gap, use_gmp=use_gmp)

    def forward(self, tensor):
        if self.use_att:
            return self.att(self.layers(tensor))
        else:  # This is actually somewhat redundant since CA already has a no attention option.
            return self.layers(tensor)


class Bilinear(nn.Module):
    def __init__(self, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, tensor):
        return F.interpolate(tensor, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)


class KSSEModule(nn.Module):
    def __init__(self, in_chans, ext_chans, out_chans, min_ext_size, max_ext_size, use_bias=True):
        super().__init__()
        assert isinstance(min_ext_size, int) and isinstance(max_ext_size, int), 'Extractor sizes must be integers.'
        assert 1 <= min_ext_size <= max_ext_size, 'Invalid extractor sizes.'
        assert (min_ext_size % 2) and (max_ext_size % 2), 'Extractor sizes must be odd numbers.'

        self.ext_layers = nn.ModuleList()

        if min_ext_size <= 1:
            conv = nn.Conv2d(in_chans, ext_chans, kernel_size=1, bias=use_bias)
            self.ext_layers.append(conv)

        min_ext_size = max(min_ext_size, 3)
        for size in range(min_ext_size, max_ext_size + 1, 2):
            # 1NN1 pattern.
            conv = nn.Sequential(  # Number of channels is different for the two layers.
                nn.Conv2d(in_channels=in_chans, out_channels=ext_chans,
                          kernel_size=(1, size), padding=(0, size // 2), bias=use_bias),
                nn.Conv2d(in_channels=ext_chans, out_channels=ext_chans,  # Takes previous output as input.
                          kernel_size=(size, 1), padding=(size // 2, 0), bias=use_bias)
            )
            self.ext_layers.append(conv)

        self.relu = nn.ReLU()
        self.conv1x1 = nn.Conv2d(in_channels=ext_chans * len(self.ext_layers), out_channels=out_chans, kernel_size=1)

    def forward(self, tensor):
        outputs = torch.cat([ext(tensor) for ext in self.ext_layers], dim=1)
        outputs = self.relu(outputs)
        outputs = self.conv1x1(outputs)
        outputs = self.relu(outputs)
        return outputs


class UNetSkipKSSE(nn.Module):
    def __init__(self, in_chans, out_chans, chans, num_pool_layers, ext_chans, min_ext_size, max_ext_size, use_ext_bias,
                 pool='avg', use_skip=True, use_att=True, reduction=16, use_gap=True, use_gmp=True):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.use_skip = use_skip

        pool = pool.lower()
        if pool == 'avg':
            self.pool = nn.AvgPool2d(2)
        elif pool == 'max':
            self.pool = nn.MaxPool2d(2)
        else:
            raise ValueError('`pool` must be either `avg` or `max`.')

        self.interpolate = Bilinear(scale_factor=2)

        conv_kwargs = dict(use_att=use_att, reduction=reduction, use_gap=use_gap, use_gmp=use_gmp)

        ksse = KSSEModule(in_chans=in_chans, ext_chans=ext_chans, out_chans=chans,
                          min_ext_size=min_ext_size, max_ext_size=max_ext_size, use_bias=use_ext_bias)

        self.down_sample_layers = nn.ModuleList([ksse])

        ch = chans
        for _ in range(num_pool_layers - 1):
            conv = AttConvBlock(in_chans=ch, out_chans=ch * 2, **conv_kwargs)
            self.down_sample_layers.append(conv)
            ch *= 2

        self.conv_mid = AttConvBlock(in_chans=ch, out_chans=ch, **conv_kwargs)

        self.up_sample_layers = nn.ModuleList()
        for _ in range(num_pool_layers - 1):
            conv = AttConvBlock(in_chans=ch * 2, out_chans=ch // 2, **conv_kwargs)
            self.up_sample_layers.append(conv)
            ch //= 2
        else:
            conv = AttConvBlock(in_chans=ch * 2, out_chans=ch, **conv_kwargs)
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


