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


class SignalExtractor(nn.Module):
    def __init__(self, in_chans, out_chans, ext_chans, min_ext_size, max_ext_size, ext_mode='1NN1'):
        super().__init__()
        assert isinstance(min_ext_size, int) and isinstance(max_ext_size, int), 'Extractor sizes must be integers.'
        assert 1 <= min_ext_size <= max_ext_size, 'Invalid extractor sizes.'
        assert (min_ext_size % 2) and (max_ext_size % 2), 'Extractor sizes must be odd numbers.'

        self.ext_layers = nn.ModuleList()

        if min_ext_size <= 1:
            conv = nn.Conv2d(in_chans, ext_chans, kernel_size=1)
            self.ext_layers.append(conv)

        min_ext_size = max(min_ext_size, 3)
        for size in range(min_ext_size, max_ext_size + 1, 2):
            # 1NN1 pattern.
            if ext_mode.upper() == '1NN1':
                conv = nn.Sequential(  # Number of channels is different for the two layers.
                    nn.Conv2d(in_channels=in_chans, out_channels=ext_chans,
                              kernel_size=(1, size), padding=(0, size // 2)),
                    nn.Conv2d(in_channels=ext_chans, out_channels=ext_chans,  # Takes previous output as input.
                              kernel_size=(size, 1), padding=(size // 2, 0))
                )
            elif ext_mode.upper() == 'N11N':
                conv = nn.Sequential(  # Number of channels is different for the two layers.
                    nn.Conv2d(in_channels=in_chans, out_channels=ext_chans,  # Takes previous output as input.
                              kernel_size=(size, 1), padding=(size // 2, 0)),
                    nn.Conv2d(in_channels=ext_chans, out_channels=ext_chans,
                              kernel_size=(1, size), padding=(0, size // 2))
                )
            else:
                raise ValueError('Invalid mode!')

            self.ext_layers.append(conv)

        self.relu = nn.ReLU()
        self.conv1x1 = nn.Conv2d(in_channels=ext_chans * len(self.ext_layers), out_channels=out_chans, kernel_size=1)

    def forward(self, tensor):
        outputs = torch.cat([ext(tensor) for ext in self.ext_layers], dim=1)
        outputs = self.relu(outputs)
        outputs = self.conv1x1(outputs)
        outputs = self.relu(outputs)
        return outputs


class UNetModelKSSE(nn.Module):
    """
    UNet model with attention (channel and spatial), skip connections in blocks, and residual final outputs.
    All mentioned capabilities are optional and tunable. Also has k-space signal extractor.
    Normalization is performed by Group Normalization because of small expected mini-batch size.
    The down-sampling is performed either by max-pooling or avg-pooling.
    Up-sampling is done by bilinear interpolation.
    Frankly, I think that there are far too many parameters to control. Removing some might be beneficial.
    """
    def __init__(self, in_chans, out_chans, chans, num_pool_layers, num_groups, use_residual=True,
                 pool_type='avg', use_skip=False, min_ext_size=1, max_ext_size=9, ext_mode='1NN1',
                 use_ca=True, reduction=16, use_gap=True, use_gmp=True,
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

        # Replace here!!!
        ksse = SignalExtractor(in_chans=in_chans, out_chans=chans, ext_chans=chans,  # Simplification.
                               min_ext_size=min_ext_size, max_ext_size=max_ext_size, ext_mode=ext_mode)
        self.down_sample_layers = nn.ModuleList([ksse])

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
