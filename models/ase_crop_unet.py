import torch
from torch import nn
import torch.nn.functional as F

from models.attention import ChannelAttention


class LayerCenterCrop(nn.Module):
    def __init__(self, scale, pad=0):
        super().__init__()

        # Type checking for pad.
        if isinstance(pad, int):
            pad = (pad, pad, pad, pad)
        elif isinstance(pad, (list, tuple)):
            if len(pad) == 2:
                pad = (pad[0], pad[0], pad[1], pad[1])
            elif len(pad) != 4:
                raise ValueError(
                    f'Invalid pad length. `pad` must have either 2 or 4 elements but input has {len(pad)} elements.')
        else:
            raise TypeError('Invalid pad input. `pad` must be either an integer or list/tuple of length 2 or 4.')

        self.pad = pad
        self.scale = scale

    def forward(self, tensor):
        top = tensor.size(-2) - (tensor.size(-2) // self.scale)
        bottom = top + tensor.size(-2)
        top -= self.pad[0]
        bottom += self.pad[1]
        left = tensor.size(-1) - (tensor.size(-1) // self.scale)
        right = left + tensor.size(-1)
        left -= self.pad[2]
        right += self.pad[3]

        return tensor[..., top:bottom, left:right]


class SignalExtractor(nn.Module):  # TODO: Add new extractor types and methods
    def __init__(self, in_chans, out_chans, ext_chans, min_ext_size, max_ext_size, use_bias=True):
        super().__init__()
        assert isinstance(min_ext_size, int) and isinstance(max_ext_size, int), 'Extractor sizes must be integers.'
        assert 1 <= min_ext_size <= max_ext_size, 'Invalid extractor sizes.'
        assert (min_ext_size % 2) and (max_ext_size % 2), 'Extractor sizes must be odd numbers.'

        # Added 1x1 convolution, not specified, so very bad for API. Add specification later.
        self.ext_layers = nn.ModuleList()

        if min_ext_size <= 1:
            conv = nn.Conv2d(in_chans, ext_chans, kernel_size=1, bias=use_bias)
            self.ext_layers.append(conv)

        min_ext_size = max(min_ext_size, 3)
        # print(f'min_ext_size: {min_ext_size}')  # For debugging
        # The cases where the maximum size is smaller than 5 will automatically be dealt with by the for-loop.
        for size in range(min_ext_size, max_ext_size + 1, 2):
            # 1NN1 style extractor.
            # Left-right, then up-down. This is because of the sampling pattern.
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
        else:
            return self.layers(tensor)


class Bilinear(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, tensor):
        return F.interpolate(tensor, scale_factor=2, mode='bilinear', align_corners=False)


class UNetCropASE(nn.Module):  # This model needs testing. It has not been tested at all.
    def __init__(self, in_chans, out_chans, ext_chans, chans, num_pool_layers,
                 min_ext_size, max_ext_size, use_ext_bias=True, use_block_att=True):

        super().__init__()
        self.extractor = SignalExtractor(
            in_chans=in_chans, out_chans=chans, ext_chans=ext_chans,
            min_ext_size=min_ext_size, max_ext_size=max_ext_size, use_bias=use_ext_bias)

        self.interp = Bilinear()
        self.use_block_att = use_block_att
        self.down_sample_layers = nn.ModuleList()
        self.crop = LayerCenterCrop(scale=2)
        ch = chans

        for n in range(num_pool_layers - 1):
            conv = AttConvBlock(in_chans=ch, out_chans=ch * 2, use_att=use_block_att)
            self.down_sample_layers.append(conv)
            ch *= 2

        self.conv_mid = AttConvBlock(in_chans=ch, out_chans=ch, use_att=use_block_att)

        self.up_sample_layers = nn.ModuleList()
        for n in range(num_pool_layers - 1):
            conv = AttConvBlock(in_chans=ch * 2, out_chans=ch // 2, use_att=use_block_att)
            self.up_sample_layers.append(conv)
            ch //= 2
        else:
            conv = AttConvBlock(in_chans=ch * 2, out_chans=ch, use_att=use_block_att)
            self.up_sample_layers.append(conv)

        self.conv_last = nn.Conv2d(in_channels=ch, out_channels=out_chans, kernel_size=1)

    def forward(self, tensor):
        stack = list()
        output = self.extractor(tensor)
        stack.append(output)
        output = self.crop(output)

        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = self.crop(output)

        output = self.conv_mid(output)

        for layer in self.up_sample_layers:
            output = self.interp(output)
            output = torch.cat([output, stack.pop()], dim=1)
            output = layer(output)

        return self.conv_last(output)
