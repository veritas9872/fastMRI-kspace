import torch
from torch import nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, num_chans, reduction=16, use_gap=True, use_gmp=True):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(output_size=(1, 1))  # Global Average Pooling.
        self.gmp = nn.AdaptiveMaxPool2d(output_size=(1, 1))  # Global Maximum Pooling.

        self.use_gap = use_gap
        self.use_gmp = use_gmp

        self.layer = nn.Sequential(
            nn.Linear(in_features=num_chans, out_features=num_chans // reduction),
            nn.ReLU(),
            nn.Linear(in_features=num_chans // reduction, out_features=num_chans)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, tensor):
        batch, chans, _, _ = tensor.shape
        if self.use_gap and self.use_gmp:
            gap = self.gap(tensor).view(batch, chans)
            gmp = self.gmp(tensor).view(batch, chans)
            # Maybe batch-norm the two pooling types to make their scales more similar.
            # This might make training slower, however.
            features = self.layer(gap) + self.layer(gmp)
            att = self.sigmoid(features).view(batch, chans, 1, 1)

        elif self.use_gap:
            gap = self.gap(tensor).view(batch, chans)
            features = self.layer(gap)
            att = self.sigmoid(features).view(batch, chans, 1, 1)

        elif self.use_gmp:
            gmp = self.gmp(tensor).view(batch, chans)
            features = self.layer(gmp)
            att = self.sigmoid(features).view(batch, chans, 1, 1)

        else:
            att = 1

        return tensor * att


class AsymmetricSignalExtractor(nn.Module):
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

        # I think that 3x3 kernels do not need the 1x3, 3x1 separation.
        if min_ext_size <= 3 <= max_ext_size:
            conv = nn.Conv2d(in_chans, ext_chans, kernel_size=3, padding=1, bias=use_bias)
            self.ext_layers.append(conv)

        min_ext_size = max(min_ext_size, 5)
        # The cases where the maximum size is smaller than 5 will automatically be dealt with by the for-loop.
        for size in range(min_ext_size, max_ext_size + 1, 2):
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


class ConvBlock(nn.Module):
    def __init__(self, in_chans, out_chans):
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

    def forward(self, tensor):
        return self.layers(tensor)


class ConvAttention(nn.Module):
    def __init__(self, in_chans, out_chans, reduction=16):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans

        # Conv layers
        self.cbrlayers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_chans),
            nn.ReLU()
        )

        self.gap = nn.AdaptiveAvgPool2d(output_size=(1, 1))  # Global Average Pooling.
        self.gmp = nn.AdaptiveMaxPool2d(output_size=(1, 1))  # Global Maximum Pooling.

        # Attention layers
        self.alayers = nn.Sequential(
            nn.Linear(in_features=out_chans, out_features=out_chans // reduction),
            nn.ReLU(),
            nn.Linear(in_features=out_chans // reduction, out_features=out_chans)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, tensor):

        conv_tensor = self.cbrlayers(tensor)

        batch, chans, _, _ = conv_tensor.shape
        gap = self.gap(conv_tensor).view(batch, chans)
        gmp = self.gmp(conv_tensor).view(batch, chans)

        features = self.alayers(gap) + self.alayers(gmp)
        att = self.sigmoid(features).view(batch, chans, 1, 1)

        return conv_tensor * att


class Bilinear(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, tensor):
        return F.interpolate(tensor, scale_factor=2, mode='bilinear', align_corners=False)


class UnetASE(nn.Module):
    def __init__(self, in_chans, out_chans, ext_chans, chans, num_pool_layers,
                 min_ext_size, max_ext_size, use_ext_bias=True):

        super().__init__()
        self.extractor = AsymmetricSignalExtractor(
            in_chans=in_chans, out_chans=chans, ext_chans=ext_chans,
            min_ext_size=min_ext_size, max_ext_size=max_ext_size, use_bias=use_ext_bias)

        self.pool = nn.AvgPool2d(2)
        self.interp = Bilinear()
        self.input_att = ChannelAttention(num_chans=chans, reduction=16, use_gap=True, use_gmp=True)
        self.down_sample_layers = nn.ModuleList()
        ch = chans

        for n in range(num_pool_layers - 1):
            conv = ConvAttention(in_chans=ch, out_chans=ch * 2)
            self.down_sample_layers.append(conv)
            ch *= 2

        self.conv_mid = ConvAttention(in_chans=ch, out_chans=ch)

        self.up_sample_layers = nn.ModuleList()
        for n in range(num_pool_layers - 1):
            conv = ConvAttention(in_chans=ch * 2, out_chans=ch // 2)
            self.up_sample_layers.append(conv)
            ch //= 2
        else:
            conv = ConvAttention(in_chans=ch * 2, out_chans=ch)
            self.up_sample_layers.append(conv)

        self.conv_last = nn.Conv2d(in_channels=ch, out_channels=out_chans, kernel_size=1)

    def forward(self, tensor):
        stack = list()
        output = self.extractor(tensor)
        # Added channel attention to input layer after feature extraction, compression, and ReLU.
        output = self.input_att(output)
        stack.append(output)
        output = self.pool(output)

        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = self.pool(output)

        output = self.conv_mid(output)

        for layer in self.up_sample_layers:
            output = self.interp(output)
            output = torch.cat([output, stack.pop()], dim=1)
            output = layer(output)

        return self.conv_last(output)
