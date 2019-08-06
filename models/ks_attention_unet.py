import torch
from torch import nn
import torch.nn.functional as F
import math


class AsymmetricSignalExtractor(nn.Module):
    def __init__(self, in_chans, out_chans, ext_chans, min_ext_size, max_ext_size, use_bias=True):
        super().__init__()
        assert isinstance(min_ext_size, int) and isinstance(max_ext_size, int), 'Extractor sizes must be integers.'
        assert 3 <= min_ext_size <= max_ext_size, 'Invalid extractor sizes.'
        assert (min_ext_size // 2) and (max_ext_size // 2), 'Extractor sizes must be odd numbers.'

        # Added 1x1 convolution, not specified, so very bad for API. Add specification later.
        # Why does 1x1 convolution exist?
        # Check if things work without it
        self.ext_layers = nn.ModuleList([nn.Conv2d(in_chans, ext_chans, kernel_size=1, bias=use_bias)])
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


class CCAMBlock(nn.Module):
    def __init__(self, in_chans, pool_size):
        super().__init__()
        self.in_chans = in_chans

        self.avgpool = nn.AvgPool2d(pool_size)
        # print(f'in_chans: {in_chans}')
        self.MLP1 = nn.Linear(in_chans, int(in_chans / 16))
        self.MLP2 = nn.Linear(int(in_chans / 16), in_chans)
        self.activ = nn.Sigmoid()

    # self.layers = nn.Sequential(
    #     nn.AvgPool2d(pool_size),
    #     nn.Linear(in_chans, int(in_chans / 2)),
    #     nn.Linear(int(in_chans / 2), in_chans),
    #     nn.Sigmoid()
    # )
    def forward(self, tensor):
        # print(f'1 shape:{tensor.shape}')
        y = self.avgpool(tensor)
        y = y.view(tensor.shape[0], 1, 1, -1)
        # print(f'2 shape:{y.shape}')
        y = self.MLP1(y)
        # print(f'3 shape:{y.shape}')
        y = self.MLP2(y)
        # print(f'4 shape:{y.shape}')
        y = self.activ(y)
        # print(f'5 shape:{y.shape}')
        y = y.view(y.shape[0], -1, 1, 1)
        y = torch.mul(tensor, y)
        # print(f'final shape: {y.shape}')

        return y


class ChannelAttention(nn.Module):
    def __init__(self, in_chans):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.MLP1 = nn.Linear(in_chans, int(in_chans / 16))
        self.MLP2 = nn.Linear(int(in_chans / 16), in_chans)
        self.sigmoid = nn.Sigmoid()

    def forward(self, tensor):
        y = self.gap(tensor)
        y = y.view(tensor.shape[0], 1, 1, -1)
        y = self.MLP1(y)
        y = self.MLP2(y)
        y = self.sigmoid(y)
        y = y.view(y.shape[0], -1, 1, 1)
        y = torch.mul(tensor, y)

        return y

        # Maybe batch-norm the two pooling types to make their scales more similar.
        att = self.sigmoid(gap + gmp)
        return tensor * att


class Bilinear(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, tensor):
        return F.interpolate(tensor, scale_factor=2, mode='bilinear', align_corners=False)


class UnetKSA(nn.Module):
    def __init__(self, in_chans, out_chans, ext_chans, chans, num_pool_layers,
                 min_ext_size, max_ext_size, use_ext_bias=True):

        super().__init__()

        self.extractor = AsymmetricSignalExtractor(
            in_chans=in_chans, out_chans=chans, ext_chans=ext_chans,
            min_ext_size=min_ext_size, max_ext_size=max_ext_size, use_bias=use_ext_bias)

        self.cam = ChannelAttention(chans)
        self.pool = nn.AvgPool2d(2)
        self.interp = Bilinear()

        self.down_sample_layers = nn.ModuleList()
        ch = chans

        for n in range(num_pool_layers - 1):
            conv = ConvBlock(in_chans=ch, out_chans=ch * 2)
            self.down_sample_layers.append(conv)
            ch *= 2

        self.conv_mid = ConvBlock(in_chans=ch, out_chans=ch)

        self.up_sample_layers = nn.ModuleList()
        for n in range(num_pool_layers - 1):
            conv = ConvBlock(in_chans=ch * 2, out_chans=ch // 2)
            self.up_sample_layers.append(conv)
            #self.up_sample_layers.append(self.cam)
            ch //= 2
        else:
            conv = ConvBlock(in_chans=ch * 2, out_chans=ch)
            self.up_sample_layers.append(conv)

        self.conv_last = nn.Conv2d(in_channels=ch, out_channels=out_chans, kernel_size=1)

    def forward(self, tensor):
        stack = list()
        output = self.extractor(tensor)
        output = self.cam(output)
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