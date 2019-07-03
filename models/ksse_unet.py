import torch
from torch import nn
import torch.nn.functional as F


class KSSELayer(nn.Module):
    def __init__(self, in_chans, out_chans, ext_chans, min_ext_size=1, max_ext_size=17, use_bias=True):
        super().__init__()
        self.extractors = nn.ModuleList()
        for dilation in range(min_ext_size // 2, (max_ext_size+1) // 2):
            conv = nn.Conv2d(in_channels=in_chans, out_channels=ext_chans, kernel_size=3,
                             stride=1, dilation=(dilation+1) // 2, padding=dilation // 2, bias=use_bias)
            self.extractors.append(conv)

        self.relu = nn.ReLU()
        self.conv1x1 = nn.Conv2d(in_channels=len(self.extractors) * ext_chans,
                                 out_channels=out_chans, kernel_size=1, stride=1, bias=True)

    def forward(self, tensor):
        output = torch.cat([ext(tensor) for ext in self.extractors], dim=1)
        output = self.relu(output)
        return self.conv1x1(output)


class ConvLayer(nn.Module):
    def __init__(self, in_chans, out_chans, stride=1):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=in_chans, out_channels=out_chans, kernel_size=3,
                      stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_chans),
            nn.ReLU()
        )

    def forward(self, tensor):
        return self.layer(tensor)


class ConvBlock(nn.Module):
    def __init__(self, in_chans, out_chans, stride=1):
        super().__init__()
        self.layer = nn.Sequential(
            ConvLayer(in_chans=in_chans, out_chans=out_chans, stride=stride),
            ConvLayer(in_chans=out_chans, out_chans=out_chans, stride=1)
        )

    def forward(self, tensor):
        return self.layer(tensor)


class Interpolate(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=None):
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, tensor):
        return F.interpolate(tensor, self.size, self.scale_factor, self.mode, self.align_corners)


class UnetKSSE(nn.Module):
    def __init__(self, in_chans, out_chans, chans, num_pool_layers,
                 min_ext_size, max_ext_size, use_ext_bias=True, use_res_block=True):
        super().__init__()

        self.use_res_block = use_res_block

        ch = chans

        self.ksse = KSSELayer(in_chans=in_chans, out_chans=ch, ext_chans=ch,
                              min_ext_size=min_ext_size, max_ext_size=max_ext_size, use_bias=use_ext_bias)

        self.down_sample_layers = nn.ModuleList()

        for _ in range(num_pool_layers - 1):  # Down-sampling by strided convolution.
            self.down_sample_layers += [ConvBlock(in_chans=ch, out_chans=ch * 2, stride=2)]
            ch *= 2

        self.conv_mid = ConvBlock(in_chans=ch, out_chans=ch, stride=2)

        self.interp = Interpolate(scale_factor=2, mode='bilinear', align_corners=False)

        self.up_sample_layers = nn.ModuleList()
        for _ in range(num_pool_layers - 1):
            self.up_sample_layers += [ConvBlock(in_chans=ch * 2, out_chans=ch // 2, stride=1)]
            ch //= 2

        self.up_sample_layers += [ConvBlock(in_chans=ch * 2, out_chans=ch, stride=1)]

        self.conv_last = nn.Conv2d(in_channels=ch, out_channels=out_chans, kernel_size=1)

    def forward(self, tensor):
        stack = list()

        output = self.ksse(tensor)
        stack.append(output)

        for layer in self.down_sample_layers:
            output = output + layer(output) if self.use_res_block else layer(output)
            stack.append(output)

        output = output + self.conv_mid(output) if self.use_res_block else self.conv_mid(output)

        for layer in self.up_sample_layers:
            output = self.interp(output)
            output = torch.cat([output, stack.pop()], dim=1)
            output = output + layer(output) if self.use_res_block else layer(output)

        output = output + self.conv_last(output) if self.use_res_block else self.conv_last(output)

        return output
