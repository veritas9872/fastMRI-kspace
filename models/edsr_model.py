"""
My implementation of EDSR.
There are several differences with the original model.
First, Squeeze Excitation modules are included in all residual layers.
Second, the last convolution has 1x1 convolutions instead of 3x3 convolutions.
"""
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, num_chans, reduction=16):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling.
        self.layer = nn.Sequential(
            nn.Linear(in_features=num_chans, out_features=num_chans // reduction),
            nn.ReLU(),
            nn.Linear(in_features=num_chans // reduction, out_features=num_chans)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, tensor):
        batch, chans, _, _ = tensor.shape
        gap = self.gap(tensor).view(batch, chans)
        features = self.layer(gap)
        att = self.sigmoid(features).view(batch, chans, 1, 1)
        return tensor * att


class ResBlock(nn.Module):
    def __init__(self, num_chans, kernel_size=3, res_scale=1., reduction=16):
        super().__init__()
        assert kernel_size % 2, 'Kernel size is expected to be an odd number.'
        padding = kernel_size // 2
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=num_chans, out_channels=num_chans, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(in_channels=num_chans, out_channels=num_chans, kernel_size=kernel_size, padding=padding),
        )
        self.res_scale = res_scale
        self.ca = ChannelAttention(num_chans=num_chans, reduction=reduction)

    def forward(self, tensor):  # The addition of the residual is also a non-linearity.
        return tensor + self.res_scale * self.ca(self.layer(tensor))


class EDSRModel(nn.Module):
    def __init__(self, in_chans, out_chans, chans, num_depth_blocks, res_scale, reduction, use_residual=False):
        super().__init__()
        self.num_depth_blocks = num_depth_blocks
        self.use_residual = use_residual
        self.head = nn.Conv2d(in_channels=in_chans, out_channels=chans, kernel_size=3, padding=1)

        body = list()
        for _ in range(num_depth_blocks):
            body.append(ResBlock(num_chans=chans, kernel_size=3, res_scale=res_scale, reduction=reduction))
        self.body = nn.Sequential(*body)

        self.tail = nn.Conv2d(in_channels=chans, out_channels=out_chans, kernel_size=1)

    def forward(self, tensor):
        output = self.head(tensor)
        output = output + self.body(output)  # Residual in the entire body as well.
        output = self.tail(output)
        if self.use_residual:
            output = tensor + output
        return output
