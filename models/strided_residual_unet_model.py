"""
Copyright (c) Facebook, Inc. and its affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
from torch import nn
from torch.nn import functional as F
from data.data_transforms import kspace_to_nchw, nchw_to_kspace


class FeatureExtractionLayer(nn.Module):
    def __init__(self, in_chans, out_chans, width=700):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.width = width

        self.layers = nn.Sequential(
            nn.Linear(width*2, width),
            nn.ReLU(),
            nn.Linear(width, width*2),
        )

    def forward(self, tensor):
        tensor = nchw_to_kspace(tensor)
        # data_width = tensor.size(-2)
        output = tensor.contiguous().view(-1, self.width)
        # margin = self.width - data_width
        # output = F.pad(output, pad=[margin, margin], value=0)
        # output = self.layers(output)[..., margin:margin+data_width*2]
        output = output.view(tensor.shape)
        return kspace_to_nchw(output)


class ConvBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, relu activation and dropout.
    """

    def __init__(self, in_chans, out_chans):
        """
        Args:
            in_chans (int): Number of channels in the input.
            out_chans (int): Number of channels in the output.
            drop_prob (float): Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=out_chans),
            # nn.BatchNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.1),
        )

    def forward(self, tensor):
        """
        Args:
            tensor (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]
        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        return self.layers(tensor)

    def __repr__(self):
        return f'ConvBlock(in_chans={self.in_chans}, out_chans={self.out_chans})'


class StridedConvBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, relu activation and dropout.
    """

    def __init__(self, in_chans, out_chans):
        """
        Args:
            in_chans (int): Number of channels in the input.
            out_chans (int): Number of channels in the output.
            drop_prob (float): Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, stride=2),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=out_chans),
            # nn.BatchNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.1),
        )

    def forward(self, tensor):
        """
        Args:
            tensor (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]
        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        return self.layers(tensor)

    def __repr__(self):
        return f'ConvBlock(in_chans={self.in_chans}, out_chans={self.out_chans})'


class UnetModel(nn.Module):
    """
    PyTorch implementation of a U-Net model.
    This is based on:
        Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks
        for biomedical image segmentation. In International Conference on Medical image
        computing and computer-assisted intervention, pages 234–241. Springer, 2015.
    """

    def __init__(self, in_chans, out_chans, chans, num_pool_layers):
        """
        Args:
            in_chans (int): Number of channels in the input to the U-Net model.
            out_chans (int): Number of channels in the output to the U-Net model.
            chans (int): Number of output channels of the first convolution layer.
            num_pool_layers (int): Number of down-sampling and up-sampling layers.
            drop_prob (float): Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers

        # self.extractor = FeatureExtractionLayer(in_chans, out_chans, width=320)

        # self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans)])
        self.down_sample_layers = nn.ModuleList([])
        self.in_conv = ConvBlock(in_chans, chans)
        ch = chans
        for i in range(num_pool_layers - 1):
            self.down_sample_layers += [StridedConvBlock(ch, ch * 2)]
            ch *= 2
        self.conv = StridedConvBlock(ch, ch)

        self.up_sample_layers = nn.ModuleList()
        for i in range(num_pool_layers - 1):
            self.up_sample_layers += [ConvBlock(ch * 2, ch // 2)]
            ch //= 2
        self.up_sample_layers += [ConvBlock(ch * 2, ch)]
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(ch, out_chans, kernel_size=1),
        )

    def forward(self, tensor):
        """
        Args:
            tensor (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]
        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        stack = list()
        # output = self.extractor(tensor)
        # output = tensor
        # Apply down-sampling layers
        output = self.in_conv(tensor)
        for layer in self.down_sample_layers:
            stack.append(output)
            output = layer(output)

            # output = F.avg_pool2d(output, kernel_size=3, stride=2, padding=1)
        stack.append(output)

        output = self.conv(output)

        # Apply up-sampling layers
        for layer in self.up_sample_layers:
            output = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=False)
            output = torch.cat((output, stack.pop()), dim=1)
            output = layer(output)
        return self.conv2(output) + tensor
