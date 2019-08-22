"""
UNET model same as that of Facebook's but with processing steps included in the model.
"""

import torch
from torch import nn
from torch.nn import functional as F

# This is a normalized ifft2D with ifftshift already included.
from data.data_transforms import ifft2, nchw_to_kspace, complex_abs


class ConvBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization and relu activation.
    """

    def __init__(self, in_chans, out_chans):
        """
        Args:
            in_chans (int): Number of channels in the input.
            out_chans (int): Number of channels in the output.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_chans),
            nn.ReLU(),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_chans),
            nn.ReLU(),
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
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers

        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans)])
        ch = chans
        for i in range(num_pool_layers - 1):
            self.down_sample_layers += [ConvBlock(ch, ch * 2)]
            ch *= 2
        self.conv = ConvBlock(ch, ch)

        self.up_sample_layers = nn.ModuleList()
        for i in range(num_pool_layers - 1):
            self.up_sample_layers += [ConvBlock(ch * 2, ch // 2)]
            ch //= 2
        self.up_sample_layers += [ConvBlock(ch * 2, ch)]
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch, ch // 2, kernel_size=1),
            nn.Conv2d(ch // 2, out_chans, kernel_size=1),
            nn.Conv2d(out_chans, out_chans, kernel_size=1),
        )

    def forward(self, tensor, out_shape):  # Using out_shape only works for batch size of 1.
        """
        Args:
            tensor (torch.Tensor): Input tensor of shape [batch_size, in_chans, height, width]
            out_shape (tuple): shape [batch_size, num_coils, true_height, true_width].
            Note that in_chans = 2 * num_coils
        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        stack = list()
        output = tensor
        # Apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = F.max_pool2d(output, kernel_size=2)

        output = self.conv(output)

        # Apply up-sampling layers
        for layer in self.up_sample_layers:
            output = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=False)
            output = torch.cat((output, stack.pop()), dim=1)
            output = layer(output)

        output = self.conv2(output)  # End of learning.

        # For removing width dimension padding. Recall that k-space form has 2 as last dim size.
        left = (output.size(-1) - out_shape[-1]) // 2  # This depends on mini-batch size being 1 to work.
        right = left + out_shape[-1]

        # Previously, cropping was done by  [pad:-pad]. However, this fails catastrophically when pad=0 as
        # all values are wiped out in those cases where [0:0] creates an empty tensor.

        # Cropping width dimension by pad.
        output = output[..., left:right]

        # Processing to k-space form.
        output = nchw_to_kspace(output)

        # Convert to image.
        output = complex_abs(ifft2(output))

        assert output.size() == out_shape  # Checking just in case.
        return output
