import torch
from torch import nn
from torch.nn import functional as F
from data.data_transforms import kspace_to_nchw, nchw_to_kspace

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
            nn.ReLU(),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=out_chans),
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


class PartialConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.input_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                    stride, padding, dilation, groups, bias)
        self.mask_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                   stride, padding, dilation, groups, False)

        torch.nn.init.constant_(self.mask_conv.weight, 1.0)

        # mask is not updated
        for param in self.mask_conv.parameters():
            param.requires_grad = False

    def forward(self, input, mask):
        # http://masc.cs.gmu.edu/wiki/partialconv
        # C(X) = W^T * X + b, C(0) = b, D(M) = 1 * M + 0 = sum(M)
        # W^T* (M .* X) / sum(M) + b = [C(M .* X) – C(0)] / D(M) + C(0)
        # import ipdb; ipdb.set_trace()
        output = self.input_conv(input * mask)
        if self.input_conv.bias is not None:
            output_bias = self.input_conv.bias.view(1, -1, 1, 1).expand_as(
                output)
        else:
            output_bias = torch.zeros_like(output)

        with torch.no_grad():
            output_mask = self.mask_conv(mask)

        no_update_holes = output_mask == 0
        mask_sum = output_mask.masked_fill_(no_update_holes, 1.0)

        output_pre = (output - output_bias) / mask_sum + output_bias
        output = output_pre.masked_fill_(no_update_holes, 0.0)

        new_mask = torch.ones_like(output)
        new_mask = new_mask.masked_fill_(no_update_holes, 0.0)

        return output, new_mask


class PartialConvBlock(nn.Module):
    def __init__(self, in_chans, out_chans):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans

        self.conv1 = PartialConv2d(in_chans, out_chans, 3, 1, 1)
        self.activ1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(out_chans)
        self.conv2 = PartialConv2d(out_chans, out_chans, 3, 1, 1)
        self.activ2 = nn.ReLU()
        self.bn2 = nn.GroupNorm(num_groups=8, num_channels=out_chans)

    def forward(self, tensor, mask):
        h, h_mask = self.conv1(tensor, mask)
        h = self.activ1(h)
        h = self.bn1(h)
        h, out_mask = self.conv2(h, h_mask)
        h = self.activ2(h)
        h = self.bn2(h)

        return h, out_mask


class PCUnet(nn.Module):
    def __init__(self, in_chans, out_chans, chans, num_pool_layers):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers

        ch = chans
        self.down_sample_layers = nn.ModuleList([PartialConvBlock(in_chans, chans)])
        for i in range(num_pool_layers - 1):
            self.down_sample_layers += [PartialConvBlock(ch, ch * 2)]
            ch *= 2
        self.conv = PartialConvBlock(ch, ch)

        self.up_sample_layers = nn.ModuleList()
        for i in range(num_pool_layers - 1):
            self.up_sample_layers += [PartialConvBlock(ch * 2, ch // 2)]
            ch //= 2
        self.up_sample_layers += [PartialConvBlock(ch * 2, ch)]
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch, ch // 2, kernel_size=1),
            nn.Conv2d(ch // 2, out_chans, kernel_size=1),
            nn.Conv2d(out_chans, out_chans, kernel_size=1),
        )

    def forward(self, tensor, mask):

        stack = list()
        stack_mask = list()
        output = tensor
        # Apply down-sampling layers
        for layer in self.down_sample_layers:
            output, mask = layer(output, mask)
            stack.append(output)
            stack_mask.append(mask)
            output = F.max_pool2d(output, kernel_size=2)
            mask = F.max_pool2d(mask, kernel_size=2)

        output, mask = self.conv(output, mask)

        # Apply up-sampling layers
        for layer in self.up_sample_layers:
            output = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=False)
            mask = F.interpolate(mask, scale_factor=2, mode='bilinear', align_corners=False)
            output = torch.cat((output, stack.pop()), dim=1)
            mask = torch.cat((mask, stack_mask.pop()), dim=1)
            output, mask = layer(output, mask)

        return self.conv2(output) + tensor
