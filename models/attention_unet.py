import torch
import torch.nn.functional as F
from torch import nn, cuda
from torch.autograd import Variable


class PartialConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):

        # whether the mask is multi-channel or not
        if 'multi_channel' in kwargs:
            self.multi_channel = kwargs['multi_channel']
            kwargs.pop('multi_channel')
        else:
            self.multi_channel = False

        if 'return_mask' in kwargs:
            self.return_mask = kwargs['return_mask']
            kwargs.pop('return_mask')
        else:
            self.return_mask = False

        super(PartialConv2d, self).__init__(*args, **kwargs)

        if self.multi_channel:
            self.weight_maskUpdater = torch.ones(self.out_channels, self.in_channels, self.kernel_size[0],
                                                 self.kernel_size[1])
        else:
            self.weight_maskUpdater = torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1])

        self.slide_winsize = self.weight_maskUpdater.shape[1] * self.weight_maskUpdater.shape[2] * \
                             self.weight_maskUpdater.shape[3]

        self.last_size = (None, None, None, None)
        self.update_mask = None
        self.mask_ratio = None

    def forward(self, input, mask_in=None):
        assert len(input.shape) == 4
        if mask_in is not None or self.last_size != tuple(input.shape):
            self.last_size = tuple(input.shape)

            with torch.no_grad():
                if self.weight_maskUpdater.type() != input.type():
                    self.weight_maskUpdater = self.weight_maskUpdater.to(input)

                if mask_in is None:
                    # if mask is not provided, create a mask
                    if self.multi_channel:
                        mask = torch.ones(input.data.shape[0], input.data.shape[1], input.data.shape[2],
                                          input.data.shape[3]).to(input)
                    else:
                        mask = torch.ones(1, 1, input.data.shape[2], input.data.shape[3]).to(input)
                else:
                    mask = mask_in

                self.update_mask = F.conv2d(mask, self.weight_maskUpdater, bias=None, stride=self.stride,
                                            padding=self.padding, dilation=self.dilation, groups=1)

                self.mask_ratio = self.slide_winsize / (self.update_mask + 1e-8)
                # self.mask_ratio = torch.max(self.update_mask)/(self.update_mask + 1e-8)
                self.update_mask = torch.clamp(self.update_mask, 0, 1)
                self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)

        # if self.update_mask.type() != input.type() or self.mask_ratio.type() != input.type():
        #     self.update_mask.to(input)
        #     self.mask_ratio.to(input)

        raw_out = super(PartialConv2d, self).forward(torch.mul(input, mask) if mask_in is not None else input)

        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
            output = torch.mul(output, self.update_mask)
        else:
            output = torch.mul(raw_out, self.mask_ratio)

        if self.return_mask:
            return output, self.update_mask
        else:
            return output


class PartialConvBlock(nn.Module):
    def __init__(self, in_chans, out_chans):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans

        self.layers = nn.Sequential(
            PartialConv2d(in_chans, out_chans, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=out_chans),
            PartialConv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_chans),
            nn.ReLU()
        )

    def forward(self, tensor):
        return self.layers(tensor)


class ConvBlock(nn.Module):
    def __init__(self, in_chans, out_chans):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=out_chans),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_chans),
            nn.ReLU()
        )

    def forward(self, tensor):
        return self.layers(tensor)


class ConvWA(nn.Module):
    def __init__(self, in_chans, out_chans):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=out_chans),
            nn.Conv2d(out_chans, out_chans * 2, kernel_size=3, padding=1, bias=True),
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
            nn.BatchNorm2d(num_features=out_chans),
            nn.ReLU(),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=True),
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


class ConvAttentionWA(nn.Module):
    def __init__(self, in_chans, out_chans, reduction=16):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans

        # Conv layers
        self.cbrlayers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(num_features=out_chans),
            nn.ReLU(),
            nn.Conv2d(out_chans, out_chans*2, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(num_features=out_chans*2),
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


class Unet(nn.Module):

    def __init__(self, in_chans, out_chans, chans, num_pool_layers):
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
        self.conv2 = nn.Conv2d(ch, out_chans, kernel_size=1)  # Simplified this part since there were no ReLUs anyway.

    def forward(self, tensor):  # Using out_shape only works for batch size of 1.
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

        return self.conv2(output) + tensor


class UnetA(nn.Module):

    def __init__(self, in_chans, out_chans, chans, num_pool_layers):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers

        self.down_sample_layers = nn.ModuleList([ConvAttention(in_chans, chans)])
        ch = chans
        for i in range(num_pool_layers - 1):
            self.down_sample_layers += [ConvAttention(ch, ch * 2)]
            ch *= 2
        self.conv = ConvAttention(ch, ch)

        self.up_sample_layers = nn.ModuleList()
        for i in range(num_pool_layers - 1):
            self.up_sample_layers += [ConvAttention(ch * 2, ch // 2)]
            ch //= 2
        self.up_sample_layers += [ConvAttention(ch * 2, ch)]
        self.conv2 = nn.Conv2d(ch, out_chans, kernel_size=1)  # Simplified this part since there were no ReLUs anyway.

    def forward(self, tensor):  # Using out_shape only works for batch size of 1.
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

        return self.conv2(output) + tensor


class DAUnet(nn.Module):

    def __init__(self, in_chans, out_chans, chans, num_pool_layers):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers

        self.down_sample_layers = nn.ModuleList([ConvAttention(in_chans, chans)])
        ch = chans
        for i in range(num_pool_layers - 1):
            self.down_sample_layers += [ConvAttention(ch, ch * 2)]
            ch *= 2
        self.conv = ConvAttention(ch, ch)

        self.up_sample_layers = nn.ModuleList()
        for i in range(num_pool_layers - 1):
            self.up_sample_layers += [ConvAttention(ch * 2, ch // 2)]
            ch //= 2
        self.up_sample_layers += [ConvAttention(ch * 2, ch)]
        self.out_conv1 = nn.Conv2d(ch, ch, kernel_size=1)
        self.out_conv2 = nn.Conv2d(ch, out_chans, kernel_size=1)

    def forward(self, tensor):  # Using out_shape only works for batch size of 1.
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

        output = self.out_conv1(output)

        return self.out_conv2(output) + tensor


class DAUnetA(nn.Module):

    def __init__(self, in_chans, out_chans, chans, num_pool_layers):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers

        self.down_sample_layers = nn.ModuleList([ConvAttention(in_chans, chans)])
        ch = chans
        for i in range(num_pool_layers - 1):
            self.down_sample_layers += [ConvAttention(ch, ch * 2)]
            ch *= 2
        self.conv = ConvAttention(ch, ch)

        self.up_sample_layers = nn.ModuleList()
        for i in range(num_pool_layers - 1):
            self.up_sample_layers += [ConvAttention(ch * 2, ch // 2)]
            ch //= 2
        self.up_sample_layers += [ConvAttention(ch * 2, ch)]
        self.out_conv1 = nn.Conv2d(ch, ch, kernel_size=1)
        self.out_conv2 = nn.Conv2d(ch, out_chans, kernel_size=1)

    def forward(self, tensor):  # Using out_shape only works for batch size of 1.
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

        output = self.out_conv1(output)

        return self.out_conv2(output) + tensor


class WAUnet(nn.Module):

    def __init__(self, in_chans, out_chans, chans, num_pool_layers):
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
        self.up_sample_layers += [ConvWA(ch * 2, ch)]
        self.conv2 = nn.Conv2d(ch * 2, out_chans, kernel_size=1)  # Simplified this part since there were no ReLUs anyway.

    def forward(self, tensor):  # Using out_shape only works for batch size of 1.
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

        return self.conv2(output) + tensor


class WAUnetA(nn.Module):

    def __init__(self, in_chans, out_chans, chans, num_pool_layers):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers

        self.down_sample_layers = nn.ModuleList([ConvAttention(in_chans, chans)])
        ch = chans
        for i in range(num_pool_layers - 1):
            self.down_sample_layers += [ConvAttention(ch, ch * 2)]
            ch *= 2
        self.conv = ConvAttention(ch, ch)

        self.up_sample_layers = nn.ModuleList()
        for i in range(num_pool_layers - 1):
            self.up_sample_layers += [ConvAttention(ch * 2, ch // 2)]
            ch //= 2
        self.up_sample_layers += [ConvAttentionWA(ch * 2, ch)]
        self.conv2 = nn.Conv2d(ch * 2, out_chans, kernel_size=1)  # Simplified this part since there were no ReLUs anyway.

    def forward(self, tensor):  # Using out_shape only works for batch size of 1.
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

        return self.conv2(output) + tensor