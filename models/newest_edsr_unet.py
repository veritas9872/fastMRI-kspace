import torch
from torch import nn
import torch.nn.functional as F


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


class AdapterConv(nn.Module):
    def __init__(self, in_chans, out_chans, stride=1, reduction=16):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, stride=stride, padding=1),
            nn.ReLU()
        )
        self.ca = ChannelAttention(num_chans=out_chans, reduction=reduction)

    def forward(self, tensor):
        return self.ca(self.layers(tensor))


class ResBlock(nn.Module):
    def __init__(self, num_chans, kernel_size=3, res_scale=1., reduction=16):
        super().__init__()
        assert kernel_size % 2, 'Kernel size is expected to be an odd number.'
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=num_chans, out_channels=num_chans, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Conv2d(in_channels=num_chans, out_channels=num_chans, kernel_size=kernel_size, padding=kernel_size // 2),
        )
        self.ca = ChannelAttention(num_chans=num_chans, reduction=reduction)
        self.res_scale = res_scale

    def forward(self, tensor):  # The addition of the residual is also a non-linearity.
        return tensor + self.res_scale * self.ca(self.layer(tensor))


class ResizeConv(nn.Module):
    def __init__(self, in_chans, out_chans, scale_factor=2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=in_chans, out_channels=out_chans, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.scale_factor = scale_factor

    def forward(self, tensor):
        output = F.interpolate(tensor, scale_factor=self.scale_factor, mode='nearest')
        return self.layers(output)


class UNet(nn.Module):
    def __init__(self, in_chans, out_chans, chans, num_pool_layers, num_depth_blocks,
                 res_scale=1., use_residual=False, reduction=16):
        super().__init__()
        self.num_pool_layers = num_pool_layers
        self.num_depth_blocks = num_depth_blocks  # This must be a positive integer.
        self.use_residual = use_residual

        # First block should have no reduction in feature map size.
        conv = AdapterConv(in_chans=in_chans, out_chans=chans, stride=1, reduction=reduction)
        self.down_layers = nn.ModuleList([conv])

        ch = chans
        for _ in range(num_pool_layers - 1):
            conv = AdapterConv(in_chans=ch, out_chans=ch * 2, stride=2, reduction=reduction)
            self.down_layers.append(conv)
            ch *= 2

        # Size reduction happens at the beginning of a block, hence the need for stride here.
        self.mid_conv = AdapterConv(in_chans=ch, out_chans=ch, stride=2, reduction=reduction)
        mid_res_blocks = list()
        for _ in range(num_depth_blocks):  # Dropout present in middle residual layers.
            mid_res_blocks.append(ResBlock(num_chans=ch, res_scale=res_scale, reduction=reduction))
        self.mid_res_blocks = nn.Sequential(*mid_res_blocks)

        self.upscale_layers = nn.ModuleList()
        self.up_reshape_layers = nn.ModuleList()
        self.up_res_blocks = nn.ModuleList()
        for _ in range(num_pool_layers - 1):
            deconv = ResizeConv(in_chans=ch, out_chans=ch, scale_factor=2)
            conv = AdapterConv(in_chans=ch * 2, out_chans=ch // 2, stride=1, reduction=reduction)
            res = ResBlock(num_chans=ch // 2, res_scale=res_scale, reduction=reduction)
            self.upscale_layers.append(deconv)
            self.up_reshape_layers.append(conv)
            self.up_res_blocks.append(res)
            ch //= 2
        else:  # Last block of up-sampling.
            deconv = ResizeConv(in_chans=ch, out_chans=ch,  scale_factor=2)
            conv = AdapterConv(in_chans=ch * 2, out_chans=ch, stride=1, reduction=reduction)
            res = ResBlock(num_chans=ch, res_scale=res_scale, reduction=reduction)
            self.upscale_layers.append(deconv)
            self.up_reshape_layers.append(conv)
            self.up_res_blocks.append(res)
            assert chans == ch, 'Channel indexing error!'

        self.final_layers = nn.Conv2d(in_channels=ch, out_channels=out_chans, kernel_size=1)

        assert len(self.down_layers) == len(self.upscale_layers) == len(self.up_reshape_layers)\
            == len(self.up_res_blocks) == self.num_pool_layers, 'Layer number error!'

    def forward(self, tensor):
        stack = list()
        output = tensor

        # Down-Sampling
        for layer in self.down_layers:
            output = layer(output)
            stack.append(output)

        # Middle blocks
        output = self.mid_conv(output)
        output = output + self.mid_res_blocks(output)

        # Up-Sampling.
        for idx in range(self.num_pool_layers):
            output = self.upscale_layers[idx](output)
            output = torch.cat([output, stack.pop()], dim=1)
            output = self.up_reshape_layers[idx](output)
            output = self.up_res_blocks[idx](output)

        output = self.final_layers(output)
        return (tensor + output) if self.use_residual else output
