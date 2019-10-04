import torch
from torch import nn


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
        self.ca = ChannelAttention(num_chans=out_chans, reduction=reduction)
        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, stride=stride, padding=1),
            nn.ReLU()
        )

    def forward(self, tensor):
        return self.layers(self.ca(tensor))


class ResBlock(nn.Module):
    def __init__(self, num_chans, res_scale=1., reduction=16):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=num_chans, out_channels=num_chans, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=num_chans, out_channels=num_chans, kernel_size=3, padding=1),
        )
        self.ca = ChannelAttention(num_chans=num_chans, reduction=reduction)
        self.res_scale = res_scale

    def forward(self, tensor):  # The addition of the residual is also a non-linearity.
        # Should the channel attention be inside the layer or outside??
        return tensor + self.res_scale * self.ca(self.layer(tensor))


class ShuffleUp(nn.Module):
    def __init__(self, in_chans, out_chans, scale_factor=2, reduction=16):
        super().__init__()
        self.ca = ChannelAttention(num_chans=in_chans, reduction=reduction)
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=in_chans, out_channels=out_chans * scale_factor ** 2, kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor=scale_factor),
            nn.ReLU()
        )

    def forward(self, tensor):
        return self.layer(self.ca(tensor))


class ConcatConv(nn.Module):
    """
    Concatenated convolution layer for Dense block.
    Procedure:
    1. Channel attention applied on the input tensor.
    2. Convolution and ReLU applied on input with attention. Has growth_rate output channels.
    3. Concatenate input (without attention) and output to increase total channel number by growth_rate.
    """
    def __init__(self, in_chans, growth_rate, reduction=16):
        super().__init__()
        self.ca = ChannelAttention(num_chans=in_chans, reduction=reduction)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_chans, out_channels=growth_rate, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, tensor):
        return torch.cat([tensor, self.conv(self.ca(tensor))], dim=1)


class DenseBlock(nn.Module):
    def __init__(self, in_chans, out_chans, growth_rate, num_layers, reduction=16):
        super().__init__()
        layers = list()
        for idx in range(num_layers):
            ch = in_chans + idx * growth_rate
            conv = ConcatConv(in_chans=ch, growth_rate=growth_rate)
            layers.append(conv)

        self.layers = nn.Sequential(*layers)
        ch = in_chans + num_layers * growth_rate
        self.ca = ChannelAttention(num_chans=ch, reduction=reduction)
        self.feature_fusion = nn.Conv2d(in_channels=ch, out_channels=out_chans, kernel_size=1)

    def forward(self, tensor):
        return self.feature_fusion(self.ca(self.layers(tensor)))


class UNet(nn.Module):
    def __init__(self, in_chans, out_chans, chans, num_pool_layers,
                 num_depth_blocks, growth_rate, num_layers, res_scale=0.1, reduction=16):

        super().__init__()
        self.num_pool_layers = num_pool_layers

        if isinstance(num_layers, int):
            num_layers = [num_layers] * num_pool_layers
        elif isinstance(num_layers, (list, tuple)):
            assert len(num_layers) == num_pool_layers
        else:
            raise RuntimeError('Invalid type for num_layers.')

        # First block should have no reduction in feature map size.
        # I am wondering whether the input dense-net should be allowed to have the raw inputs or not.
        # Removing ReLU from the first layer because that is what everyone else is doing.
        conv = nn.Conv2d(in_channels=in_chans, out_channels=chans, kernel_size=3, padding=1)
        block = DenseBlock(in_chans=chans, out_chans=chans, growth_rate=growth_rate,
                           num_layers=num_layers[0], reduction=reduction)
        self.down_layers = nn.ModuleList([conv])
        self.down_dense_blocks = nn.ModuleList([block])

        ch = chans
        for idx in range(num_pool_layers - 1):
            conv = AdapterConv(in_chans=ch, out_chans=ch, stride=2, reduction=reduction)
            block = DenseBlock(in_chans=ch, out_chans=ch * 2, growth_rate=growth_rate,
                               num_layers=num_layers[idx + 1], reduction=reduction)
            self.down_layers += [conv]
            self.down_dense_blocks += [block]
            ch *= 2

        # Size reduction happens at the beginning of a block, hence the need for stride here.
        self.mid_conv = AdapterConv(in_chans=ch, out_chans=ch, stride=2, reduction=reduction)
        mid_res_blocks = list()
        for _ in range(num_depth_blocks):
            mid_res_blocks.append(ResBlock(num_chans=ch, res_scale=res_scale, reduction=reduction))
        self.mid_res_blocks = nn.Sequential(*mid_res_blocks)

        self.upscale_layers = nn.ModuleList()
        self.up_dense_blocks = nn.ModuleList()

        for idx in range(num_pool_layers - 1):
            shuffle = ShuffleUp(in_chans=ch, out_chans=ch, scale_factor=2)
            block = DenseBlock(in_chans=ch * 2, out_chans=ch // 2, growth_rate=growth_rate,
                               num_layers=num_layers[-idx-1], reduction=reduction)
            self.upscale_layers += [shuffle]
            self.up_dense_blocks += [block]
            ch //= 2
        else:  # Last block of up-sampling.
            shuffle = ShuffleUp(in_chans=ch, out_chans=ch, scale_factor=2)
            block = DenseBlock(in_chans=ch * 2, out_chans=ch, growth_rate=growth_rate,
                               num_layers=num_layers[-num_pool_layers], reduction=reduction)
            self.upscale_layers += [shuffle]
            self.up_dense_blocks += [block]
            assert chans == ch, 'Channel indexing error!'

        self.final_layers = nn.Conv2d(in_channels=ch, out_channels=out_chans, kernel_size=1)

        assert len(self.down_layers) == len(self.down_dense_blocks) == len(self.upscale_layers) \
            == len(self.up_dense_blocks) == self.num_pool_layers, 'Layer number error!'

    def forward(self, tensor):
        stack = list()
        output = tensor

        # Down-Sampling
        for idx in range(self.num_pool_layers):
            output = self.down_layers[idx](output)
            output = self.down_dense_blocks[idx](output)
            stack.append(output)

        # Middle blocks
        output = self.mid_conv(output)
        output = output + self.mid_res_blocks(output)

        # Up-Sampling.
        for idx in range(self.num_pool_layers):
            output = self.upscale_layers[idx](output)
            output = torch.cat([output, stack.pop()], dim=1)
            output = self.up_dense_blocks[idx](output)

        return self.final_layers(output)
