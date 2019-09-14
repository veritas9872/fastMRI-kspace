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


class AdapterConv(nn.Module):  # Removed channel attention due to redundancy in DenseNet model, unlike ResNet model.
    def __init__(self, in_chans, out_chans, stride=1, reduction=16):
        super().__init__()
        self.ca = ChannelAttention(num_chans=in_chans, reduction=reduction)
        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, stride=stride, padding=1),
            nn.ReLU()
        )

    def forward(self, tensor):
        return self.layers(self.ca(tensor))


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


class ResDenseBlock(nn.Module):
    def __init__(self, in_chans, growth_rate, num_layers, reduction=16):
        super().__init__()
        layers = list()
        for idx in range(num_layers):
            ch = in_chans + idx * growth_rate
            conv = ConcatConv(in_chans=ch, growth_rate=growth_rate)
            layers.append(conv)

        self.layers = nn.Sequential(*layers)
        ch = in_chans + num_layers * growth_rate
        self.ca = ChannelAttention(num_chans=ch, reduction=reduction)
        self.feature_fusion = nn.Conv2d(in_channels=ch, out_channels=in_chans, kernel_size=1)

    def forward(self, tensor):
        return tensor + self.feature_fusion(self.ca(self.layers(tensor)))


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


class ResidualDenseUNet(nn.Module):
    def __init__(self, in_chans, out_chans, chans, num_pool_layers, growth_rate, num_layers_per_block,
                 num_depth_blocks, reduction=16):
        super().__init__()
        self.num_pool_layers = num_pool_layers
        assert isinstance(num_layers_per_block, (list, tuple))
        assert len(num_layers_per_block) == num_pool_layers + 1

        # First block should have no reduction in feature map size.
        conv = nn.Conv2d(in_channels=in_chans, out_channels=chans, kernel_size=3, padding=1)
        self.down_layers = nn.ModuleList([conv])
        num_layers = num_layers_per_block[0]
        block = ResDenseBlock(in_chans=chans, growth_rate=growth_rate, num_layers=num_layers, reduction=reduction)
        self.down_dense_blocks = nn.ModuleList([block])

        for idx in range(num_pool_layers - 1):
            conv = AdapterConv(in_chans=chans, out_chans=chans, stride=2, reduction=reduction)
            num_layers = num_layers_per_block[idx + 1]
            block = ResDenseBlock(in_chans=chans, growth_rate=growth_rate, num_layers=num_layers, reduction=reduction)
            self.down_layers += [conv]
            self.down_dense_blocks += [block]

        # Size reduction happens at the beginning of a block, hence the need for stride here.
        self.mid_conv = AdapterConv(in_chans=chans, out_chans=chans, stride=2, reduction=reduction)
        mid_dense_blocks = list()
        num_layers = num_layers_per_block[-1]
        # No global feature fusion implemented.
        for _ in range(num_depth_blocks):  # Dropout present in middle residual layers.
            block = ResDenseBlock(in_chans=chans, growth_rate=growth_rate, num_layers=num_layers, reduction=reduction)
            mid_dense_blocks.append(block)
        self.mid_dense_blocks = nn.Sequential(*mid_dense_blocks)

        self.up_shuffle_layers = nn.ModuleList()
        self.upscale_layers = nn.ModuleList()
        self.up_dense_blocks = nn.ModuleList()
        for idx in range(num_pool_layers):
            shuffle = ShuffleUp(in_chans=chans, out_chans=chans, scale_factor=2, reduction=reduction)
            conv = AdapterConv(in_chans=chans * 2, out_chans=chans, stride=1, reduction=reduction)
            num_layers = num_layers_per_block[-idx - 2]
            block = ResDenseBlock(in_chans=chans, growth_rate=growth_rate, num_layers=num_layers, reduction=reduction)
            self.up_shuffle_layers += [shuffle]
            self.upscale_layers += [conv]
            self.up_dense_blocks += [block]

        self.final_layers = nn.Conv2d(in_channels=chans, out_channels=out_chans, kernel_size=1)

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
        output = output + self.mid_dense_blocks(output)

        # Up-Sampling.
        for idx in range(self.num_pool_layers):
            output = self.up_shuffle_layers[idx](output)
            output = torch.cat([output, stack.pop()], dim=1)
            output = self.upscale_layers[idx](output)
            output = self.up_dense_blocks[idx](output)

        return self.final_layers(output)
