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
        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, stride=stride, padding=1, bias=True),
            nn.ReLU()
        )
        self.ca = ChannelAttention(num_chans=out_chans, reduction=reduction)

    def forward(self, tensor):
        return self.ca(self.layers(tensor))


class ResBlock(nn.Module):
    def __init__(self, num_chans, res_scale=1., reduction=16):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=num_chans, out_channels=num_chans, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0),
            nn.Conv2d(in_channels=num_chans, out_channels=num_chans, kernel_size=3, padding=1),
        )
        self.ca = ChannelAttention(num_chans=num_chans, reduction=reduction)
        self.res_scale = res_scale

    def forward(self, tensor):  # The addition of the residual is also a non-linearity.
        return tensor + self.res_scale * self.ca(self.layer(tensor))


class ResGroup(nn.Module):
    def __init__(self, num_res_blocks_per_group, num_chans, res_scale=1., reduction=16):
        super().__init__()
        layers = list()
        for _ in range(num_res_blocks_per_group):
            layers.append(ResBlock(num_chans=num_chans, res_scale=res_scale, reduction=reduction))
        else:
            layers.append(nn.Conv2d(in_channels=num_chans, out_channels=num_chans, kernel_size=3, padding=1))

        self.layers = nn.Sequential(*layers)

    def forward(self, tensor):
        return tensor + self.layers(tensor)


class ShuffleUp(nn.Module):
    def __init__(self, in_chans, out_chans, scale_factor=2, reduction=16):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=in_chans, out_channels=out_chans * scale_factor ** 2, kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor=scale_factor),
            nn.ReLU()
        )
        self.ca = ChannelAttention(num_chans=out_chans, reduction=reduction)

    def forward(self, tensor):
        return self.ca(self.layer(tensor))


class ConcatConv(nn.Module):
    """
    Concatenated convolution layer for Dense block.
    Procedure:
    1. Channel attention applied on the input tensor (if channel attention is used).
    2. Convolution and ReLU applied on input with attention. Has growth_rate output channels.
    3. Concatenate input (without attention) and output to increase total channel number by growth_rate.
    """
    def __init__(self, in_chans, growth_rate, use_ca=True, reduction=16):
        super().__init__()
        self.use_ca = use_ca
        if use_ca:
            self.ca = ChannelAttention(num_chans=in_chans, reduction=reduction)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_chans, out_channels=growth_rate, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, tensor):
        if self.use_ca:
            output = self.conv(self.ca(tensor))
        else:
            output = self.conv(tensor)

        return torch.cat([tensor, output], dim=1)


class DenseBlock(nn.Module):
    def __init__(self, in_chans, out_chans, growth_rate, num_layers, use_ca=True, reduction=16):
        """
        N.B. use_ca only refers to the inner channel attention modules within the dense block, not the external one.
        """
        super().__init__()
        layers = list()
        for idx in range(num_layers):
            conv = ConcatConv(in_chans=in_chans + idx * growth_rate,
                              growth_rate=growth_rate, use_ca=use_ca, reduction=reduction)
            layers.append(conv)

        self.layers = nn.Sequential(*layers)
        self.conv = nn.Conv2d(in_channels=in_chans + num_layers * growth_rate,
                              out_channels=out_chans, kernel_size=3, padding=1)
        self.ca = ChannelAttention(num_chans=out_chans, reduction=reduction)

    def forward(self, tensor):
        return self.ca(self.conv(self.layers(tensor)))


class UNet(nn.Module):
    def __init__(self, in_chans, out_chans, chans, num_pool_layers,
                 num_res_groups, num_res_blocks_per_group, growth_rate, num_dense_layers, use_dense_ca=True,
                 num_res_layers=1, res_scale=0.1, reduction=16, thick_base=False):
        super().__init__()
        self.num_pool_layers = num_pool_layers

        if isinstance(num_res_layers, int):
            num_res_layers = [num_res_layers] * num_pool_layers
        elif isinstance(num_res_layers, (list, tuple)):
            assert len(num_res_layers) == num_pool_layers, 'Invalid number of residual layer numbers.'
        else:
            raise RuntimeError('Invalid type for num_res_layers.')

        # First block should have no reduction in feature map size.

        self.head = DenseBlock(in_chans=in_chans, out_chans=chans, growth_rate=growth_rate,
                               num_layers=num_dense_layers, use_ca=use_dense_ca, reduction=reduction)

        # conv = AdapterConv(in_chans=chans, out_chans=chans, stride=1, reduction=reduction)
        #
        # block = list()
        # for _ in range(num_res_layers[0]):
        #     res = ResBlock(num_chans=chans, res_scale=res_scale, reduction=reduction)
        #     block.append(res)
        # block = nn.Sequential(*block)

        self.down_reshape_layers = nn.ModuleList()
        self.down_res_blocks = nn.ModuleList()

        ch = chans
        for idx in range(num_pool_layers - 1):
            conv = AdapterConv(in_chans=ch, out_chans=ch * 2, stride=2, reduction=reduction)

            block = list()
            for _ in range(num_res_layers[idx + 1]):
                res = ResBlock(num_chans=ch * 2, res_scale=res_scale, reduction=reduction)
                block.append(res)
            block = nn.Sequential(*block)

            self.down_reshape_layers += [conv]
            self.down_res_blocks += [block]
            ch *= 2

        # Size reduction happens at the beginning of a block, hence the need for stride here.
        depth_chans = ch * 2 if thick_base else ch
        self.mid_conv = AdapterConv(in_chans=ch, out_chans=depth_chans, stride=2, reduction=reduction)
        # mid_res_blocks = list()
        # for _ in range(num_depth_blocks):  # Maybe add residual in residual later?
        #     mid_res_blocks.append(ResBlock(num_chans=depth_chans, res_scale=res_scale, reduction=reduction))
        # self.mid_res_blocks = nn.Sequential(*mid_res_blocks)

        mid_res_groups = list()
        for _ in range(num_res_groups):
            res_group = ResGroup(num_res_blocks_per_group=num_res_blocks_per_group,
                                 num_chans=depth_chans, res_scale=res_scale, reduction=reduction)
            mid_res_groups.append(res_group)
        self.mid_res_groups = nn.Sequential(*mid_res_groups)

        self.upscale_layers = nn.ModuleList()
        self.up_reshape_layers = nn.ModuleList()
        for idx in range(num_pool_layers - 1):  # No residual blocks while scaling up.
            if idx == 0:  # Reducing channel numbers if thick base has been used.
                upscale = ShuffleUp(in_chans=depth_chans, out_chans=ch, scale_factor=2)
            else:
                upscale = ShuffleUp(in_chans=ch, out_chans=ch, scale_factor=2)
            conv = AdapterConv(in_chans=ch * 2, out_chans=ch // 2, stride=1, reduction=reduction)
            self.upscale_layers += [upscale]
            self.up_reshape_layers += [conv]
            ch //= 2
        else:  # Last block of up-sampling.
            upscale = ShuffleUp(in_chans=ch, out_chans=ch, scale_factor=2)
            conv = AdapterConv(in_chans=ch * 2, out_chans=ch, stride=1, reduction=reduction)
            self.upscale_layers += [upscale]
            self.up_reshape_layers += [conv]
            assert chans == ch, 'Channel indexing error!'

        self.final_layers = nn.Conv2d(in_channels=ch, out_channels=out_chans, kernel_size=1)

    def forward(self, tensor):
        stack = list()
        output = self.head(tensor)
        stack.append(output)

        # Down-Sampling
        for idx in range(self.num_pool_layers - 1):
            output = self.down_reshape_layers[idx](output)
            output = self.down_res_blocks[idx](output)
            stack.append(output)

        # Middle blocks
        output = self.mid_conv(output)
        output = output + self.mid_res_groups(output)

        # Up-Sampling.
        for idx in range(self.num_pool_layers):
            output = self.upscale_layers[idx](output)
            output = torch.cat([output, stack.pop()], dim=1)
            output = self.up_reshape_layers[idx](output)

        return self.final_layers(output)