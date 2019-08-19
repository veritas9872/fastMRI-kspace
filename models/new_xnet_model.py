import torch
from torch import nn, Tensor
import torch.nn.functional as F


class BasicLayer(nn.Module):
    def __init__(self, in_chans: int, out_chans: int, stride=1, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=in_chans, out_channels=out_chans,
                      kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU()
        )

    def forward(self, tensor: Tensor) -> Tensor:
        return self.layer(tensor)


class ResBlock(nn.Module):
    def __init__(self, num_chans: int, dilation=1, res_scale=1.):
        super().__init__()
        self.res_scale = res_scale
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=num_chans, out_channels=num_chans,
                      kernel_size=3, stride=1, padding=dilation, dilation=dilation),
            nn.ReLU(),
            nn.Conv2d(in_channels=num_chans, out_channels=num_chans,
                      kernel_size=3, stride=1, padding=dilation, dilation=dilation)
        )

    def forward(self, tensor: Tensor) -> Tensor:
        return tensor + self.res_scale * self.layer(tensor)


class ResizeConv(nn.Module):
    def __init__(self, in_chans, out_chans, scale_factor=2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=in_chans, out_channels=out_chans, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.scale_factor = scale_factor

    def forward(self, tensor: Tensor) -> Tensor:
        output = F.interpolate(tensor, scale_factor=self.scale_factor, mode='nearest')
        return self.layers(output)


class XNet(nn.Module):
    def __init__(self, in_chans: int, out_chans: int, chans: int, num_pool_layers: int,
                 num_depth_blocks: int, dilation: int, res_scale: float):
        super().__init__()

        # The dilation parameter is not used up to the neck of the network.
        # This has not been tested to have better performance but is just a hunch.
        # Also, GroupNorm is added to make the different domains have a similar scale. This may or may not be necessary.
        # Group number was chosen without serious testing too.

        self.num_pool_layers = num_pool_layers
        self.magnitude_head = nn.Sequential(
            BasicLayer(in_chans=in_chans, out_chans=chans, kernel_size=3, stride=1),
            ResBlock(num_chans=chans, dilation=1, res_scale=res_scale),
            # nn.GroupNorm(num_groups=16, num_channels=chans)
        )
        self.phase_head = nn.Sequential(
            BasicLayer(in_chans=in_chans, out_chans=chans, kernel_size=3, stride=1),
            ResBlock(num_chans=chans, dilation=1, res_scale=res_scale),
            # nn.GroupNorm(num_groups=16, num_channels=chans)
        )
        self.neck = nn.Sequential(
            BasicLayer(in_chans=chans * 2, out_chans=chans, kernel_size=3, stride=1),
            ResBlock(num_chans=chans, dilation=1, res_scale=res_scale)
        )

        ch = chans
        self.down_reshape_layers = nn.ModuleList()
        self.down_res_blocks = nn.ModuleList()
        for _ in range(num_pool_layers - 1):
            conv = BasicLayer(in_chans=ch, out_chans=ch * 2, kernel_size=3, stride=2)
            res = ResBlock(num_chans=ch * 2, dilation=dilation, res_scale=res_scale)
            self.down_reshape_layers.append(conv)
            self.down_res_blocks.append(res)
            ch *= 2

        self.mid_layer = BasicLayer(in_chans=ch, out_chans=ch, kernel_size=3, stride=2)

        mid_res_blocks = list()
        for _ in range(num_depth_blocks):
            res = ResBlock(num_chans=ch, dilation=dilation)
            mid_res_blocks.append(res)
        self.mid_res_blocks = nn.Sequential(*mid_res_blocks)

        self.upscale_layers = nn.ModuleList()
        self.up_reshape_layers = nn.ModuleList()
        self.up_res_blocks = nn.ModuleList()
        for _ in range(num_pool_layers - 1):
            deconv = ResizeConv(in_chans=ch, out_chans=ch, scale_factor=2)
            conv = BasicLayer(in_chans=ch * 2, out_chans=ch // 2, kernel_size=3, stride=1)
            res = ResBlock(num_chans=ch // 2, dilation=dilation, res_scale=res_scale)
            self.upscale_layers.append(deconv)
            self.up_reshape_layers.append(conv)
            self.up_res_blocks.append(res)
            ch //= 2
        else:
            deconv = ResizeConv(in_chans=ch, out_chans=ch, scale_factor=2)
            conv = BasicLayer(in_chans=ch * 2, out_chans=ch, kernel_size=3, stride=1)
            res = ResBlock(num_chans=ch, dilation=dilation, res_scale=res_scale)
            self.upscale_layers.append(deconv)
            self.up_reshape_layers.append(conv)
            self.up_res_blocks.append(res)
            assert chans == ch, 'Incorrect channel calculations.'

        self.magnitude_tail = nn.Sequential(
            nn.Conv2d(in_channels=chans, out_channels=chans, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=chans, out_channels=chans, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=chans, out_channels=out_chans, kernel_size=1),
        )

        self.phase_tail = nn.Sequential(
            nn.Conv2d(in_channels=chans, out_channels=chans, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=chans, out_channels=chans, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=chans, out_channels=out_chans, kernel_size=1),
        )
        assert len(self.down_reshape_layers) == len(self.down_res_blocks) == num_pool_layers - 1
        assert len(self.upscale_layers) == len(self.up_reshape_layers) == len(self.up_res_blocks) == num_pool_layers

    def forward(self, magnitudes: Tensor, phases: Tensor) -> (Tensor, Tensor):
        stack = list()
        output = torch.cat([self.magnitude_head(magnitudes), self.phase_head(phases)], dim=1)
        output = self.neck(output)
        stack.append(output)

        # Down-Sampling.  # The two heads and neck function as the first block.
        for idx in range(self.num_pool_layers - 1):
            output = self.down_reshape_layers[idx](output)
            output = self.down_res_blocks[idx](output)
            stack.append(output)

        # Middle blocks
        output = self.mid_layer(output)
        output = output + self.mid_res_blocks(output)  # Residual of middle portion.

        for idx in range(self.num_pool_layers):
            output = self.upscale_layers[idx](output)
            output = torch.cat([output, stack.pop()], dim=1)
            output = self.up_reshape_layers[idx](output)
            output = self.up_res_blocks[idx](output)

        # Always uses residual outputs.
        magnitudes = magnitudes + self.magnitude_tail(output)
        phases = phases + self.phase_tail(output)
        return magnitudes, phases
