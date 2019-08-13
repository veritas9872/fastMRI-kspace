import torch
from torch import nn, Tensor
import torch.nn.functional as F

import math


class BasicLayer(nn.Module):
    def __init__(self, in_chans: int, out_chans: int, kernel_size: int, stride: int, negative_slope: float):
        super().__init__()
        padding = kernel_size // 2
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=in_chans, out_channels=out_chans,
                      kernel_size=kernel_size, stride=stride, padding=padding),
            nn.LeakyReLU(negative_slope=negative_slope)
        )

    def forward(self, tensor: Tensor) -> Tensor:
        return self.layer(tensor)


class BasicBlock(nn.Module):
    def __init__(self, in_chans: int, out_chans: int, num_groups: int, negative_slope: float):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=in_chans, out_channels=out_chans, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Conv2d(in_channels=out_chans, out_channels=out_chans, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_chans),
            nn.LeakyReLU(negative_slope=negative_slope)
        )

    def forward(self, tensor: Tensor) -> Tensor:
        return self.layer(tensor)


class Interpolate(nn.Module):
    def __init__(self, scale_factor=2, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        if mode in ('linear', 'bilinear', 'trilinear'):
            self.align_corners = False
        elif mode == 'bicubic':
            self.align_corners = True
        else:
            self.align_corners = None

    def forward(self, tensor):
        return F.interpolate(tensor, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)


class XNet(nn.Module):
    def __init__(self, in_chans: int, out_chans: int, chans: int, num_pool_layers: int,
                 num_groups: int, negative_slope: float, residual_magnitude: bool, residual_phase: bool):
        super().__init__()
        self.residual_magnitude = residual_magnitude
        self.residual_phase = residual_phase
        # self.pi = torch.tensor(math.pi, dtype=torch.float32)  # Maybe use later.

        kwargs = dict(num_groups=num_groups, negative_slope=negative_slope)
        self.magnitude_head = nn.Sequential(
            BasicLayer(in_chans=in_chans, out_chans=chans, kernel_size=7, stride=1, negative_slope=negative_slope),
            BasicBlock(in_chans=chans, out_chans=chans, **kwargs)
        )

        self.phase_head = nn.Sequential(
            BasicLayer(in_chans=in_chans, out_chans=chans, kernel_size=7, stride=1, negative_slope=negative_slope),
            BasicBlock(in_chans=chans, out_chans=chans, **kwargs)
        )

        self.neck = BasicBlock(in_chans=chans * 2, out_chans=chans, **kwargs)
        self.interpolate = Interpolate(scale_factor=2, mode='nearest')
        self.down_sample_layers = nn.ModuleList()
        ch = chans

        for _ in range(num_pool_layers - 1):
            conv = nn.Sequential(
                BasicLayer(in_chans=ch, out_chans=ch, kernel_size=3, stride=2, negative_slope=negative_slope),
                BasicBlock(in_chans=ch, out_chans=ch * 2, **kwargs)
            )
            self.down_sample_layers.append(conv)
            ch *= 2

        self.mid_layer = nn.Sequential(
            BasicLayer(in_chans=ch, out_chans=ch, kernel_size=3, stride=2, negative_slope=negative_slope),
            BasicBlock(in_chans=ch, out_chans=ch, **kwargs)
        )

        self.up_sample_layers = nn.ModuleList()

        for _ in range(num_pool_layers - 1):
            conv = BasicBlock(in_chans=ch * 2, out_chans=ch // 2, **kwargs)
            self.up_sample_layers.append(conv)
            ch //= 2
        else:
            conv = BasicBlock(in_chans=ch * 2, out_chans=ch, **kwargs)
            self.up_sample_layers.append(conv)

        assert chans == ch, 'Incorrect channel calculations.'

        self.magnitude_tail = nn.Sequential(
            BasicBlock(in_chans=chans, out_chans=chans, **kwargs),
            BasicLayer(in_chans=chans, out_chans=out_chans, kernel_size=1, stride=1, negative_slope=negative_slope),
            nn.ReLU()  # Because magnitudes must be non-negative.
        )

        self.phase_tail = nn.Sequential(
            BasicBlock(in_chans=chans, out_chans=chans, **kwargs),
            BasicLayer(in_chans=chans, out_chans=out_chans, kernel_size=1, stride=1, negative_slope=negative_slope)
        )

    def forward(self, magnitudes: Tensor, phases: Tensor) -> (Tensor, Tensor):
        stack = list()
        output = torch.cat([self.magnitude_head(magnitudes), self.phase_head(phases)], dim=1)
        output = self.neck(output)
        stack.append(output)

        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)

        output = self.mid_layer(output)

        for layer in self.up_sample_layers:
            output = self.interpolate(output)
            output = torch.cat([output, stack.pop()], dim=1)
            output = layer(output)

        if self.residual_magnitude:
            magnitudes = magnitudes + self.magnitude_tail(output)
        else:
            magnitudes = self.magnitude_tail(output)

        if self.residual_phase:
            phases = phases + self.phase_tail(output)
        else:
            phases = self.phase_tail(output)

        return magnitudes, phases
