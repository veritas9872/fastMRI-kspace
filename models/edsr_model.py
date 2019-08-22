import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, num_chans, kernel_size=3, res_scale=1.):
        super().__init__()
        assert kernel_size % 2, 'Kernel size is expected to be an odd number.'
        padding = kernel_size // 2
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=num_chans, out_channels=num_chans, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(in_channels=num_chans, out_channels=num_chans, kernel_size=kernel_size, padding=padding),
        )
        self.res_scale = res_scale

    def forward(self, tensor):  # The addition of the residual is also a non-linearity.
        return tensor + self.res_scale * self.layer(tensor)


class ResBlockDSC(nn.Module):
    def __init__(self,  num_chans, kernel_size=3, res_scale=1.):
        super().__init__()
        assert kernel_size % 2, 'Kernel size is expected to be an odd number.'
        padding = kernel_size // 2
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=num_chans, out_channels=num_chans,
                      kernel_size=kernel_size, padding=padding, groups=num_chans),
            nn.Conv2d(in_channels=num_chans, out_channels=num_chans, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=num_chans, out_channels=num_chans,
                      kernel_size=kernel_size, padding=padding, groups=num_chans),
            nn.Conv2d(in_channels=num_chans, out_channels=num_chans, kernel_size=1)
        )
        self.res_scale = res_scale

    def forward(self, tensor):  # The addition of the residual is also a non-linearity.
        return tensor + self.res_scale * self.layer(tensor)


class EDSR(nn.Module):
    def __init__(self, in_chans, out_chans, num_res_blocks, chans, res_scale, use_dsc=False):
        super().__init__()
        self.num_res_blocks = num_res_blocks
        self.head = nn.Conv2d(in_channels=in_chans, out_channels=chans, kernel_size=3, padding=1)

        body = list()
        for _ in range(num_res_blocks):
            if use_dsc:
                body.append(ResBlockDSC(num_chans=chans, kernel_size=3, res_scale=res_scale))
            else:
                body.append(ResBlock(num_chans=chans, kernel_size=3, res_scale=res_scale))
        self.body = nn.Sequential(*body)

        self.tail = nn.Conv2d(in_channels=chans, out_channels=out_chans, kernel_size=1)

    def forward(self, tensor):
        output = self.head(tensor)
        output = output + self.body(output)  # Residual in the entire body as well.
        return self.tail(output)

