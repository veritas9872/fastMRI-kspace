import torch
from torch import nn
import torch.nn.functional as F

import math

from data.data_transforms import center_crop, root_sum_of_squares, fft2


class PostProcessXNet(nn.Module):
    def __init__(self, challenge, resolution=320):
        super().__init__()
        if challenge not in ('singlecoil', 'multicoil'):
            raise ValueError(f'Challenge should either be "singlecoil" or "multicoil"')
        self.challenge = challenge
        self.resolution = resolution

    def forward(self, outputs, targets, extra_params):
        img_recon, phase_recon = outputs

        if img_recon.size(0) > 1:
            raise NotImplementedError('Only one at a time for now.')

        img_target = targets['img_targets']
        assert img_recon.shape == img_target.shape, 'Reconstruction and target sizes are different.'

        # Input transform had addition of pi as pre-processing.
        phase_recon = phase_recon - math.pi  # No clamping implemented since the loss is MSE.
        cmg_recon = torch.stack([img_recon * torch.cos(phase_recon), img_recon * torch.sin(phase_recon)], dim=-1)
        kspace_recon = fft2(cmg_recon)

        recons = {'img_recons': img_recon, 'phase_recons': phase_recon,
                  'cmg_recons': cmg_recon, 'kspace_recons': kspace_recon}

        if self.challenge == 'multicoil':
            rss_recon = center_crop(img_recon, (self.resolution, self.resolution)) * extra_params['img_scales']
            rss_recon = root_sum_of_squares(rss_recon, dim=1).squeeze()
            recons['rss_recons'] = rss_recon

        return recons  # recons are not rescaled except rss_recons.

