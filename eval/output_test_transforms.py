import torch
from torch import nn, Tensor
import torch.nn.functional as F

from data.data_transforms import center_crop, root_sum_of_squares


class PostProcessTestIMG(nn.Module):
    def __init__(self, challenge, resolution=320):
        super().__init__()
        if challenge not in ('singlecoil', 'multicoil'):
            raise ValueError(f'Challenge should either be "singlecoil" or "multicoil"')
        self.challenge = challenge
        self.resolution = resolution

    def forward(self, img_output, targets, extra_params):
        if img_output.size(0) > 1:
            raise NotImplementedError('Only one at a time for now.')

        # Removing values below 0, which are impossible anyway.
        img_recon = F.relu(center_crop(img_output, shape=(self.resolution, self.resolution)))
        img_recon *= extra_params['img_scales']

        if self.challenge == 'multicoil':
            img_recon = root_sum_of_squares(img_recon, dim=1)

        return img_recon.squeeze()


class PostProcessTestRSS(nn.Module):
    """
    Super-hack implementation for post-processing of RSS outputs.
    """
    def __init__(self, challenge: str, residual_rss=True, resolution=320):
        super().__init__()
        assert challenge == 'multicoil', 'Challenge must be multicoil for this.'
        self.challenge = challenge
        self.residual_rss = residual_rss
        self.resolution = resolution  # Useless variable. The input is always the same size as the target.

    def forward(self, rss_output: Tensor, targets: dict, extra_params: dict) -> dict:
        if rss_output.size(0) > 1:
            raise NotImplementedError('Only one at a time for now.')

        rss_recon = rss_output.squeeze()  # Remove single batch and channel dimensions.

        if self.residual_rss:  # Residual RSS image is added.
            rss_recon = (rss_recon + targets['rss_inputs'])

        # Removing impossible negative values.
        rss_recon = F.relu(rss_recon)

        # Rescaling to original scale. Problematic if scale sensitive losses such as L1 or MSE are used.
        rss_recon = rss_recon * extra_params['img_scales']

        return rss_recon
