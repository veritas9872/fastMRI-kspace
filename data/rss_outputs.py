import torch
from torch import nn, Tensor
import torch.nn.functional as F


class PostProcessRSS(nn.Module):
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

        rss_target = targets['rss_targets']
        rss_recon = rss_output.squeeze()  # Remove single batch and channel dimensions.
        assert rss_recon.shape == rss_target.shape

        if self.residual_rss:  # Residual RSS image is added.
            rss_recon = (rss_recon + targets['rss_inputs'])

        # Removing impossible negative values.
        rss_recon = F.relu(rss_recon)

        # Rescaling to original scale. Problematic if scale sensitive losses such as L1 or MSE are used.
        rss_recon = rss_recon * extra_params['img_scales']
        recons = {'rss_recons': rss_recon}

        return recons
