import torch
from torch import nn
import torch.nn.functional as F

from data.data_transforms import center_crop, root_sum_of_squares


class PostProcessTestIMG(nn.Module):
    def __init__(self, resolution=320):
        super().__init__()
        self.resolution = resolution

    def forward(self, img_output, extra_params):
        if img_output.size(0) > 1:
            raise NotImplementedError('Only one at a time for now.')

        # Removing values below 0, which are impossible anyway.
        img_recon = F.relu(center_crop(img_output, shape=(self.resolution, self.resolution)))
        img_recon *= extra_params['img_scales']

        if img_recon.size(1) == 15:
            img_recon = root_sum_of_squares(img_recon, dim=1)

        img_recon = img_recon.squeeze()

        return img_recon