import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from data.data_transforms import ifft2, kspace_to_nchw, fft1, nchw_to_kspace, complex_abs, ifft1


class IntermediateTransformImageCroppedK(nn.Module):
    """
    Outputs are expected to be k-space data that need reshaping and conversion to the image domain.
    Inputs and targets are expected to be scaled already.
    Currently, the implementation expects only 1 batch.
    Also, I removed rescaling as the loss needs to be calculated on the standardized values for data scale invariance.
    Final reconstructions can be obtained by dividing by the k_scale value since
    the Fourier transform and its relatives are all linear functions.
    """
    def __init__(self, divisor=1):
        super().__init__()
        self.divisor = divisor

    def forward(self, image_outputs, targets, extra_params):

        if image_outputs.size(0) > 1:
            raise NotImplementedError('Only one batch at a time for now.')

        # For removing width dimension padding. Recall that k-space form has 2 as last dim size.
        left = (image_outputs.size(-1) - targets['cmg_targets'].size(-2)) // 2
        right = left + targets['cmg_targets'].size(-2)

        # Cropping width dimension by pad. # Multiply by scales to restore the original scaling. --> Not rescaling!!!
        image_outputs = image_outputs[..., left:right]  # * extra_params['k_scales']

        image_recons = nchw_to_kspace(image_outputs)

        assert image_recons.shape == targets['cmg_targets'].shape, 'Reconstruction and target sizes are different.'

        # image_recons = torch.stack([image_recons[..., 0] * torch.cos(image_recons[..., 1]), image_recons[..., 0] * torch.sin(image_recons[..., 1])], dim=-1)

        masked_semi_kspace = fft1(image_recons, direction='width')
        kspace_recons = fft1(masked_semi_kspace, direction='height')
        masked_semi_kspace = masked_semi_kspace / extra_params['norm_mask']
        output = kspace_to_nchw(masked_semi_kspace)
        masked_semi_kspace = masked_semi_kspace * extra_params['masks']

        margin = output.size(-1) % self.divisor
        if margin > 0:
            pad = [(self.divisor - margin) // 2, (1 + self.divisor - margin) // 2]
        else:  # This is a temporary fix to prevent padding by half the divisor when margin=0.
            pad = [0, 0]

        # This pads at the last dimension of a tensor with 0.
        output = F.pad(output, pad=pad, value=0)

        recons = {'img_recons': complex_abs(image_recons), 'masked_semi_kspace': masked_semi_kspace, 'kspace_recons': kspace_recons}

        return output, recons  # Returning scaled reconstructions. Not rescaled.


class IntermediateTransformE2EKI(nn.Module):
    """
    Outputs are expected to be k-space data that need reshaping and conversion to the image domain.
    Inputs and targets are expected to be scaled already.
    Currently, the implementation expects only 1 batch.
    Also, I removed rescaling as the loss needs to be calculated on the standardized values for data scale invariance.
    Final reconstructions can be obtained by dividing by the k_scale value since
    the Fourier transform and its relatives are all linear functions.
    """
    def __init__(self, divisor=1):
        super().__init__()
        self.divisor = divisor

    def forward(self, kspace_outputs, targets, extra_params):

        if kspace_outputs.size(0) > 1:
            raise NotImplementedError('Only one batch at a time for now.')

        # For removing width dimension padding. Recall that k-space form has 2 as last dim size.
        left = (kspace_outputs.size(-1) - targets['cmg_targets'].size(-2)) // 2
        right = left + targets['cmg_targets'].size(-2)

        # Cropping width dimension by pad. # Multiply by scales to restore the original scaling. --> Not rescaling!!!
        kspace_outputs = kspace_outputs[..., left:right]  # * extra_params['k_scales']

        semi_kspace_recons = nchw_to_kspace(kspace_outputs)

        assert semi_kspace_recons.shape == targets['cmg_targets'].shape, 'Reconstruction and target sizes are different.'

        # kspace_recons = torch.stack([kspace_recons[..., 0] * torch.cos(kspace_recons[..., 1]), kspace_recons[..., 0] * torch.sin(kspace_recons[..., 1])], dim=-1)

        # masked_semi_kspace = kspace_recons
        masked_semi_kspace = semi_kspace_recons * extra_params['masks']
        semi_kspace_recons = semi_kspace_recons * (1 - extra_params['masks']) + targets['masked_semi_kspace']
        semi_kspace_recons = semi_kspace_recons * extra_params['norm_mask']
        cmg_recons = ifft1(semi_kspace_recons, direction='width')
        img_recons = complex_abs(cmg_recons)
        kspace_recons = fft1(semi_kspace_recons, direction='height')
        img_scale = torch.std(img_recons)
        output = img_recons / img_scale

        extra_params['img_scale'] = img_scale

        margin = output.size(-1) % self.divisor
        if margin > 0:
            pad = [(self.divisor - margin) // 2, (1 + self.divisor - margin) // 2]
        else:  # This is a temporary fix to prevent padding by half the divisor when margin=0.
            pad = [0, 0]

        # This pads at the last dimension of a tensor with 0.
        output = F.pad(output, pad=pad, value=0)

        recons = {'img_recons': img_recons, 'kspace_recons': kspace_recons, 'masked_semi_kspace': masked_semi_kspace}

        return output, recons  # Returning scaled reconstructions. Not rescaled.
