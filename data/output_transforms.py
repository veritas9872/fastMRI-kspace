import torch
from torch import nn

from data.data_transforms import nchw_to_kspace, ifft2, complex_abs, ifft1


class OutputReplaceTransformK(nn.Module):
    """
    Outputs are expected to be k-space data that need reshaping and conversion to the image domain.
    Inputs and targets are expected to be scaled already.
    Currently, the implementation expects only 1 batch.
    Also, I removed rescaling as the loss needs to be calculated on the standardized values for data scale invariance.
    Final reconstructions can be obtained by dividing by the k_scale value since
    the Fourier transform and its relatives are all linear functions.
    """
    def __init__(self):
        super().__init__()

    def forward(self, kspace_outputs, targets, extra_params):

        if kspace_outputs.size(0) > 1:
            raise NotImplementedError('Only one batch at a time for now.')

        # For removing width dimension padding. Recall that k-space form has 2 as last dim size.
        left = (kspace_outputs.size(-1) - targets['kspace_targets'].size(-2)) // 2
        right = left + targets['kspace_targets'].size(-2)

        # Cropping width dimension by pad. # Multiply by scales to restore the original scaling. --> Not rescaling!!!
        kspace_outputs = kspace_outputs[..., left:right]  # * extra_params['k_scales']

        kspace_recons = nchw_to_kspace(kspace_outputs)

        assert kspace_recons.shape == targets['kspace_targets'].shape, 'Reconstruction and target sizes are different.'

        kspace_recons = kspace_recons * (1 - extra_params['masks']) + targets['kspace_targets'] * extra_params['masks']

        cmg_recons = ifft2(kspace_recons)

        img_recons = complex_abs(cmg_recons)

        recons = {'kspace_recons': kspace_recons, 'cmg_recons': cmg_recons, 'img_recons': img_recons}

        return recons  # Returning scaled reconstructions. Not rescaled.


class PostProcessSemiK(nn.Module):
    def __init__(self, direction='height'):
        """
        Post-Processing function for Semi-k-space learning.

        Args:
            direction: The direction that the input data was transformed.
                The output data will be transformed in the other direction to convert to image data.
        """
        super().__init__()

        if direction == 'height':
            self.recon_direction = 'width'
        elif direction == 'width':
            self.recon_direction = 'height'
        else:
            raise ValueError('`direction` should either be `height` or `width')

        self.direction = direction

    def forward(self, sks_outputs, targets, extra_params):
        if sks_outputs.size(0) > 1:
            raise NotImplementedError('Batch size is expected to be 1 for now.')

        sks_targets = targets['sks_targets']
        mask = extra_params['masks']

        # For removing width dimension padding. Recall that k-space form has 2 as last dim size.
        left = (sks_outputs.size(-1) - sks_targets.size(-2)) // 2
        right = left + sks_targets.size(-2)

        sks_recons = nchw_to_kspace(sks_outputs[..., left:right])

        assert sks_recons.shape == sks_targets.shape, 'Reconstruction and target sizes are different.'

        sks_recons = sks_recons * (1 - mask) + sks_targets * mask
        kspace_recons = ifft1(sks_recons, direction=self.direction)
        cmg_recons = ifft1(sks_recons, direction=self.recon_direction)
        img_recons = complex_abs(cmg_recons)

        # This is inefficient memory-wise but memory is not a serious issue for me right now.
        recons = {'sks_recons': sks_recons, 'kspace_recons': kspace_recons,
                  'cmg_recons': cmg_recons, 'img_recons': img_recons}

        return recons


class WeightedReplacePostProcess(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, kspace_outputs, targets, extra_params):
        if kspace_outputs.size(0) > 1:
            raise NotImplementedError('Only one batch at a time for now.')

        kspace_targets = targets['kspace_targets']
        mask = extra_params['masks']
        weighting = extra_params['weightings']

        # For removing width dimension padding. Recall that k-space form has 2 as last dim size.
        left = (kspace_outputs.size(-1) - kspace_targets.size(-2)) // 2
        right = left + kspace_targets.size(-2)

        # Cropping width dimension by pad.
        kspace_outputs = kspace_outputs[..., left:right]

        kspace_recons = nchw_to_kspace(kspace_outputs)

        assert kspace_recons.shape == kspace_targets.shape, 'Reconstruction and target sizes are different.'
        assert (kspace_recons.size(-3) % 2 == 0) and (kspace_recons.size(-2) % 2 == 0), \
            'Not impossible but not expected to have odd lengthed sides.'

        # Removing weighting.
        kspace_recons = kspace_recons / weighting

        kspace_recons = kspace_recons * (1 - mask) + kspace_targets * mask

        cmg_recons = ifft2(kspace_recons)

        img_recons = complex_abs(cmg_recons)

        recons = {'kspace_recons': kspace_recons, 'cmg_recons': cmg_recons, 'img_recons': img_recons}

        return recons  # Returning scaled reconstructions. Not rescaled.


