import torch
from torch import nn

from data.data_transforms import nchw_to_kspace, ifft2, complex_abs


class OutputReplaceTransformK(nn.Module):
    """
    Outputs are expected to be k-space data that need reshaping and conversion to the image domain.
    """
    def __init__(self):
        super().__init__()

    def forward(self, kspace_outputs, targets, extra_params):

        if kspace_outputs.size(0) > 1:
            raise NotImplementedError('Only one batch at a time for now.')

        # For removing width dimension padding. Recall that k-space form has 2 as last dim size.
        left = (kspace_outputs.size(-1) - targets['kspace_targets'].size(-2)) // 2
        right = left + targets['kspace_targets'].size(-2)

        # Cropping width dimension by pad. Multiply by scales to restore the original scaling.
        kspace_outputs = kspace_outputs[..., left:right] * extra_params['k_scales']

        # Processing to k-space form. This is where the batch_size == 1 is important.
        kspace_recons = nchw_to_kspace(kspace_outputs)

        assert kspace_recons.shape == targets['kspace_targets'].shape, 'Reconstruction and target sizes are different.'

        kspace_recons = kspace_recons * (1 - extra_params['masks']) + targets['kspace_targets'] * extra_params['masks']

        cmg_recons = ifft2(kspace_recons)

        img_recons = complex_abs(cmg_recons)

        recons = {'kspace_recons': kspace_recons, 'cmg_recons': cmg_recons, 'img_recons': img_recons}

        return recons
