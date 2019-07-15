import torch
from torch import nn

from data.data_transforms import nchw_to_kspace, ifft2, complex_abs, fft1, ifft1, exp_weighting


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


class OutputReplaceTransformKK(nn.Module):
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

    def forward(self, kspace_outputs, feature_kspace, targets, extra_params):

        if kspace_outputs.size(0) > 1:
            raise NotImplementedError('Only one batch at a time for now.')

        # For removing width dimension padding. Recall that k-space form has 2 as last dim size.
        left = (kspace_outputs.size(-1) - targets['kspace_targets'].size(-2)) // 2
        right = left + targets['kspace_targets'].size(-2)

        # Cropping width dimension by pad. # Multiply by scales to restore the original scaling. --> Not rescaling!!!
        kspace_outputs = kspace_outputs[..., left:right]  # * extra_params['k_scales']

        kspace_recons = nchw_to_kspace(kspace_outputs)

        if feature_kspace.size(0) > 1:
            raise NotImplementedError('Only one batch at a time for now.')

        # For removing width dimension padding. Recall that k-space form has 2 as last dim size.
        left = (feature_kspace.size(-1) - targets['kspace_targets'].size(-2)) // 2
        right = left + targets['kspace_targets'].size(-2)

        # Cropping width dimension by pad. # Multiply by scales to restore the original scaling. --> Not rescaling!!!
        feature_kspace = feature_kspace[..., left:right]  # * extra_params['k_scales']

        feature_kspace = nchw_to_kspace(feature_kspace)

        assert kspace_recons.shape == targets['kspace_targets'].shape, 'Reconstruction and target sizes are different.'

        kspace_recons = kspace_recons * (1 - extra_params['masks']) + targets['kspace_targets'] * extra_params['masks']

        cmg_recons = ifft2(kspace_recons)

        img_recons = complex_abs(cmg_recons)

        recons = {'kspace_recons': kspace_recons, 'cmg_recons': cmg_recons, 'img_recons': img_recons, 'feature_kspace':feature_kspace}

        return recons  # Returning scaled reconstructions. Not rescaled.


class OutputReplaceTransformSemiK(nn.Module):
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

        semi_kspace_recons = nchw_to_kspace(kspace_outputs) * extra_params['norm_mask']
        # kspace_recons = torch.fft(semi_kspace_recons, signal_ndim=3, normalized=True)
        kspace_recons = fft1(semi_kspace_recons, direction='width')

        assert kspace_recons.shape == targets['kspace_targets'].shape, 'Reconstruction and target sizes are different.'

        # kspace_recons = kspace_recons * (1 - extra_params['masks']) + targets['kspace_targets'] * extra_params['masks']

        cmg_recons = ifft1(semi_kspace_recons, direction='height')

        img_recons = complex_abs(cmg_recons)

        recons = {'kspace_recons': kspace_recons, 'cmg_recons': cmg_recons, 'img_recons': img_recons}

        return recons  # Returning scaled reconstructions. Not rescaled.


class OutputReplaceTransformLoggedK(nn.Module):
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

        # kspace_recons = kspace_recons * (1 - extra_params['masks']) + targets['kspace_targets'] * extra_params['masks']

        cmg_recons = ifft2(exp_weighting(kspace_recons))

        img_recons = complex_abs(cmg_recons)

        recons = {'kspace_recons': kspace_recons, 'cmg_recons': cmg_recons, 'img_recons': img_recons}

        return recons  # Returning scaled reconstructions. Not rescaled.


class OutputReplaceTransformNormK(nn.Module):
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

        cmg_recons = ifft2(kspace_recons * extra_params['norm_mask'])

        img_recons = complex_abs(cmg_recons)

        recons = {'kspace_recons': kspace_recons, 'cmg_recons': cmg_recons, 'img_recons': img_recons}

        return recons  # Returning scaled reconstructions. Not rescaled.

