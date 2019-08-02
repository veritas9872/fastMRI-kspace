import torch
from torch import nn
import numpy as np

from data.data_transforms import nchw_to_kspace, ifft2, complex_abs, ifft1, fft1, root_sum_of_squares


# class OutputReplaceTransformK(nn.Module):
#     """
#     Outputs are expected to be k-space data that need reshaping and conversion to the image domain.
#     Inputs and targets are expected to be scaled already.
#     Currently, the implementation expects only 1 batch.
#     Also, I removed rescaling as the loss needs to be calculated on the standardized values for data scale invariance.
#     Final reconstructions can be obtained by dividing by the k_scale value since
#     the Fourier transform and its relatives are all linear functions.
#     """
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, kspace_outputs, targets, extra_params):
#
#         if kspace_outputs.size(0) > 1:
#             raise NotImplementedError('Only one batch at a time for now.')
#
#         # For removing width dimension padding. Recall that k-space form has 2 as last dim size.
#         left = (kspace_outputs.size(-1) - targets['kspace_targets'].size(-2)) // 2
#         right = left + targets['kspace_targets'].size(-2)
#
#         # Cropping width dimension by pad. # Multiply by scales to restore the original scaling. --> Not rescaling!!!
#         kspace_outputs = kspace_outputs[..., left:right]  # * extra_params['k_scales']
#
#         kspace_recons = nchw_to_kspace(kspace_outputs)
#
#         assert kspace_recons.shape == targets['kspace_targets'].shape, 'Reconstruction and target sizes are different.'
#
#         kspace_recons = kspace_recons * (1 - extra_params['masks']) + targets['kspace_targets'] * extra_params['masks']
#
#         cmg_recons = ifft2(kspace_recons)
#
#         img_recons = complex_abs(cmg_recons)
#
#         recons = {'kspace_recons': kspace_recons, 'cmg_recons': cmg_recons, 'img_recons': img_recons}
#
#         return recons  # Returning scaled reconstructions. Not rescaled.


# class PostProcessSemiK(nn.Module):
#     def __init__(self, direction='height'):
#         """
#         Post-Processing function for Semi-k-space learning.
#
#         Args:
#             direction: The direction that the input data was transformed.
#                 The output data will be transformed in the other direction to convert to image data.
#         """
#         super().__init__()
#
#         if direction == 'height':
#             self.recon_direction = 'width'
#         elif direction == 'width':
#             self.recon_direction = 'height'
#         else:
#             raise ValueError('`direction` should either be `height` or `width')
#
#         self.direction = direction
#
#     def forward(self, sks_outputs, targets, extra_params):
#         if sks_outputs.size(0) > 1:
#             raise NotImplementedError('Batch size is expected to be 1 for now.')
#
#         sks_targets = targets['sks_targets']
#         mask = extra_params['masks']
#
#         # For removing width dimension padding. Recall that k-space form has 2 as last dim size.
#         left = (sks_outputs.size(-1) - sks_targets.size(-2)) // 2
#         right = left + sks_targets.size(-2)
#
#         sks_recons = nchw_to_kspace(sks_outputs[..., left:right])
#
#         assert sks_recons.shape == sks_targets.shape, 'Reconstruction and target sizes are different.'
#
#         sks_recons = sks_recons * (1 - mask) + sks_targets * mask
#         kspace_recons = ifft1(sks_recons, direction=self.direction)
#         cmg_recons = ifft1(sks_recons, direction=self.recon_direction)
#         img_recons = complex_abs(cmg_recons)
#
#         # This is inefficient memory-wise but memory is not a serious issue for me right now.
#         recons = {'sks_recons': sks_recons, 'kspace_recons': kspace_recons,
#                   'cmg_recons': cmg_recons, 'img_recons': img_recons}
#
#         return recons


class WeightedReplacePostProcessK(nn.Module):
    def __init__(self, weighted=True, replace=True):
        super().__init__()
        self.weighted = weighted
        self.replace = replace

    def forward(self, kspace_outputs, targets, extra_params):
        if kspace_outputs.size(0) > 1:
            raise NotImplementedError('Only one slice at a time for now.')

        kspace_targets = targets['kspace_targets']

        # For removing width dimension padding. Recall that k-space form has 2 as last dim size.
        left = (kspace_outputs.size(-1) - kspace_targets.size(-2)) // 2
        right = left + kspace_targets.size(-2)

        # Cropping width dimension by pad.
        kspace_recons = nchw_to_kspace(kspace_outputs[..., left:right])

        assert kspace_recons.shape == kspace_targets.shape, 'Reconstruction and target sizes are different.'
        assert (kspace_recons.size(-3) % 2 == 0) and (kspace_recons.size(-2) % 2 == 0), \
            'Not impossible but not expected to have sides with odd lengths.'

        # Removing weighting.
        if self.weighted:
            weighting = extra_params['weightings']
            kspace_recons = kspace_recons / weighting

        if self.replace:  # Replace with original k-space if replace=True
            mask = extra_params['masks']
            kspace_recons = kspace_recons * (1 - mask) + kspace_targets * mask

        cmg_recons = ifft2(kspace_recons)
        img_recons = complex_abs(cmg_recons)
        recons = {'kspace_recons': kspace_recons, 'cmg_recons': cmg_recons, 'img_recons': img_recons}

        return recons  # Returning scaled reconstructions. Not rescaled.


class WeightedReplacePostProcessSemiK(nn.Module):
    def __init__(self, weighted=True, replace=True, direction='height'):
        super().__init__()
        self.weighted = weighted
        self.replace = replace

        if direction == 'height':
            self.recon_direction = 'width'
        elif direction == 'width':
            self.recon_direction = 'height'
        else:
            raise ValueError('`direction` should either be `height` or `width')

        self.direction = direction

    def forward(self, semi_kspace_outputs, targets, extra_params):
        if semi_kspace_outputs.size(0) > 1:
            raise NotImplementedError('Only one batch at a time for now.')

        semi_kspace_targets = targets['semi_kspace_targets']
        # For removing width dimension padding. Recall that k-space form has 2 as last dim size.
        left = (semi_kspace_outputs.size(-1) - semi_kspace_targets.size(-2)) // 2
        right = left + semi_kspace_targets.size(-2)

        # Cropping width dimension by pad.
        semi_kspace_recons = nchw_to_kspace(semi_kspace_outputs[..., left:right])

        assert semi_kspace_recons.shape == semi_kspace_targets.shape, 'Reconstruction and target sizes are different.'
        assert (semi_kspace_recons.size(-3) % 2 == 0) and (semi_kspace_recons.size(-2) % 2 == 0), \
            'Not impossible but not expected to have sides with odd lengths.'

        # Removing weighting.
        if self.weighted:
            weighting = extra_params['weightings']
            semi_kspace_recons = semi_kspace_recons / weighting

        if self.replace:
            mask = extra_params['masks']
            semi_kspace_recons = semi_kspace_recons * (1 - mask) + semi_kspace_targets * mask

        kspace_recons = fft1(semi_kspace_recons, direction=self.direction)
        cmg_recons = ifft1(semi_kspace_recons, direction=self.recon_direction)
        img_recons = complex_abs(cmg_recons)

        recons = {'semi_kspace_recons': semi_kspace_recons, 'kspace_recons': kspace_recons,
                  'cmg_recons': cmg_recons, 'img_recons': img_recons}

        return recons  # Returning scaled reconstructions. Not rescaled.


class PostProcessWK(nn.Module):
    def __init__(self, weighted=True, replace=True, residual_acs=False, resolution=320):
        super().__init__()
        self.weighted = weighted
        self.replace = replace
        self.resolution = resolution
        self.residual_acs = residual_acs

    def forward(self, kspace_outputs, targets, extra_params):
        if kspace_outputs.size(0) > 1:
            raise NotImplementedError('Only one slice at a time for now.')

        kspace_targets = targets['kspace_targets']

        # For removing width dimension padding. Recall that k-space form has 2 as last dim size.
        left = (kspace_outputs.size(-1) - kspace_targets.size(-2)) // 2
        right = left + kspace_targets.size(-2)

        # Cropping width dimension by pad.
        kspace_recons = nchw_to_kspace(kspace_outputs[..., left:right])
        assert kspace_recons.shape == kspace_targets.shape, 'Reconstruction and target sizes are different.'
        assert (kspace_recons.size(-3) % 2 == 0) and (kspace_recons.size(-2) % 2 == 0), \
            'Not impossible but not expected to have sides with odd lengths.'

        # Removing weighting.
        if self.weighted:
            weighting = extra_params['weightings']
            kspace_recons = kspace_recons / weighting

        if self.residual_acs:
            num_low_freqs = extra_params['num_low_frequency']
            acs_mask = find_acs_mask(kspace_recons, num_low_freqs)
            kspace_recons = kspace_recons + acs_mask * kspace_targets

        if self.replace:  # Replace with original k-space if replace=True
            mask = extra_params['masks']
            kspace_recons = kspace_recons * (1 - mask) + kspace_targets * mask

        cmg_recons = ifft2(kspace_recons)
        img_recons = complex_abs(cmg_recons)
        recons = {'kspace_recons': kspace_recons, 'cmg_recons': cmg_recons, 'img_recons': img_recons}

        if img_recons.size(1) == 15:
            top = (img_recons.size(-2) - self.resolution) // 2
            left = (img_recons.size(-1) - self.resolution) // 2
            rss_recon = img_recons[:, :, top:top + self.resolution, left:left + self.resolution]
            rss_recon = root_sum_of_squares(rss_recon, dim=1).squeeze()  # rss_recon is in 2D
            recons['rss_recons'] = rss_recon

        return recons  # Returning scaled reconstructions. Not rescaled.


class PostProcessWSemiK(nn.Module):
    def __init__(self, weighted=True, replace=True, residual_acs=False, direction='height', resolution=320):
        super().__init__()
        self.weighted = weighted
        self.replace = replace
        self.resolution = resolution
        self.residual_acs = residual_acs

        if direction == 'height':
            self.recon_direction = 'width'
        elif direction == 'width':
            self.recon_direction = 'height'
        else:
            raise ValueError('`direction` should either be `height` or `width')

        self.direction = direction

    def forward(self, semi_kspace_outputs, targets, extra_params):
        if semi_kspace_outputs.size(0) > 1:
            raise NotImplementedError('Only one at a time for now.')

        semi_kspace_targets = targets['semi_kspace_targets']
        # For removing width dimension padding. Recall that k-space form has 2 as last dim size.
        left = (semi_kspace_outputs.size(-1) - semi_kspace_targets.size(-2)) // 2
        right = left + semi_kspace_targets.size(-2)

        # Cropping width dimension by pad.
        semi_kspace_recons = nchw_to_kspace(semi_kspace_outputs[..., left:right])

        assert semi_kspace_recons.shape == semi_kspace_targets.shape, 'Reconstruction and target sizes are different.'
        assert (semi_kspace_recons.size(-3) % 2 == 0) and (semi_kspace_recons.size(-2) % 2 == 0), \
            'Not impossible but not expected to have sides with odd lengths.'

        # Removing weighting.
        if self.weighted:
            weighting = extra_params['weightings']
            semi_kspace_recons = semi_kspace_recons / weighting

        if self.residual_acs:
            num_low_freqs = extra_params['num_low_frequency']
            acs_mask = find_acs_mask(semi_kspace_recons, num_low_freqs)
            semi_kspace_recons = semi_kspace_recons + acs_mask * semi_kspace_targets

        if self.replace:
            mask = extra_params['masks']
            semi_kspace_recons = semi_kspace_recons * (1 - mask) + semi_kspace_targets * mask

        kspace_recons = fft1(semi_kspace_recons, direction=self.direction)
        cmg_recons = ifft1(semi_kspace_recons, direction=self.recon_direction)
        img_recons = complex_abs(cmg_recons)

        recons = {'semi_kspace_recons': semi_kspace_recons, 'kspace_recons': kspace_recons,
                  'cmg_recons': cmg_recons, 'img_recons': img_recons}

        if semi_kspace_targets.size(1) == 15:
            top = (img_recons.size(-2) - self.resolution) // 2
            left = (img_recons.size(-1) - self.resolution) // 2
            rss_recons = img_recons[:, :, top:top+self.resolution, left:left+self.resolution]
            rss_recons = root_sum_of_squares(rss_recons, dim=1).squeeze()
            rss_recons *= extra_params['sk_scales']  # This value was divided in the inputs. It is thus multiplied here.
            recons['rss_recons'] = rss_recons

        return recons  # Returning scaled reconstructions. Not rescaled. RSS images are rescaled.


def find_acs_mask(kspace_recons: torch.Tensor, num_low_freqs: int):
    assert kspace_recons.dim() == 5, 'Reconstructed tensor in k-space format is expected.'
    num_cols = kspace_recons.size(-2)
    pad = (num_cols - num_low_freqs + 1) // 2
    mask = np.zeros(num_cols, dtype=bool)
    mask[pad:pad+num_low_freqs] = True
    mask = torch.from_numpy(mask).to(dtype=kspace_recons.dtype, device=kspace_recons.device).view(1, 1, 1, -1, 1)
    return mask
