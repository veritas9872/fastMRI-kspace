import torch
import torch.nn.functional as F

import numpy as np

from data.data_transforms import to_tensor, ifft2, complex_abs, apply_mask, kspace_to_nchw


class InputTransformK:
    def __init__(self, mask_func, challenge, device, use_seed=True, divisor=1):

        if challenge not in ('singlecoil', 'multicoil'):
            raise ValueError(f'Challenge should either be "singlecoil" or "multicoil"')
        self.mask_func = mask_func
        self.challenge = challenge
        self.device = device
        self.use_seed = use_seed
        self.divisor = divisor

    def __call__(self, k_slice, target, attrs, file_name, slice_num):
        assert np.iscomplexobj(k_slice), 'kspace must be complex.'
        assert k_slice.shape[-1] % 2 == 0, 'k-space data width must be even.'

        if k_slice.ndim == 2:  # For singlecoil. Makes data processing later on much easier.
            k_slice = np.expand_dims(k_slice, axis=0)
        elif k_slice.ndim != 3:  # Prevents possible errors.
            raise TypeError('Invalid slice type')

        with torch.no_grad():
            kspace_target = to_tensor(k_slice).to(device=self.device).unsqueeze(dim=0)
            cmg_target = ifft2(kspace_target)
            img_target = complex_abs(cmg_target)

            # Use plurals to reduce confusion.
            targets = {'kspace_targets': kspace_target,
                       'cmg_targets': cmg_target,
                       'img_targets': img_target}

            # Apply mask
            seed = None if not self.use_seed else tuple(map(ord, file_name))
            masked_kspace, mask = apply_mask(kspace_target, self.mask_func, seed)

            k_scale = torch.std(masked_kspace)

            extra_params = {'k_scales': k_scale, 'masks': mask}

            masked_kspace /= k_scale
            masked_kspace = kspace_to_nchw(masked_kspace)  # Assumes single batch output.

            margin = masked_kspace.size(-1) % self.divisor
            if margin > 0:
                pad = [(self.divisor - margin) // 2, (1 + self.divisor - margin) // 2]
            else:  # This is a temporary fix to prevent padding by half the divisor when margin=0.
                pad = [0, 0]

            # This pads at the last dimension of a tensor with 0.
            inputs = F.pad(masked_kspace, pad=pad, value=0)

        return inputs, targets, extra_params
