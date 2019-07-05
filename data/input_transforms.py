import torch
import torch.nn.functional as F

import numpy as np

from data.data_transforms import to_tensor, ifft2, complex_abs, apply_mask, kspace_to_nchw, ifft1


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


class Prefetch2Device:
    """
    Fetches input data to GPU device.
    Using this to minimize overhead from passing tensors on device from one process to another.
    Also, on HDD devices this will give the data gathering process more time to get data.

    """
    def __init__(self, device):
        self.device = device

    def __call__(self, k_slice, target, attrs, file_name, slice_num):
        if k_slice.ndim == 2:  # For singlecoil. Makes data processing later on much easier.
            k_slice = np.expand_dims(k_slice, axis=0)
        elif k_slice.ndim != 3:  # Prevents possible errors.
            raise TypeError('Invalid slice type')

        # I hope that async copy works for passing between processes but I am not sure.
        kspace_target = to_tensor(k_slice).to(device=self.device, non_blocking=True)

        # Necessary since None cannot pass the default collate function.
        target = 0 if target is None else target.to(device=self.device, non_blocking=True)

        return kspace_target, target, attrs, file_name, slice_num


class TrainPreProcessK:
    def __init__(self, mask_func, challenge, device, use_seed=True, divisor=1):
        if challenge not in ('singlecoil', 'multicoil'):
            raise ValueError(f'Challenge should either be "singlecoil" or "multicoil"')
        self.mask_func = mask_func
        self.challenge = challenge
        self.device = device
        self.use_seed = use_seed
        self.divisor = divisor

    def __call__(self, kspace_target, target, attrs, file_name, slice_num):
        assert isinstance(kspace_target, torch.Tensor)
        if kspace_target.dim() == 4:  # If the collate function does not expand dimensions.
            kspace_target = kspace_target.unsqueeze(dim=0)
        elif kspace_target.dim() != 5:  # Expanded k-space should have 5 dimensions.
            raise RuntimeError('k-space target has invalid shape!')

        if kspace_target.size(0) != 1:
            raise NotImplementedError('Batch size should be 1 for now.')

        with torch.no_grad():
            # Apply mask
            seed = None if not self.use_seed else tuple(map(ord, file_name))
            masked_kspace, mask = apply_mask(kspace_target, self.mask_func, seed)

            k_scale = torch.std(masked_kspace)
            k_scaling = torch.tensor(1) / k_scale

            masked_kspace *= k_scaling  # Multiplication is faster than division.
            masked_kspace = kspace_to_nchw(masked_kspace)

            extra_params = {'k_scales': k_scale, 'masks': mask}

            # Target must be on the same scale as the inputs for scale invariance of data.
            # kspace_target *= (torch.tensor(1) / k_scale)

            # Recall that the Fourier transform is a linear transform.
            # Performing scaling after ifft for numerical stability
            cmg_target = ifft2(kspace_target) * k_scaling
            img_target = complex_abs(cmg_target)
            kspace_target *= k_scaling

            # Use plurals as keys to reduce confusion.
            targets = {'kspace_targets': kspace_target, 'cmg_targets': cmg_target, 'img_targets': img_target}

            margin = masked_kspace.size(-1) % self.divisor
            if margin > 0:
                pad = [(self.divisor - margin) // 2, (1 + self.divisor - margin) // 2]
            else:  # This is a temporary fix to prevent padding by half the divisor when margin=0.
                pad = [0, 0]

            # This pads at the last dimension of a tensor with 0.
            inputs = F.pad(masked_kspace, pad=pad, value=0)

        return inputs, targets, extra_params


class PartialKPreProcess:
    def __init__(self, mask_func, challenge, device, use_seed=True, divisor=1):
        if challenge not in ('singlecoil', 'multicoil'):
            raise ValueError(f'Challenge should either be "singlecoil" or "multicoil"')
        self.mask_func = mask_func
        self.challenge = challenge
        self.device = device
        self.use_seed = use_seed
        self.divisor = divisor

    def __call__(self, kspace_target, target, attrs, file_name, slice_num):
        assert isinstance(kspace_target, torch.Tensor)
        if kspace_target.dim() == 4:  # If the collate function does not expand dimensions.
            kspace_target = kspace_target.unsqueeze(dim=0)
        elif kspace_target.dim() != 5:  # Expanded k-space should have 5 dimensions.
            raise RuntimeError('k-space target has invalid shape!')

        if kspace_target.size(0) != 1:
            raise NotImplementedError('Batch size should be 1 for now.')

        with torch.no_grad():
            # Apply mask
            seed = None if not self.use_seed else tuple(map(ord, file_name))
            masked_kspace, mask = apply_mask(kspace_target, self.mask_func, seed)





