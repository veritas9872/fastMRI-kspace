import torch
import torch.nn.functional as F
import torch.nn

from data.data_transforms import complex_center_crop, complex_height_crop, complex_width_crop

import numpy as np

from data.data_transforms import to_tensor, fft2, ifft2, complex_abs, apply_mask, apply_info_mask, \
    kspace_to_nchw, split_four_cols, fft1, ifft1, apply_PCmask
from data.k2wgt import k2wgt


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


class CyclePrefetch2Device:
    """
    Fetches input data to GPU device.
    Using this to minimize overhead from passing tensors on device from one process to another.
    Also, on HDD devices this will give the data gathering process more time to get data.

    """
    def __init__(self, device):
        self.device = device

    def __call__(self, k_slice_cfc, k_slice_fcf, target, attrs, file_name, slice_num):
        if k_slice_cfc.ndim == 2:  # For singlecoil. Makes data processing later on much easier.
            k_slice_cfc = np.expand_dims(k_slice_cfc, axis=0)
        elif k_slice_cfc.ndim != 3:  # Prevents possible errors.
            raise TypeError('Invalid slice type')

        # I hope that async copy works for passing between processes but I am not sure.
        kspace_target_cfc = to_tensor(k_slice_cfc).to(device=self.device, non_blocking=True)
        kspace_target_fcf = to_tensor(k_slice_fcf).to(device=self.device, non_blocking=True)

        # Necessary since None cannot pass the default collate function.
        if target is None:
            target = 0
        else:
            target = torch.from_numpy(target)
            target = target.to(device=self.device, non_blocking=True)

        return kspace_target_cfc, kspace_target_fcf, target, attrs, file_name, slice_num


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
        if target is None:
            target = 0
        else:
            target = torch.from_numpy(target)
            target = target.to(device=self.device, non_blocking=True)

        return kspace_target, target, attrs, file_name, slice_num


class TrainPreProcessCC:
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

            full_im_target = ifft2(kspace_target) * k_scaling
            cmg_target_toim = complex_center_crop(full_im_target, (320, 320))
            kspace_target = fft2(cmg_target_toim)
            cmg_target = kspace_to_nchw(cmg_target_toim)
            img_target = complex_abs(cmg_target_toim)

            full_im_input = ifft2(masked_kspace)
            cc_im_input = complex_center_crop(full_im_input, (320, 320))

            cmg_input = kspace_to_nchw(cc_im_input)
            img_input = complex_abs(cc_im_input)

            extra_params = {'img_inputs': img_input, 'k_scales': k_scale, 'masks': mask}

            # Use plurals as keys to reduce confusion.
            targets = {'kspace_targets': kspace_target, 'cmg_targets': cmg_target, 'img_targets': img_target}

            margin = masked_kspace.size(-1) % self.divisor
            if margin > 0:
                pad = [(self.divisor - margin) // 2, (1 + self.divisor - margin) // 2]
            else:  # This is a temporary fix to prevent padding by half the divisor when margin=0.
                pad = [0, 0]

            # This pads at the last dimension of a tensor with 0.
            inputs = cmg_input

        return inputs, targets, extra_params


class TrainPreProcessHC:
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

            full_im_target = ifft2(kspace_target) * k_scaling
            # cmg_target_toim = complex_center_crop(full_im_target, (320, 320))
            cmg_target_toim = complex_height_crop(full_im_target, 320)
            kspace_target = fft2(cmg_target_toim)
            cmg_target = kspace_to_nchw(cmg_target_toim)
            img_target = complex_abs(cmg_target_toim)

            full_im_input = ifft2(masked_kspace)
            cc_im_input = complex_height_crop(full_im_input, 320)

            cmg_input = kspace_to_nchw(cc_im_input)
            img_input = complex_abs(cc_im_input)

            extra_params = {'img_inputs': img_input, 'k_scales': k_scaling, 'masks': mask}
            # Use plurals as keys to reduce confusion.
            targets = {'kspace_targets': kspace_target, 'cmg_targets': cmg_target, 'img_targets': img_target}

            margin = cmg_input.size(-1) % self.divisor
            if margin > 0:
                pad = [(self.divisor - margin) // 2, (1 + self.divisor - margin) // 2]
            else:  # This is a temporary fix to prevent padding by half the divisor when margin=0.
                pad = [0, 0]

            # This pads at the last dimension of a tensor with 0.

            inputs = F.pad(cmg_input, pad=pad, value=0)
            targets['cmg_targets'] = F.pad(targets['cmg_targets'], pad=pad, value=0)
            targets['img_targets'] = F.pad(targets['img_targets'], pad=pad, value=0)

        return inputs, targets, extra_params


class TrainPreProcessHCRSS:
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

            full_im_target = ifft2(kspace_target) * k_scaling
            # cmg_target_toim = complex_center_crop(full_im_target, (320, 320))
            cmg_target_toim = complex_height_crop(full_im_target, 320)
            kspace_target = fft2(cmg_target_toim)
            cmg_target = kspace_to_nchw(cmg_target_toim)
            img_target = complex_abs(cmg_target_toim)
            rss_target = (img_target ** 2).sum(dim=1).sqrt().squeeze()

            full_im_input = ifft2(masked_kspace)
            cc_im_input = complex_height_crop(full_im_input, 320)

            cmg_input = kspace_to_nchw(cc_im_input)
            img_input = complex_abs(cc_im_input)

            extra_params = {'img_inputs': img_input, 'k_scales': k_scaling, 'masks': mask}
            # Use plurals as keys to reduce confusion.
            targets = {'kspace_targets': kspace_target, 'cmg_targets': cmg_target, 'img_targets': img_target,
                       'rss_targets': rss_target}

            margin = cmg_input.size(-1) % self.divisor
            if margin > 0:
                pad = [(self.divisor - margin) // 2, (1 + self.divisor - margin) // 2]
                pad2 = [0, 0, (self.divisor - margin) // 2, (1 + self.divisor - margin) // 2]
            else:  # This is a temporary fix to prevent padding by half the divisor when margin=0.
                pad = [0, 0]
                pad2 = [0, 0]

            # This pads at the last dimension of a tensor with 0.

            inputs = F.pad(cmg_input, pad=pad, value=0)
            targets['cmg_targets'] = F.pad(targets['cmg_targets'], pad=pad, value=0)
            targets['img_targets'] = F.pad(targets['img_targets'], pad=pad, value=0)
            targets['rss_targets'] = F.pad(targets['rss_targets'], pad=pad, value=0)
            targets['kspace_targets'] = F.pad(targets['kspace_targets'], pad=pad2, value=0)
            extra_params['masks'] = F.pad(extra_params['masks'], pad=pad2, value=0)

        return inputs, targets, extra_params


class TrainPreProcessHCCC:
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
            mask = torch.squeeze(mask, dim=-1)

            k_scale = torch.std(masked_kspace)
            k_scaling = torch.tensor(1) / k_scale

            masked_kspace *= k_scaling  # Multiplication is faster than division.

            full_im_target = ifft2(kspace_target) * k_scaling
            cmg_target_toim = complex_height_crop(full_im_target, 320)
            cc_cmg_target_toim = complex_width_crop(cmg_target_toim, 320)
            # cmg_target_toim = complex_center_crop(full_im_target, (320, 320))
            kspace_target = fft2(cmg_target_toim)
            kspace_target = kspace_to_nchw(kspace_target)
            cc_kspace_target = fft2(cc_cmg_target_toim)
            cmg_target = kspace_to_nchw(cmg_target_toim)
            cc_cmg_target = kspace_to_nchw(cc_kspace_target)
            img_target = complex_abs(cmg_target_toim)
            cc_img_target = complex_abs(cc_cmg_target_toim)

            full_im_input = ifft2(masked_kspace)
            cc_im_input = complex_height_crop(full_im_input, 320)

            cmg_input = kspace_to_nchw(cc_im_input)
            img_input = complex_abs(cc_im_input)

            extra_params = {'img_inputs': img_input, 'k_scales': k_scaling, 'masks': mask}
            # Use plurals as keys to reduce confusion.
            targets = {'kspace_targets': kspace_target, 'cmg_targets': cmg_target, 'img_targets': img_target,
                       'cc_kspace_targets': cc_kspace_target, 'cc_cmg_targets': cc_cmg_target,
                       'cc_img_targets': cc_img_target}

            margin = cmg_input.size(-1) % self.divisor
            if margin > 0:
                pad = [(self.divisor - margin) // 2, (1 + self.divisor - margin) // 2]
            else:  # This is a temporary fix to prevent padding by half the divisor when margin=0.
                pad = [0, 0]

            # This pads at the last dimension of a tensor with 0.
            inputs = F.pad(cmg_input, pad=pad, value=0)
            targets['kspace_targets'] = F.pad(targets['kspace_targets'], pad=pad, value=0)
            extra_params['masks'] = F.pad(extra_params['masks'], pad=pad, value=0)

        return inputs, targets, extra_params


class TrainPreProcessCCInfo:
    def __init__(self, mask_func, challenge, device, use_seed=True, divisor=1, use_gt=True):
        if challenge not in ('singlecoil', 'multicoil'):
            raise ValueError(f'Challenge should either be "singlecoil" or "multicoil"')
        self.mask_func = mask_func
        self.challenge = challenge
        self.device = device
        self.use_seed = use_seed
        self.divisor = divisor
        self.use_gt = use_gt

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
            masked_kspace, mask, info = apply_info_mask(kspace_target, self.mask_func, seed)

            k_scale = torch.std(masked_kspace)
            k_scaling = torch.tensor(1) / k_scale

            masked_kspace *= k_scaling  # Multiplication is faster than division.

            full_im_target = ifft2(kspace_target) * k_scaling
            cmg_target_toim = complex_center_crop(full_im_target, (320, 320))
            kspace_target = fft2(cmg_target_toim)
            cmg_target = kspace_to_nchw(cmg_target_toim)

            if self.use_gt:
                img_target = target * k_scaling
            else:
                img_target = complex_abs(cmg_target)

            full_im_input = ifft2(masked_kspace)
            cc_im_input = complex_center_crop(full_im_input, (320, 320))

            cmg_input = kspace_to_nchw(cc_im_input)
            img_input = complex_abs(cc_im_input)

            extra_params = {'img_inputs': img_input, 'k_scales': k_scaling}
            extra_params.update(info)

            # Use plurals as keys to reduce confusion.
            targets = {'kspace_targets': kspace_target, 'cmg_targets': cmg_target, 'img_targets': img_target}

            margin = masked_kspace.size(-1) % self.divisor
            if margin > 0:
                pad = [(self.divisor - margin) // 2, (1 + self.divisor - margin) // 2]
            else:  # This is a temporary fix to prevent padding by half the divisor when margin=0.
                pad = [0, 0]

            # This pads at the last dimension of a tensor with 0.
            inputs = cmg_input

        return inputs, targets, extra_params


class TrainPreProcessCCInfoScale:
    def __init__(self, mask_func, challenge, device, use_seed=True, divisor=1, use_gt=True):
        if challenge not in ('singlecoil', 'multicoil'):
            raise ValueError(f'Challenge should either be "singlecoil" or "multicoil"')
        self.mask_func = mask_func
        self.challenge = challenge
        self.device = device
        self.use_seed = use_seed
        self.divisor = divisor
        self.use_gt = use_gt

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
            masked_kspace, mask, info = apply_info_mask(kspace_target, self.mask_func, seed)

            k_scale = torch.std(masked_kspace)
            k_scaling = torch.tensor(1) / k_scale

            masked_kspace *= k_scaling  # Multiplication is faster than division.

            full_im_target = ifft2(kspace_target)
            cmg_target_toim = complex_center_crop(full_im_target, (320, 320)) * k_scaling
            kspace_target = fft2(cmg_target_toim)
            cmg_target = kspace_to_nchw(cmg_target_toim)

            if self.use_gt:
                img_target = complex_abs(cmg_target_toim)
                real_img_target = target * k_scaling
            else:
                img_target = complex_abs(cmg_target_toim)
                real_img_target = None

            full_im_input = ifft2(masked_kspace)
            cc_im_input = complex_center_crop(full_im_input, (320, 320))

            cmg_input = kspace_to_nchw(cc_im_input)
            img_input = complex_abs(cc_im_input)

            extra_params = {'img_inputs': img_input, 'k_scales': k_scaling, 'masks': mask}
            extra_params.update(info)

            # Use plurals as keys to reduce confusion.
            targets = {'kspace_targets': kspace_target, 'cmg_targets': cmg_target, 'img_targets': img_target,
                       'real_targets': real_img_target}

            margin = masked_kspace.size(-1) % self.divisor
            if margin > 0:
                pad = [(self.divisor - margin) // 2, (1 + self.divisor - margin) // 2]
            else:  # This is a temporary fix to prevent padding by half the divisor when margin=0.
                pad = [0, 0]

            # This pads at the last dimension of a tensor with 0.
            inputs = cmg_input

        return inputs, targets, extra_params


class TrainPreProcessCycle:
    def __init__(self, mask_func, challenge, device, use_seed=True, divisor=1, use_gt=True):
        if challenge not in ('singlecoil', 'multicoil'):
            raise ValueError(f'Challenge should either be "singlecoil" or "multicoil"')
        self.mask_func = mask_func
        self.challenge = challenge
        self.device = device
        self.use_seed = use_seed
        self.divisor = divisor
        self.use_gt = use_gt

    def __call__(self, kspace_target_cfc, kspace_target_fcf, target, attrs, file_name, slice_num):
        assert isinstance(kspace_target_cfc, torch.Tensor)
        if kspace_target_cfc.dim() == 4:  # If the collate function does not expand dimensions.
            kspace_target_cfc = kspace_target_cfc.unsqueeze(dim=0)
        elif kspace_target_cfc.dim() != 5:  # Expanded k-space should have 5 dimensions.
            raise RuntimeError('k-space target has invalid shape!')
        if kspace_target_fcf.dim() == 4:  # If the collate function does not expand dimensions.
            kspace_target_fcf = kspace_target_fcf.unsqueeze(dim=0)
        elif kspace_target_fcf.dim() != 5:  # Expanded k-space should have 5 dimensions.
            raise RuntimeError('k-space target has invalid shape!')

        with torch.no_grad():
            # Apply mask
            seed = None if not self.use_seed else tuple(map(ord, file_name))
            masked_kspace, mask, info = apply_info_mask(kspace_target_cfc, self.mask_func, seed)

            k_scale = torch.std(masked_kspace)
            k_scaling = torch.tensor(1) / k_scale

            masked_kspace *= k_scaling  # Multiplication is faster than division.

            full_im_target = ifft2(kspace_target_cfc) * k_scaling
            cmg_target = kspace_to_nchw(full_im_target)
            img_target = complex_abs(full_im_target)

            full_im_input = ifft2(masked_kspace)
            cmg_input = kspace_to_nchw(full_im_input)
            img_input = complex_abs(full_im_input)

            extra_params = {'img_inputs': img_input, 'k_scales': k_scaling, 'masks': mask}
            extra_params.update(info)

            # Use plurals as keys to reduce confusion.
            targets = {'kspace_targets': kspace_target_cfc, 'cmg_targets': cmg_target, 'img_targets': img_target}

            margin = masked_kspace.size(-2) % self.divisor
            if margin > 0:
                pad = [(self.divisor - margin) // 2, (1 + self.divisor - margin) // 2]
                pad2 = [0, 0, (self.divisor - margin) // 2, (1 + self.divisor - margin) // 2]
            else:  # This is a temporary fix to prevent padding by half the divisor when margin=0.
                pad = [0, 0]
                pad2 = [0, 0, 0, 0]

            # This pads at the last dimension of a tensor with 0.
            inputs = F.pad(cmg_input, pad=pad, value=0)
            kspace_target_fcf = F.pad(kspace_target_fcf, pad=pad2, value=0)
            targets['cmg_targets'] = F.pad(targets['cmg_targets'], pad=pad, value=0)
            targets['img_targets'] = F.pad(targets['img_targets'], pad=pad, value=0)
            targets['kspace_targets'] = F.pad(targets['kspace_targets'], pad=pad2, value=0)
            extra_params['masks'] = F.pad(extra_params['masks'], pad=pad2, value=0)

        return inputs, kspace_target_fcf, targets, extra_params


class TrainPreProcessHCCycle:
    def __init__(self, mask_func, challenge, device, use_seed=True, divisor=1, use_gt=True):
        if challenge not in ('singlecoil', 'multicoil'):
            raise ValueError(f'Challenge should either be "singlecoil" or "multicoil"')
        self.mask_func = mask_func
        self.challenge = challenge
        self.device = device
        self.use_seed = use_seed
        self.divisor = divisor
        self.use_gt = use_gt

    def __call__(self, kspace_target_cfc, kspace_target_fcf, target, attrs, file_name, slice_num):
        assert isinstance(kspace_target_cfc, torch.Tensor)
        if kspace_target_cfc.dim() == 4:  # If the collate function does not expand dimensions.
            kspace_target_cfc = kspace_target_cfc.unsqueeze(dim=0)
        elif kspace_target_cfc.dim() != 5:  # Expanded k-space should have 5 dimensions.
            raise RuntimeError('k-space target has invalid shape!')
        if kspace_target_fcf.dim() == 4:  # If the collate function does not expand dimensions.
            kspace_target_fcf = kspace_target_fcf.unsqueeze(dim=0)
        elif kspace_target_fcf.dim() != 5:  # Expanded k-space should have 5 dimensions.
            raise RuntimeError('k-space target has invalid shape!')

        with torch.no_grad():
            # Height crop

            # Apply mask
            seed = None if not self.use_seed else tuple(map(ord, file_name))
            masked_kspace, mask, info = apply_info_mask(kspace_target_cfc, self.mask_func, seed)

            k_scale = torch.std(masked_kspace)
            k_scaling = torch.tensor(1) / k_scale

            masked_kspace *= k_scaling  # Multiplication is faster than division.

            full_im_target = ifft2(kspace_target_cfc) * k_scaling
            hc_im_target = complex_height_crop(full_im_target, 320)
            cmg_target = kspace_to_nchw(hc_im_target)
            img_target = complex_abs(hc_im_target)

            full_im_input = ifft2(masked_kspace)
            hc_im_input = complex_height_crop(full_im_input, 320)
            cmg_input = kspace_to_nchw(hc_im_input)
            img_input = complex_abs(hc_im_input)

            extra_params = {'img_inputs': img_input, 'k_scales': k_scaling, 'masks': mask}
            extra_params.update(info)

            # Use plurals as keys to reduce confusion.
            targets = {'cmg_targets': cmg_target, 'img_targets': img_target}

            margin = masked_kspace.size(-2) % self.divisor
            if margin > 0:
                pad = [(self.divisor - margin) // 2, (1 + self.divisor - margin) // 2]
                pad2 = [0, 0, (self.divisor - margin) // 2, (1 + self.divisor - margin) // 2]
            else:  # This is a temporary fix to prevent padding by half the divisor when margin=0.
                pad = [0, 0]
                pad2 = [0, 0, 0, 0]

            # This pads at the last dimension of a tensor with 0.
            inputs = F.pad(cmg_input, pad=pad, value=0)
            hc_kspace_target_fcf = F.pad(kspace_target_fcf, pad=pad2, value=0)
            targets['cmg_targets'] = F.pad(targets['cmg_targets'], pad=pad, value=0)
            targets['img_targets'] = F.pad(targets['img_targets'], pad=pad, value=0)
            extra_params['masks'] = F.pad(extra_params['masks'], pad=pad2, value=0)

        return inputs, hc_kspace_target_fcf, targets, extra_params


class ValPreProcessCycle:
    def __init__(self, mask_func, challenge, device, use_seed=True, divisor=1, use_gt=True):
        if challenge not in ('singlecoil', 'multicoil'):
            raise ValueError(f'Challenge should either be "singlecoil" or "multicoil"')
        self.mask_func = mask_func
        self.challenge = challenge
        self.device = device
        self.use_seed = use_seed
        self.divisor = divisor
        self.use_gt = use_gt

    def __call__(self, kspace_target, target, attrs, file_name, slice_num):
        assert isinstance(kspace_target, torch.Tensor)
        if kspace_target.dim() == 4:  # If the collate function does not expand dimensions.
            kspace_target = kspace_target.unsqueeze(dim=0)
        elif kspace_target.dim() != 5:  # Expanded k-space should have 5 dimensions.
            raise RuntimeError('k-space target has invalid shape!')

        with torch.no_grad():
            # Apply mask
            seed = None if not self.use_seed else 42
            masked_kspace, mask, info = apply_info_mask(kspace_target, self.mask_func, seed)

            k_scale = torch.std(masked_kspace)
            k_scaling = torch.tensor(1) / k_scale

            masked_kspace *= k_scaling  # Multiplication is faster than division.

            full_im_target = ifft2(kspace_target) * k_scaling
            hc_im_target = complex_height_crop(full_im_target, 320)
            cmg_target = kspace_to_nchw(hc_im_target)
            img_target = complex_abs(hc_im_target)

            full_im_input = ifft2(masked_kspace)
            hc_im_input = complex_height_crop(full_im_input, 320)
            cmg_input = kspace_to_nchw(hc_im_input)
            img_input = complex_abs(hc_im_input)

            extra_params = {'img_inputs': img_input, 'k_scales': k_scaling, 'masks': mask}
            extra_params.update(info)

            # Use plurals as keys to reduce confusion.
            targets = {'cmg_targets': cmg_target, 'img_targets': img_target}

            margin = masked_kspace.size(-2) % self.divisor
            if margin > 0:
                pad = [(self.divisor - margin) // 2, (1 + self.divisor - margin) // 2]
                pad2 = [0, 0, (self.divisor - margin) // 2, (1 + self.divisor - margin) // 2]
            else:  # This is a temporary fix to prevent padding by half the divisor when margin=0.
                pad = [0, 0]
                pad2 = [0, 0, 0, 0]

            # This pads at the last dimension of a tensor with 0.
            inputs = F.pad(cmg_input, pad=pad, value=0)
            targets['cmg_targets'] = F.pad(targets['cmg_targets'], pad=pad, value=0)
            targets['img_targets'] = F.pad(targets['img_targets'], pad=pad, value=0)
            extra_params['masks'] = F.pad(extra_params['masks'], pad=pad2, value=0)

        return inputs, targets, extra_params


class ValPreProcessHCCycle:
    def __init__(self, mask_func, challenge, device, use_seed=True, divisor=1, use_gt=True):
        if challenge not in ('singlecoil', 'multicoil'):
            raise ValueError(f'Challenge should either be "singlecoil" or "multicoil"')
        self.mask_func = mask_func
        self.challenge = challenge
        self.device = device
        self.use_seed = use_seed
        self.divisor = divisor
        self.use_gt = use_gt

    def __call__(self, kspace_target, target, attrs, file_name, slice_num):
        assert isinstance(kspace_target, torch.Tensor)
        if kspace_target.dim() == 4:  # If the collate function does not expand dimensions.
            kspace_target = kspace_target.unsqueeze(dim=0)
        elif kspace_target.dim() != 5:  # Expanded k-space should have 5 dimensions.
            raise RuntimeError('k-space target has invalid shape!')

        with torch.no_grad():
            # Height crop

            # Apply mask
            seed = None if not self.use_seed else 42
            masked_kspace, mask, info = apply_info_mask(kspace_target, self.mask_func, seed)

            k_scale = torch.std(masked_kspace)
            k_scaling = torch.tensor(1) / k_scale

            masked_kspace *= k_scaling  # Multiplication is faster than division.

            full_im_target = ifft2(kspace_target) * k_scaling
            hc_im_target = complex_height_crop(full_im_target, 320)
            cmg_target = kspace_to_nchw(hc_im_target)
            img_target = complex_abs(hc_im_target)

            full_im_input = ifft2(masked_kspace)
            hc_im_input = complex_height_crop(full_im_input, 320)
            cmg_input = kspace_to_nchw(hc_im_input)
            img_input = complex_abs(hc_im_input)

            extra_params = {'img_inputs': img_input, 'k_scales': k_scaling, 'masks': mask}
            extra_params.update(info)

            # Use plurals as keys to reduce confusion.
            targets = {'kspace_targets': kspace_target, 'cmg_targets': cmg_target, 'img_targets': img_target}

            margin = masked_kspace.size(-2) % self.divisor
            if margin > 0:
                pad = [(self.divisor - margin) // 2, (1 + self.divisor - margin) // 2]
                pad2 = [0, 0, (self.divisor - margin) // 2, (1 + self.divisor - margin) // 2]
            else:  # This is a temporary fix to prevent padding by half the divisor when margin=0.
                pad = [0, 0]
                pad2 = [0, 0, 0, 0]

            # This pads at the last dimension of a tensor with 0.
            inputs = F.pad(cmg_input, pad=pad, value=0)
            targets['cmg_targets'] = F.pad(targets['cmg_targets'], pad=pad, value=0)
            targets['img_targets'] = F.pad(targets['img_targets'], pad=pad, value=0)
            targets['kspace_targets'] = F.pad(targets['kspace_targets'], pad=pad2, value=0)
            extra_params['masks'] = F.pad(extra_params['masks'], pad=pad2, value=0)

        return inputs, targets, extra_params


class TrainPreProcessInfoScale:
    def __init__(self, mask_func, challenge, device, use_seed=True, divisor=1, use_gt=True):
        if challenge not in ('singlecoil', 'multicoil'):
            raise ValueError(f'Challenge should either be "singlecoil" or "multicoil"')
        self.mask_func = mask_func
        self.challenge = challenge
        self.device = device
        self.use_seed = use_seed
        self.divisor = divisor
        self.use_gt = use_gt

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
            masked_kspace, mask, info = apply_info_mask(kspace_target, self.mask_func, seed)

            k_scale = torch.std(masked_kspace)
            k_scaling = torch.tensor(1) / k_scale

            masked_kspace *= k_scaling  # Multiplication is faster than division.

            full_im_target = ifft2(kspace_target) * k_scaling
            cmg_target = kspace_to_nchw(full_im_target)
            img_target = complex_abs(full_im_target)

            full_im_input = ifft2(masked_kspace)
            cmg_input = kspace_to_nchw(full_im_input)
            img_input = complex_abs(full_im_input)

            extra_params = {'img_inputs': img_input, 'k_scales': k_scaling, 'masks': mask}
            extra_params.update(info)

            # Use plurals as keys to reduce confusion.
            targets = {'kspace_targets': kspace_target, 'cmg_targets': cmg_target, 'img_targets': img_target}

            margin = masked_kspace.size(-2) % self.divisor
            if margin > 0:
                pad = [(self.divisor - margin) // 2, (1 + self.divisor - margin) // 2]
                pad2 = [0, 0, (self.divisor - margin) // 2, (1 + self.divisor - margin) // 2]
            else:  # This is a temporary fix to prevent padding by half the divisor when margin=0.
                pad = [0, 0]
                pad2 = [0, 0, 0, 0]

            # This pads at the last dimension of a tensor with 0.
            inputs = F.pad(cmg_input, pad=pad, value=0)
            targets['cmg_targets'] = F.pad(targets['cmg_targets'], pad=pad, value=0)
            targets['img_targets'] = F.pad(targets['img_targets'], pad=pad, value=0)
            targets['kspace_targets'] = F.pad(targets['kspace_targets'], pad=pad2, value=0)
            extra_params['masks'] = F.pad(extra_params['masks'], pad=pad2, value=0)

        return inputs, targets, extra_params


class TestPreProcessCCInfo:
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
            masked_kspace, mask, info = apply_info_mask(kspace_target, self.mask_func, seed)

            k_scale = torch.std(masked_kspace)
            k_scaling = torch.tensor(1) / k_scale

            masked_kspace *= k_scaling  # Multiplication is faster than division.

            full_im_target = ifft2(kspace_target) * k_scaling
            cmg_target_toim = complex_center_crop(full_im_target, (320, 320))
            kspace_target = fft2(cmg_target_toim)
            cmg_target = kspace_to_nchw(cmg_target_toim)
            img_target = complex_abs(cmg_target_toim)

            full_im_input = ifft2(masked_kspace)
            cc_im_input = complex_center_crop(full_im_input, (320, 320))

            cmg_input = kspace_to_nchw(cc_im_input)
            img_input = complex_abs(cc_im_input)

            extra_params = {'img_inputs': img_input, 'k_scales': k_scaling}
            extra_params.update(info)

            # Use plurals as keys to reduce confusion.
            targets = {'kspace_targets': kspace_target, 'cmg_targets': cmg_target, 'img_targets': img_target}

            margin = masked_kspace.size(-1) % self.divisor
            if margin > 0:
                pad = [(self.divisor - margin) // 2, (1 + self.divisor - margin) // 2]
            else:  # This is a temporary fix to prevent padding by half the divisor when margin=0.
                pad = [0, 0]

            # This pads at the last dimension of a tensor with 0.
            inputs = cmg_input

        return inputs, targets, extra_params


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


class TrainPreProcessCCK:
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

            full_im_target = ifft2(kspace_target)
            cc_im_target = complex_center_crop(full_im_target, (320, 320))
            kspace_target = fft2(cc_im_target)

            # Apply mask
            seed = None if not self.use_seed else tuple(map(ord, file_name))
            masked_kspace, mask = apply_mask(kspace_target, self.mask_func, seed)

            # Scale
            k_scale = torch.std(masked_kspace)
            k_scaling = torch.tensor(1) / k_scale

            masked_kspace *= k_scaling  # Multiplication is faster than division.

            # Input visualization
            visualize_input = masked_kspace.clone().detach()
            visualize_im = ifft2(visualize_input)
            visualize_im = complex_abs(visualize_im)

            masked_kspace = kspace_to_nchw(masked_kspace)

            extra_params = {'k_scales': k_scale, 'masks': mask, 'img_inputs': visualize_im}

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


class TrainPreProcessCroppedK:
    def __init__(self, mask_func, challenge, device, use_seed=True, divisor=1):
        if challenge not in ('singlecoil', 'multicoil'):
            raise ValueError(f'Challenge should either be "singlecoil" or "multicoil"')
        self.mask_func = mask_func
        self.challenge = challenge
        self.device = device
        self.use_seed = use_seed
        self.divisor = divisor

    def __call__(self, kspace_target, target, attrs, file_name, slice_num, window_size=320):
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
            masked_kspace, mask= apply_mask(kspace_target, self.mask_func, seed)
            k_scale = torch.std(masked_kspace)
            k_scaling = torch.tensor(1) / k_scale

            masked_kspace *= k_scaling  # Multiplication is faster than division.
            masked_kspace = fft1(complex_center_crop(ifft2(masked_kspace), (window_size, window_size)), direction='width')
            # Input visualization
            visualize_input = masked_kspace.clone().detach()
            visualize_im = ifft1(visualize_input, direction='width')
            visualize_im = complex_abs(visualize_im)

            masked_kspace = kspace_to_nchw(masked_kspace)

            extra_params = {'k_scales': k_scale, 'masks': mask, 'img_inputs': visualize_im}

            # Target must be on the same scale as the inputs for scale invariance of data.
            # kspace_target *= (torch.tensor(1) / k_scale)

            # Recall that the Fourier transform is a linear transform.
            # Performing scaling after ifft for numerical stability
            kspace_target *= k_scaling
            kspace_target = fft1(complex_center_crop(ifft2(kspace_target), (window_size, window_size)), direction='width')
            cmg_target = ifft1(kspace_target, direction='width')
            img_target = complex_abs(cmg_target)

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


class TrainPreProcessCCWeightK:
    def __init__(self, mask_func, challenge, device, weight_map, use_seed=True, divisor=1):
        if challenge not in ('singlecoil', 'multicoil'):
            raise ValueError(f'Challenge should either be "singlecoil" or "multicoil"')
        self.mask_func = mask_func
        self.challenge = challenge
        self.device = device
        self.weight_map = weight_map
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

            full_im_target = ifft2(kspace_target)
            cc_im_target = complex_center_crop(full_im_target, (320, 320))
            kspace_target = fft2(cc_im_target)

            # Apply mask
            seed = None if not self.use_seed else tuple(map(ord, file_name))
            # masked_kspace, mask, type_choice = apply_random_mask(kspace_target, self.mask_func, seed)
            masked_kspace, mask = apply_mask(kspace_target, self.mask_func, seed)
            # Add img input for visualization

            k_scale = torch.std(masked_kspace)
            k_scaling = torch.tensor(1) / k_scale

            masked_kspace *= k_scaling  # Multiplication is faster than division.

            # Input visualization
            visualize_input = masked_kspace.clone()
            visualize_im = ifft2(visualize_input)
            visualize_im = complex_abs(visualize_im)

            masked_kspace = masked_kspace * self.weight_map
            masked_kspace = kspace_to_nchw(masked_kspace)

            extra_params = {'k_scales': k_scale, 'masks': mask,
                            'img_inputs': visualize_im, 'weight_map': self.weight_map}

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


class TrainPreProcessCCSplitK:
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

            full_im_target = ifft2(kspace_target)
            cc_im_target = complex_center_crop(full_im_target, (320, 320))
            kspace_target = fft2(cc_im_target)

            # Apply split-stacking to kspace_target
            split_kspace_target = split_four_cols(kspace_target)
            s_split_kspace_target = torch.cat(split_kspace_target, dim=1)

            # Apply mask
            seed = None if not self.use_seed else tuple(map(ord, file_name))
            masked_kspace, mask = apply_mask(kspace_target, self.mask_func, seed)

            k_scale = torch.std(masked_kspace)
            k_scaling = torch.tensor(1) / k_scale

            masked_kspace *= k_scaling  # Multiplication is faster than division.

            # Input visualization
            visualize_input = masked_kspace.clone()
            visualize_im = ifft2(visualize_input)
            visualize_im = complex_abs(visualize_im)

            # Split into four k-space parts
            # split masked-kspace contains 4 elements in a tuple
            split_masked_kspace = split_four_cols(masked_kspace)
            s_split_masked_kspace = torch.cat(split_masked_kspace, dim=1)
            masked_kspace = kspace_to_nchw(s_split_masked_kspace)

            extra_params = {'k_scales': k_scale, 'masks': mask, 'img_inputs': visualize_im}

            # Recall that the Fourier transform is a linear transform.
            # Performing scaling after ifft for numerical stability
            cmg_target = ifft2(s_split_kspace_target) * k_scaling
            img_target = complex_abs(cmg_target)
            s_split_kspace_target *= k_scaling

            # Use plurals as keys to reduce confusion.
            targets = {'kspace_targets': s_split_kspace_target, 'cmg_targets': cmg_target, 'img_targets': img_target}

            margin = masked_kspace.size(-1) % self.divisor
            if margin > 0:
                pad = [(self.divisor - margin) // 2, (1 + self.divisor - margin) // 2]
            else:  # This is a temporary fix to prevent padding by half the divisor when margin=0.
                pad = [0, 0]

            # This pads at the last dimension of a tensor with 0.
            inputs = F.pad(masked_kspace, pad=pad, value=0)

        return inputs, targets, extra_params


class TrainPreProcessNormCCK:
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

            # CC
            full_im_target = ifft2(kspace_target)
            cc_im_target = complex_center_crop(full_im_target, (320, 320))
            kspace_target = fft2(cc_im_target)

            # Apply mask
            seed = None if not self.use_seed else tuple(map(ord, file_name))
            masked_kspace, mask = apply_mask(kspace_target, self.mask_func, seed)

            k_scale = torch.std(masked_kspace)
            k_scaling = torch.tensor(1) / k_scale

            masked_kspace *= k_scaling  # Multiplication is faster than division.

            # Input visualization
            visualize_input = masked_kspace.clone()
            visualize_im = ifft2(visualize_input)
            visualize_im = complex_abs(visualize_im)

            n, c, h, w, i = masked_kspace.size()
            line = complex_abs(masked_kspace[..., h//2:, w//2, :]).squeeze()
            line = line.mean(dim=0)

            x = torch.arange(-w // 2, w // 2, 1, dtype=torch.float32, device=self.device)
            y = torch.arange(-h // 2, h // 2, 1, dtype=torch.float32, device=self.device)
            cc = torch.arange(0, c, 1, dtype=torch.float32, device=self.device)

            ccc, yy, xx = torch.meshgrid(cc, y, x)
            ccc = ccc.to(dtype=torch.int64)

            R = torch.floor(torch.sqrt(xx ** 2 + yy ** 2)).to(dtype=torch.int64)
            R[R >= len(line)] = len(line)-1
            norm_mask = line[R] + 1E-10
            norm_mask = torch.stack([norm_mask, norm_mask], dim=-1)
            masked_kspace = masked_kspace / norm_mask
            masked_kspace += (1 - mask) * torch.randn(masked_kspace.shape, device=self.device) * 0.01

            masked_kspace = kspace_to_nchw(masked_kspace)

            extra_params = {'k_scales': k_scaling, 'masks': mask, 'norm_mask': norm_mask, 'img_inputs': visualize_im}

            # Target must be on the same scale as the inputs for scale invariance of data.
            # kspace_target *= (torch.tensor(1) / k_scale)

            # Recall that the Fourier transform is a linear transform.
            # Performing scaling after ifft for numerical stability
            kspace_target *= k_scaling
            cmg_target = ifft2(kspace_target)
            img_target = complex_abs(cmg_target)
            kspace_target = kspace_target / norm_mask
            # kspace_target = log_weighting(kspace_target)

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


class TrainPreProcessLaplaceCCK:
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

            # CC
            full_im_target = ifft2(kspace_target)
            cc_im_target = complex_center_crop(full_im_target, (320, 320))
            kspace_target = fft2(cc_im_target)

            # Apply mask
            seed = None if not self.use_seed else tuple(map(ord, file_name))
            masked_kspace, mask = apply_mask(kspace_target, self.mask_func, seed)

            k_scale = torch.std(masked_kspace)
            k_scaling = torch.tensor(1) / k_scale

            masked_kspace *= k_scaling  # Multiplication is faster than division.

            # Input visualization
            visualize_input = masked_kspace.clone()
            visualize_im = ifft2(visualize_input)
            visualize_im = complex_abs(visualize_im)

            n, c, h, w, i = masked_kspace.size()
            line = complex_abs(masked_kspace[..., h//2:, w//2, :]).squeeze()
            line = line.mean(dim=0)

            x = torch.arange(-w // 2, w // 2, 1, dtype=torch.float32, device=self.device)
            y = torch.arange(-h // 2, h // 2, 1, dtype=torch.float32, device=self.device)
            cc = torch.arange(0, c, 1, dtype=torch.float32, device=self.device)

            ccc, yy, xx = torch.meshgrid(cc, y, x)
            ccc = ccc.to(dtype=torch.int64)

            R = torch.floor(torch.sqrt(xx ** 2 + yy ** 2)).to(dtype=torch.int64)
            R[R >= len(line)] = len(line)-1
            norm_mask = line[R] + 1E-10
            norm_mask = torch.stack([norm_mask, norm_mask], dim=-1)
            masked_kspace = masked_kspace / norm_mask
            masked_kspace += (1 - mask) * torch.randn(masked_kspace.shape, device=self.device) * 0.01

            masked_kspace = kspace_to_nchw(masked_kspace)

            extra_params = {'k_scales': k_scaling, 'masks': mask, 'norm_mask': norm_mask, 'img_inputs': visualize_im}

            # Target must be on the same scale as the inputs for scale invariance of data.
            # kspace_target *= (torch.tensor(1) / k_scale)

            # Recall that the Fourier transform is a linear transform.
            # Performing scaling after ifft for numerical stability
            kspace_target *= k_scaling
            cmg_target = ifft2(kspace_target)
            img_target = complex_abs(cmg_target)
            kspace_target = kspace_target / norm_mask
            # kspace_target = log_weighting(kspace_target)

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


class TrainPreProcessPCK:
    def __init__(self, mask_func, challenge, device, use_seed=True, divisor=1):
        if challenge not in ('singlecoil', 'multicoil'):
            raise ValueError(f'Challenge should either be "singlecoil" or "multicoil"')
        self.mask_func = mask_func
        self.challenge = challenge
        self.device = device
        self.use_seed = use_seed
        self.divisor = divisor

    def __call__(self, kspace_target, target, attrs, file_name, slice_num, window_size=320):
        assert isinstance(kspace_target, torch.Tensor)
        if kspace_target.dim() == 4:  # If the collate function does not expand dimensions.
            kspace_target = kspace_target.unsqueeze(dim=0)
        elif kspace_target.dim() != 5:  # Expanded k-space should have 5 dimensions.
            raise RuntimeError('k-space target has invalid shape!')

        if kspace_target.size(0) != 1:
            raise NotImplementedError('Batch size should be 1 for now.')

        with torch.no_grad():
            # CC
            full_im_target = ifft2(kspace_target)
            cc_im_target = complex_center_crop(full_im_target, (320, 320))
            kspace_target = fft2(cc_im_target)

            # Apply mask
            seed = None if not self.use_seed else tuple(map(ord, file_name))
            masked_kspace, mask, acceleration, mask_holder = apply_PCmask(kspace_target, self.mask_func, seed)
            mask_holder = np.expand_dims(np.expand_dims(mask_holder, -1), 0)
            mask_holder = np.tile(mask_holder, (1, 1, 1, 1, 2))
            mask_holder = torch.from_numpy(mask_holder).to(device='cuda:0')

            k_scale = torch.std(masked_kspace)
            k_scaling = torch.tensor(1) / k_scale

            masked_kspace *= k_scaling  # Multiplication is faster than division.
            masked_kspace = ifft1(masked_kspace, direction='height')
            # Input visualization
            visualize_input = masked_kspace.clone().detach()
            visualize_im = ifft1(visualize_input, direction='width')
            visualize_im = complex_abs(visualize_im)

            masked_kspace = kspace_to_nchw(masked_kspace)
            mask_holder = kspace_to_nchw(mask_holder)

            extra_params = {'k_scales': k_scale, 'masks': mask, 'img_inputs': visualize_im,
                            'acceleration': acceleration, 'mask_holder': mask_holder}

            # Target must be on the same scale as the inputs for scale invariance of data.
            # kspace_target *= (torch.tensor(1) / k_scale)

            # Recall that the Fourier transform is a linear transform.
            # Performing scaling after ifft for numerical stability
            kspace_target *= k_scaling
            kspace_target = ifft1(kspace_target, direction='height')
            cmg_target = ifft1(kspace_target, direction='width')
            img_target = complex_abs(cmg_target)

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


class TrainPreProcessIMG:
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

            cmg_target = ifft2(kspace_target) * k_scaling
            cmg_target = kspace_to_nchw(cmg_target)
            img_target = complex_abs(cmg_target)

            cmg_input = ifft2(masked_kspace)
            cmg_input = kspace_to_nchw(cmg_input)
            img_input = complex_abs(cmg_input)

            extra_params = {'img_inputs': img_input, 'k_scales': k_scale, 'masks': mask}

            # Use plurals as keys to reduce confusion.
            targets = {'kspace_targets': kspace_target, 'cmg_targets': cmg_target, 'img_targets': img_target}

            margin = masked_kspace.size(-1) % self.divisor
            if margin > 0:
                pad = [(self.divisor - margin) // 2, (1 + self.divisor - margin) // 2]
            else:  # This is a temporary fix to prevent padding by half the divisor when margin=0.
                pad = [0, 0]

            # This pads at the last dimension of a tensor with 0.
            inputs = F.pad(cmg_input, pad=pad, value=0)

        return inputs, targets, extra_params


class PreProcessIMG:
    """
    Pre-processing for image-to-image learning.
    """
    def __init__(self, mask_func, challenge, device, augment_data=False,
                 use_seed=True, crop_center=True, resolution=320, divisor=1):
        assert callable(mask_func), '`mask_func` must be a callable function.'
        if challenge not in ('singlecoil', 'multicoil'):
            raise ValueError(f'Challenge should either be "singlecoil" or "multicoil"')

        self.mask_func = mask_func
        self.challenge = challenge
        self.device = device
        self.augment_data = augment_data
        self.use_seed = use_seed
        self.crop_center = crop_center
        self.resolution = resolution  # Only has effect when center_crop is True.
        self.divisor = divisor

    def __call__(self, kspace_target, target, attrs, file_name, slice_num):
        assert isinstance(kspace_target, torch.Tensor), 'k-space target was expected to be a Pytorch Tensor.'
        if kspace_target.dim() == 3:  # If the collate function does not expand dimensions for single-coil.
            kspace_target = kspace_target.expand(1, 1, -1, -1, -1)
        elif kspace_target.dim() == 4:  # If the collate function does not expand dimensions for multi-coil.
            kspace_target = kspace_target.expand(1, -1, -1, -1, -1)
        elif kspace_target.dim() != 5:  # Expanded k-space should have 5 dimensions.
            raise RuntimeError('k-space target has invalid shape!')

        if kspace_target.size(0) != 1:
            raise NotImplementedError('Batch size should be 1 for now.')

        with torch.no_grad():
            # Apply mask
            seed = None if not self.use_seed else tuple(map(ord, file_name))
            masked_kspace, mask, info = apply_info_mask(kspace_target, self.mask_func, seed)

            image = complex_abs(ifft2(masked_kspace))

            if self.crop_center:
                image = center_crop(image, shape=(self.resolution, self.resolution))

            img_scale = torch.std(image)
            image /= img_scale

            extra_params = {'img_scales': img_scale, 'masks': mask}
            extra_params.update(info)
            extra_params.update(attrs)

            img_target = complex_abs(ifft2(kspace_target))
            img_target /= img_scale

            if self.crop_center:
                img_target = center_crop(img_target, shape=(self.resolution, self.resolution))

            # Data augmentation by flipping images up-down and left-right.
            if self.augment_data:
                flip_lr = torch.rand(()) < 0.5
                flip_ud = torch.rand(()) < 0.5

                if flip_lr and flip_ud:
                    image = torch.flip(image, dims=(-2, -1))
                    img_target = torch.flip(img_target, dims=(-2, -1))
                    target = torch.flip(target, dims=(-2, -1))

                elif flip_ud:
                    image = torch.flip(image, dims=(-2,))
                    img_target = torch.flip(img_target, dims=(-2,))
                    target = torch.flip(target, dims=(-2,))

                elif flip_lr:
                    image = torch.flip(image, dims=(-1,))
                    img_target = torch.flip(img_target, dims=(-1,))
                    target = torch.flip(target, dims=(-1,))

            # Use plurals as keys to reduce confusion.
            targets = {'img_targets': img_target, 'img_inputs': image}

            if self.challenge == 'multicoil':
                targets['rss_targets'] = target

            margin = image.size(-1) % self.divisor
            if margin > 0:
                pad = [(self.divisor - margin) // 2, (1 + self.divisor - margin) // 2]
            else:  # This is a fix to prevent padding by half the divisor when margin=0.
                pad = [0, 0]

            # This pads at the last dimension of a tensor with 0.
            inputs = F.pad(image, pad=pad, value=0)

        return inputs, targets, extra_params