import torch
import torch.nn.functional as F

import numpy as np

from data.data_transforms import to_tensor, ifft2, complex_abs, apply_mask, kspace_to_nchw, fft1, ifft1, log_weighting, center_crop, complex_phase, complex_center_crop, fft2


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


class TrainPreProcessKK:
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
            masked_kspace = masked_kspace + torch.ones(masked_kspace.shape, device=self.device) * (1 - mask)
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


class TrainPreProcessSemiK:
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
            masked_kspace = ifft1(masked_kspace, direction='width')
            # n, c, h, w, i = masked_kspace.size()
            norm_mask = complex_abs(masked_kspace).max(dim=-1).values.unsqueeze(dim=-1).unsqueeze(dim=-1)
            masked_kspace = masked_kspace / norm_mask
            masked_kspace = kspace_to_nchw(masked_kspace)

            extra_params = {'k_scales': k_scale, 'masks': mask, 'norm_mask': norm_mask}

            # Target must be on the same scale as the inputs for scale invariance of data.
            # kspace_target *= (torch.tensor(1) / k_scale)

            # Recall that the Fourier transform is a linear transform.
            # Performing scaling after ifft for numerical stability
            kspace_target = kspace_target * k_scaling
            cmg_target = ifft2(kspace_target)
            img_target = complex_abs(cmg_target)
            kspace_target = ifft1(kspace_target, direction='width')

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


class TrainPreProcessSemiK2:
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
            masked_kspace = ifft1(masked_kspace, direction='height')

            # abs_masked_kspace = complex_abs(masked_kspace)

            norm_mask = complex_abs(masked_kspace).max(dim=-2).values.view(15, 1, -1, 1) + 1E-10
            norm_mask = norm_mask + norm_mask[norm_mask > 1E-9].median() * (1 - mask)
            masked_kspace = masked_kspace / norm_mask
            masked_kspace = masked_kspace - 0.05
            masked_kspace = kspace_to_nchw(masked_kspace)

            extra_params = {'k_scales': k_scale, 'masks': mask, 'norm_mask': norm_mask}

            # Target must be on the same scale as the inputs for scale invariance of data.
            # kspace_target *= (torch.tensor(1) / k_scale)

            # Recall that the Fourier transform is a linear transform.
            # Performing scaling after ifft for numerical stability
            kspace_target = kspace_target * k_scaling
            semi_kspace_target = ifft1(kspace_target, direction='height')
            # cmg_target = ifft2(kspace_target)
            cmg_target = ifft1(semi_kspace_target, direction='width')
            img_target = complex_abs(cmg_target)
            semi_kspace_target = semi_kspace_target / norm_mask


            # Use plurals as keys to reduce confusion.
            targets = {'kspace_targets': kspace_target, 'cmg_targets': cmg_target, 'img_targets': img_target, 'semi_kspace_target': semi_kspace_target}

            margin = masked_kspace.size(-1) % self.divisor
            if margin > 0:
                pad = [(self.divisor - margin) // 2, (1 + self.divisor - margin) // 2]
            else:  # This is a temporary fix to prevent padding by half the divisor when margin=0.
                pad = [0, 0]

            # This pads at the last dimension of a tensor with 0.
            inputs = F.pad(masked_kspace, pad=pad, value=0)

        return inputs, targets, extra_params


class TrainPreProcessLoggedK:
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
            masked_kspace += torch.randn(masked_kspace.shape, device=self.device) * 0.01
            masked_kspace = log_weighting(masked_kspace)

            extra_params = {'k_scales': k_scale, 'masks': mask}

            # Target must be on the same scale as the inputs for scale invariance of data.
            # kspace_target *= (torch.tensor(1) / k_scale)

            # Recall that the Fourier transform is a linear transform.
            # Performing scaling after ifft for numerical stability
            kspace_target *= k_scaling
            cmg_target = ifft2(kspace_target)
            img_target = complex_abs(cmg_target)
            kspace_target = log_weighting(kspace_target)

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


class TrainPreProcessNormK:
    def __init__(self, mask_func, challenge, device, use_seed=True, divisor=1):
        if challenge not in ('singlecoil', 'multicoil'):
            raise ValueError(f'Challenge should either be "singlecoil" or "multicoil"')
        self.mask_func = mask_func
        self.challenge = challenge
        self.device = device
        self.use_seed = use_seed
        self.divisor = divisor
        self.norm_mask_base = self.get_norm_mask_base().view(-1, 1, 640, 700).to(device=device)

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


            # line = complex_abs(masked_kspace[..., h//2:, w//2, :]).squeeze()
            # line = line.mean(dim=0)
            #
            # x = torch.arange(-w // 2, w // 2, 1, dtype=torch.float32, device=self.device)
            # y = torch.arange(-h // 2, h // 2, 1, dtype=torch.float32, device=self.device)
            # cc = torch.arange(0, c, 1, dtype=torch.float32, device=self.device)
            #
            # ccc, yy, xx = torch.meshgrid(cc, y, x)
            # ccc = ccc.to(dtype=torch.int64)
            #
            # R = torch.floor(torch.sqrt(xx ** 2 + yy ** 2)).to(dtype=torch.int64)
            # R[R >= len(line)] = len(line)-1
            # norm_mask = line[R] + 1E-10
            # norm_mask = torch.stack([norm_mask, norm_mask], dim=-1)

            # n, c, h, w, i = masked_kspace.size()
            # abs_masked_kspace = complex_abs(masked_kspace)[..., (mask > 0.5).squeeze()]
            # maximum = abs_masked_kspace.max(dim=-1, keepdim=True).values.max(dim=-2, keepdim=True).values
            # median = abs_masked_kspace.median(dim=-1, keepdim=True).values.median(dim=-2, keepdim=True).values
            # norm_mask = (maximum - median) * center_crop(self.norm_mask_base, (h, w)) + median
            # norm_mask = norm_mask.unsqueeze(dim=-1)

            norm_mask = 1

            masked_kspace = torch.stack([complex_abs(masked_kspace), complex_phase(masked_kspace)*mask.squeeze(dim=-1)], dim=-1)

            masked_kspace = masked_kspace / norm_mask
            # masked_kspace += (1 - mask) * torch.randn(masked_kspace.shape, device=self.device) * 0.01

            masked_kspace = kspace_to_nchw(masked_kspace)

            extra_params = {'k_scales': k_scaling, 'masks': mask, 'norm_mask': norm_mask}

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

    @staticmethod
    def get_norm_mask_base(p=3):
        w = 700
        h = 640
        x = torch.arange(-w // 2, w // 2, 1, dtype=torch.float32)
        y = torch.arange(-h // 2, h // 2, 1, dtype=torch.float32)

        yy, xx = torch.meshgrid(y, x)

        norm_mask_base = p / (xx ** 2 + yy ** 2 + p)
        return norm_mask_base


class TrainPreProcessPhaseK:
    def __init__(self, mask_func, challenge, device, use_seed=True, divisor=1):
        if challenge not in ('singlecoil', 'multicoil'):
            raise ValueError(f'Challenge should either be "singlecoil" or "multicoil"')
        self.mask_func = mask_func
        self.challenge = challenge
        self.device = device
        self.use_seed = use_seed
        self.divisor = divisor
        self.norm_mask_base = self.get_norm_mask_base().view(-1, 1, 640, 700).to(device=device)

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

            # n, c, h, w, i = masked_kspace.size()
            # abs_masked_kspace = complex_abs(masked_kspace)[..., (mask > 0.5).squeeze()]
            # maximum = abs_masked_kspace.max(dim=-1, keepdim=True).values.max(dim=-2, keepdim=True).values
            # median = abs_masked_kspace.median(dim=-1, keepdim=True).values.median(dim=-2, keepdim=True).values
            # norm_mask = (maximum - median) * center_crop(self.norm_mask_base, (h, w)) + median
            # norm_mask = norm_mask.unsqueeze(dim=-1)

            # masked_kspace = torch.stack([complex_abs(masked_kspace), complex_phase(masked_kspace)*mask.squeeze(dim=-1)], dim=-1)
            masked_kspace = complex_phase(masked_kspace) * mask.squeeze(dim=-1)

            # masked_kspace = masked_kspace / norm_mask
            # masked_kspace += (1 - mask) * torch.randn(masked_kspace.shape, device=self.device) * 0.01

            # masked_kspace = kspace_to_nchw(masked_kspace)

            kspace_target *= k_scaling

            norm_mask = complex_abs(kspace_target)

            extra_params = {'k_scales': k_scaling, 'masks': mask, 'norm_mask': norm_mask}

            # Target must be on the same scale as the inputs for scale invariance of data.
            # kspace_target *= (torch.tensor(1) / k_scale)

            # Recall that the Fourier transform is a linear transform.
            # Performing scaling after ifft for numerical stability


            cmg_target = ifft2(kspace_target)
            img_target = complex_abs(cmg_target)
            kspace_target = complex_phase(kspace_target)

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

    @staticmethod
    def get_norm_mask_base(p=3):
        w = 700
        h = 640
        x = torch.arange(-w // 2, w // 2, 1, dtype=torch.float32)
        y = torch.arange(-h // 2, h // 2, 1, dtype=torch.float32)

        yy, xx = torch.meshgrid(y, x)

        norm_mask_base = p / (xx ** 2 + yy ** 2 + p)
        return norm_mask_base


class TrainPreProcessAbsPhaseK:
    def __init__(self, mask_func, challenge, device, use_seed=True, divisor=1):
        if challenge not in ('singlecoil', 'multicoil'):
            raise ValueError(f'Challenge should either be "singlecoil" or "multicoil"')
        self.mask_func = mask_func
        self.challenge = challenge
        self.device = device
        self.use_seed = use_seed
        self.divisor = divisor
        self.norm_mask_base = self.get_norm_mask_base().view(-1, 1, 640, 700).to(device=device)

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

            n, c, h, w, i = masked_kspace.size()
            abs_masked_kspace = complex_abs(masked_kspace)[..., (mask > 0.5).squeeze()]
            maximum = abs_masked_kspace.max(dim=-1, keepdim=True).values.max(dim=-2, keepdim=True).values
            median = abs_masked_kspace.median(dim=-1, keepdim=True).values.median(dim=-2, keepdim=True).values
            norm_mask = (maximum - median) * center_crop(self.norm_mask_base, (h, w)) + median
            # norm_mask = norm_mask.unsqueeze(dim=-1)

            masked_kspace = torch.stack([complex_abs(masked_kspace) / norm_mask, complex_phase(masked_kspace)*mask.squeeze(dim=-1)], dim=-1)

            # masked_kspace += (1 - mask) * torch.randn(masked_kspace.shape, device=self.device) * 0.01

            masked_kspace = kspace_to_nchw(masked_kspace)

            extra_params = {'k_scales': k_scaling, 'masks': mask, 'norm_mask': norm_mask}

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

    @staticmethod
    def get_norm_mask_base(p=3):
        w = 700
        h = 640
        x = torch.arange(-w // 2, w // 2, 1, dtype=torch.float32)
        y = torch.arange(-h // 2, h // 2, 1, dtype=torch.float32)

        yy, xx = torch.meshgrid(y, x)

        norm_mask_base = p / (xx ** 2 + yy ** 2 + p)
        return norm_mask_base


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
            masked_kspace, mask = apply_mask(kspace_target, self.mask_func, seed)

            k_scale = torch.std(masked_kspace)
            k_scaling = torch.tensor(1) / k_scale

            masked_kspace *= k_scaling  # Multiplication is faster than division.
            masked_kspace = fft2(complex_center_crop(ifft2(masked_kspace), (window_size, window_size)))
            masked_kspace = ifft1(masked_kspace, direction='height')
            masked_kspace = kspace_to_nchw(masked_kspace)

            extra_params = {'k_scales': k_scale, 'masks': mask}

            # Target must be on the same scale as the inputs for scale invariance of data.
            # kspace_target *= (torch.tensor(1) / k_scale)

            # Recall that the Fourier transform is a linear transform.
            # Performing scaling after ifft for numerical stability
            kspace_target *= k_scaling
            kspace_target = fft2(complex_center_crop(ifft2(kspace_target), (window_size, window_size)))
            cmg_target = ifft2(kspace_target)
            img_target = complex_abs(cmg_target)
            kspace_target = ifft1(kspace_target, direction='height')

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
