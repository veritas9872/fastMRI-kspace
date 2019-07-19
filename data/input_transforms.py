import torch
import torch.nn.functional as F

import numpy as np

from data.data_transforms import to_tensor, ifft2, complex_abs, apply_info_mask, kspace_to_nchw, ifft1


# class InputTransformK:
#     def __init__(self, mask_func, challenge, device, use_seed=True, divisor=1):
#
#         if challenge not in ('singlecoil', 'multicoil'):
#             raise ValueError(f'Challenge should either be "singlecoil" or "multicoil"')
#         self.mask_func = mask_func
#         self.challenge = challenge
#         self.device = device
#         self.use_seed = use_seed
#         self.divisor = divisor
#
#     def __call__(self, k_slice, target, attrs, file_name, slice_num):
#         assert np.iscomplexobj(k_slice), 'kspace must be complex.'
#         assert k_slice.shape[-1] % 2 == 0, 'k-space data width must be even.'
#
#         if k_slice.ndim == 2:  # For singlecoil. Makes data processing later on much easier.
#             k_slice = np.expand_dims(k_slice, axis=0)
#         elif k_slice.ndim != 3:  # Prevents possible errors.
#             raise TypeError('Invalid slice type')
#
#         with torch.no_grad():
#             kspace_target = to_tensor(k_slice).to(device=self.device).unsqueeze(dim=0)
#             cmg_target = ifft2(kspace_target)
#             img_target = complex_abs(cmg_target)
#
#             # Use plurals to reduce confusion.
#             targets = {'kspace_targets': kspace_target,
#                        'cmg_targets': cmg_target,
#                        'img_targets': img_target}
#
#             # Apply mask
#             seed = None if not self.use_seed else tuple(map(ord, file_name))
#             masked_kspace, mask = apply_mask(kspace_target, self.mask_func, seed)
#
#             k_scale = torch.std(masked_kspace)
#
#             extra_params = {'k_scales': k_scale, 'masks': mask}
#
#             masked_kspace /= k_scale
#             masked_kspace = kspace_to_nchw(masked_kspace)  # Assumes single batch output.
#
#             margin = masked_kspace.size(-1) % self.divisor
#             if margin > 0:
#                 pad = [(self.divisor - margin) // 2, (1 + self.divisor - margin) // 2]
#             else:  # This is a temporary fix to prevent padding by half the divisor when margin=0.
#                 pad = [0, 0]
#
#             # This pads at the last dimension of a tensor with 0.
#             inputs = F.pad(masked_kspace, pad=pad, value=0)
#
#         return inputs, targets, extra_params


# class TrainPreProcessK:
#     def __init__(self, mask_func, challenge, device, use_seed=True, divisor=1):
#         if challenge not in ('singlecoil', 'multicoil'):
#             raise ValueError(f'Challenge should either be "singlecoil" or "multicoil"')
#         self.mask_func = mask_func
#         self.challenge = challenge
#         self.device = device
#         self.use_seed = use_seed
#         self.divisor = divisor
#
#     def __call__(self, kspace_target, target, attrs, file_name, slice_num):
#         assert isinstance(kspace_target, torch.Tensor)
#         if kspace_target.dim() == 4:  # If the collate function does not expand dimensions.
#             kspace_target = kspace_target.unsqueeze(dim=0)
#         elif kspace_target.dim() != 5:  # Expanded k-space should have 5 dimensions.
#             raise RuntimeError('k-space target has invalid shape!')
#
#         if kspace_target.size(0) != 1:
#             raise NotImplementedError('Batch size should be 1 for now.')
#
#         with torch.no_grad():
#             # Apply mask
#             seed = None if not self.use_seed else tuple(map(ord, file_name))
#             masked_kspace, mask = apply_mask(kspace_target, self.mask_func, seed)
#
#             k_scale = torch.std(masked_kspace)
#             k_scaling = 1 / k_scale
#
#             masked_kspace *= k_scaling  # Multiplication is faster than division.
#             masked_kspace = kspace_to_nchw(masked_kspace)
#
#             extra_params = {'k_scales': k_scale, 'masks': mask}
#
#             # Recall that the Fourier transform is a linear transform.
#             # Performing scaling after ifft for numerical stability
#             cmg_target = ifft2(kspace_target) * k_scaling
#             img_target = complex_abs(cmg_target)
#             kspace_target *= k_scaling
#
#             # Use plurals as keys to reduce confusion.
#             targets = {'kspace_targets': kspace_target, 'cmg_targets': cmg_target, 'img_targets': img_target}
#
#             margin = masked_kspace.size(-1) % self.divisor
#             if margin > 0:
#                 pad = [(self.divisor - margin) // 2, (1 + self.divisor - margin) // 2]
#             else:  # This is a temporary fix to prevent padding by half the divisor when margin=0.
#                 pad = [0, 0]
#
#             # This pads at the last dimension of a tensor with 0.
#             inputs = F.pad(masked_kspace, pad=pad, value=0)
#
#         return inputs, targets, extra_params


# class PreProcessSemiK(nn.Module):
#     def __init__(self, mask_func, challenge, device, use_seed=True, divisor=1, direction='height'):
#         super().__init__()
#         if challenge not in ('singlecoil', 'multicoil'):
#             raise ValueError(f'Challenge should either be "singlecoil" or "multicoil"')
#
#         if direction not in ('height', 'width'):
#             raise ValueError('`direction` should either be `height` or `width')
#
#         self.mask_func = mask_func
#         self.challenge = challenge
#         self.device = device
#         self.use_seed = use_seed
#         self.divisor = divisor
#         self.direction = direction
#
#     def forward(self, kspace_target, target, attrs, file_name, slice_num):
#         assert isinstance(kspace_target, torch.Tensor)
#         if kspace_target.dim() == 4:  # If the collate function does not expand dimensions.
#             kspace_target = kspace_target.unsqueeze(dim=0)
#         elif kspace_target.dim() != 5:  # Expanded k-space should have 5 dimensions.
#             raise RuntimeError('k-space target has invalid shape!')
#
#         if kspace_target.size(0) != 1:
#             raise NotImplementedError('Batch size should be 1 for now.')
#
#         with torch.no_grad():
#             # Apply mask
#             seed = None if not self.use_seed else tuple(map(ord, file_name))
#             masked_kspace, mask = apply_mask(kspace_target, self.mask_func, seed)
#
#             masked_semi_kspace = ifft1(masked_kspace, direction=self.direction)
#             sks_scale = torch.std(masked_semi_kspace)  # SKS: Semi-kspace.
#             sks_scaling = 1 / sks_scale
#
#             masked_semi_kspace = kspace_to_nchw(masked_semi_kspace * sks_scaling)
#
#             extra_params = {'sks_scales': sks_scale, 'masks': mask}
#
#             cmg_target = ifft2(kspace_target) * sks_scaling
#             img_target = complex_abs(cmg_target)
#             sks_target = ifft1(kspace_target, direction=self.direction) * sks_scaling
#
#             # Use plurals as keys to reduce confusion.
#             targets = {'sks_targets': sks_target, 'kspace_targets': kspace_target,
#                        'cmg_targets': cmg_target, 'img_targets': img_target}
#
#             margin = masked_semi_kspace.size(-1) % self.divisor
#             if margin > 0:
#                 pad = [(self.divisor - margin) // 2, (1 + self.divisor - margin) // 2]
#             else:  # This prevents padding by half the divisor when margin=0.
#                 pad = [0, 0]
#
#             # This pads at the last dimension of a tensor with 0.
#             inputs = F.pad(masked_semi_kspace, pad=pad, value=0)
#
#         return inputs, targets, extra_params

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


class PreProcessWK:
    """
    Class for pre-processing weighted k-space.
    However, weighting is optional since a simple function that returns its input can be used to have no weighting.
    """
    def __init__(self, mask_func, weight_func, challenge, device, use_seed=True, divisor=1):
        assert callable(mask_func), '`mask_func` must be a callable function.'
        assert callable(weight_func), '`weight_func` must be a callable function.'
        if challenge not in ('singlecoil', 'multicoil'):
            raise ValueError(f'Challenge should either be "singlecoil" or "multicoil"')

        self.mask_func = mask_func
        self.weight_func = weight_func
        self.challenge = challenge
        self.device = device
        self.use_seed = use_seed
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

            weighting = self.weight_func(masked_kspace)
            masked_kspace *= weighting

            # img_input is not actually an input but what the input would look like in the image domain.
            img_input = complex_abs(ifft2(masked_kspace))

            # The slope is meaningless as the results always become the same after standardization no matter the slope.
            # The ordering could be changed to allow a difference, but this would make the inputs non-standardized.
            k_scale = torch.std(masked_kspace)
            k_scaling = 1 / k_scale

            masked_kspace *= k_scaling  # Multiplication is faster than division.
            masked_kspace = kspace_to_nchw(masked_kspace)

            extra_params = {'k_scales': k_scale, 'masks': mask, 'weightings': weighting}
            extra_params.update(info)
            extra_params.update(attrs)

            # Recall that the Fourier transform is a linear transform.
            kspace_target *= k_scaling
            cmg_target = ifft2(kspace_target)
            img_target = complex_abs(cmg_target)

            # Use plurals as keys to reduce confusion.
            targets = {'kspace_targets': kspace_target, 'cmg_targets': cmg_target,
                       'img_targets': img_target, 'img_inputs': img_input}

            margin = masked_kspace.size(-1) % self.divisor
            if margin > 0:
                pad = [(self.divisor - margin) // 2, (1 + self.divisor - margin) // 2]
            else:  # This is a temporary fix to prevent padding by half the divisor when margin=0.
                pad = [0, 0]

            # This pads at the last dimension of a tensor with 0.
            inputs = F.pad(masked_kspace, pad=pad, value=0)

        return inputs, targets, extra_params


class WeightedPreProcessK:
    def __init__(self, mask_func, challenge, device, use_seed=True, divisor=1, weight_type=False):
        if challenge not in ('singlecoil', 'multicoil'):
            raise ValueError(f'Challenge should either be "singlecoil" or "multicoil"')
        self.mask_func = mask_func
        self.challenge = challenge
        self.device = device
        self.use_seed = use_seed
        self.divisor = divisor
        self.weight_type = weight_type

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

            weighting = self.make_weighting_matrix(masked_kspace)
            masked_kspace *= weighting

            # img_input is not actually an input but what the input would look like in the image domain.
            img_input = complex_abs(ifft2(masked_kspace))

            # The slope is meaningless as the results always become the same after standardization no matter the slope.
            # The ordering could be changed to allow a difference, but this would make the inputs non-standardized.
            k_scale = torch.std(masked_kspace)
            k_scaling = 1 / k_scale

            masked_kspace *= k_scaling  # Multiplication is faster than division.
            masked_kspace = kspace_to_nchw(masked_kspace)

            extra_params = {'k_scales': k_scale, 'masks': mask, 'weightings': weighting}
            extra_params.update(info)
            extra_params.update(attrs)

            # Recall that the Fourier transform is a linear transform.
            kspace_target *= k_scaling
            cmg_target = ifft2(kspace_target)
            img_target = complex_abs(cmg_target)

            # Use plurals as keys to reduce confusion.
            targets = {'kspace_targets': kspace_target, 'cmg_targets': cmg_target,
                       'img_targets': img_target, 'img_inputs': img_input}

            margin = masked_kspace.size(-1) % self.divisor
            if margin > 0:
                pad = [(self.divisor - margin) // 2, (1 + self.divisor - margin) // 2]
            else:  # This is a temporary fix to prevent padding by half the divisor when margin=0.
                pad = [0, 0]

            # This pads at the last dimension of a tensor with 0.
            inputs = F.pad(masked_kspace, pad=pad, value=0)

        return inputs, targets, extra_params

    def make_weighting_matrix(self, tensor):
        assert isinstance(tensor, torch.Tensor), '`tensor` must be a tensor.'
        assert tensor.dim() == 5, '`tensor` is expected to be in the k-space format.'
        device = tensor.device
        height = tensor.size(-3)
        width = tensor.size(-2)
        assert (height % 2 == 0) and (width % 2 == 0), 'Not absolutely necessary but odd sizes are unexpected.'
        mid_height = height / 2
        mid_width = width / 2

        # The indexing might be a bit confusing.
        x_coords = torch.arange(start=-mid_width + 0.5, end=mid_width + 0.5, step=1,
                                device=device).view(1, width).expand(height, width)

        y_coords = torch.arange(start=-mid_height + 0.5, end=mid_height + 0.5, step=1,
                                device=device).view(height, 1).expand(height, width)

        if self.weight_type == 'distance':
            weighting_matrix = torch.sqrt((x_coords ** 2) + (y_coords ** 2))
        elif self.weight_type == 'squared_distance':  # Bad option. Do not use.
            weighting_matrix = (x_coords ** 2) + (y_coords ** 2)
        # elif self.weight_type == 'exponential_distance':  # Actually the exponent minus one.  # Total failure.
        #     distance = torch.sqrt((x_coords ** 2) + (y_coords ** 2))
        #     weighting_matrix = torch.expm1(distance)
        else:
            raise NotImplementedError('Unknown weighting type')

        weighting_matrix = weighting_matrix.view(1, 1, height, width, 1)

        return weighting_matrix


class PreProcessWSK:
    """
    Class for pre-processing weighted semi-k-space.
    """
    def __init__(self, mask_func, weight_func, challenge, device, use_seed=True, divisor=1):
        assert callable(mask_func), '`mask_func` must be a callable function.'
        assert callable(weight_func), '`weight_func` must be a callable function.'
        if challenge not in ('singlecoil', 'multicoil'):
            raise ValueError(f'Challenge should either be "singlecoil" or "multicoil"')

        self.mask_func = mask_func
        self.weight_func = weight_func
        self.challenge = challenge
        self.device = device
        self.use_seed = use_seed
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

            # img_input is not actually an input but what the input would look like in the image domain.
            img_input = complex_abs(ifft2(masked_kspace))

            semi_kspace = ifft1(masked_kspace, direction='height')

            weighting = self.weight_func(masked_kspace)
            masked_kspace *= weighting

            # The slope is meaningless as the results always become the same after standardization no matter the slope.
            # The ordering could be changed to allow a difference, but this would make the inputs non-standardized.
            k_scale = torch.std(semi_kspace)
            k_scaling = 1 / k_scale

            semi_kspace *= k_scaling  # Multiplication is faster than division.
            semi_kspace = kspace_to_nchw(semi_kspace)

            extra_params = {'k_scales': k_scale, 'masks': mask, 'weightings': weighting}
            extra_params.update(info)
            extra_params.update(attrs)

            # Recall that the Fourier transform is a linear transform.
            kspace_target *= k_scaling
            cmg_target = ifft2(kspace_target)
            img_target = complex_abs(cmg_target)
            semi_kspace_target = ifft1(kspace_target, direction='height')

            # Use plurals as keys to reduce confusion.
            targets = {'semi_kspace_targets': semi_kspace_target, 'kspace_targets': kspace_target,
                       'cmg_targets': cmg_target, 'img_targets': img_target, 'img_inputs': img_input}

            margin = semi_kspace.size(-1) % self.divisor
            if margin > 0:
                pad = [(self.divisor - margin) // 2, (1 + self.divisor - margin) // 2]
            else:  # This is a temporary fix to prevent padding by half the divisor when margin=0.
                pad = [0, 0]

            # This pads at the last dimension of a tensor with 0.
            inputs = F.pad(semi_kspace, pad=pad, value=0)

        return inputs, targets, extra_params


class WeightedPreProcessSemiK:
    def __init__(self, mask_func, challenge, device, use_seed=True, divisor=1, weight_type=False):
        if challenge not in ('singlecoil', 'multicoil'):
            raise ValueError(f'Challenge should either be "singlecoil" or "multicoil"')
        self.mask_func = mask_func
        self.challenge = challenge
        self.device = device
        self.use_seed = use_seed
        self.divisor = divisor
        self.weight_type = weight_type

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

            # img_input is not actually an input but what the input would look like in the image domain.
            img_input = complex_abs(ifft2(masked_kspace))

            semi_kspace = ifft1(masked_kspace, direction='height')
            weighting = self.make_semi_weighting_matrix(semi_kspace)
            semi_kspace *= weighting

            # The slope is meaningless as the results always become the same after standardization no matter the slope.
            # The ordering could be changed to allow a difference, but this would make the inputs non-standardized.
            k_scale = torch.std(semi_kspace)
            k_scaling = 1 / k_scale

            semi_kspace *= k_scaling  # Multiplication is faster than division.
            semi_kspace = kspace_to_nchw(semi_kspace)

            extra_params = {'k_scales': k_scale, 'masks': mask, 'weightings': weighting}
            extra_params.update(info)
            extra_params.update(attrs)

            # Recall that the Fourier transform is a linear transform.
            kspace_target *= k_scaling
            cmg_target = ifft2(kspace_target)
            img_target = complex_abs(cmg_target)
            semi_kspace_target = ifft1(kspace_target, direction='height')

            # Use plurals as keys to reduce confusion.
            targets = {'semi_kspace_targets': semi_kspace_target, 'kspace_targets': kspace_target,
                       'cmg_targets': cmg_target, 'img_targets': img_target, 'img_inputs': img_input}

            margin = semi_kspace.size(-1) % self.divisor
            if margin > 0:
                pad = [(self.divisor - margin) // 2, (1 + self.divisor - margin) // 2]
            else:  # This is a temporary fix to prevent padding by half the divisor when margin=0.
                pad = [0, 0]

            # This pads at the last dimension of a tensor with 0.
            inputs = F.pad(semi_kspace, pad=pad, value=0)

        return inputs, targets, extra_params

    def make_semi_weighting_matrix(self, tensor):  # Expects up-down IFFT to turn into image domain.
        assert isinstance(tensor, torch.Tensor), '`tensor` must be a tensor.'
        assert tensor.dim() == 5, '`tensor` is expected to be in the k-space format.'
        device = tensor.device
        width = tensor.size(-2)
        assert width % 2 == 0, 'Not absolutely necessary but odd sizes are unexpected.'
        mid_width = width / 2

        # The indexing might be a bit confusing.
        x_coords = torch.arange(start=-mid_width + 0.5, end=mid_width + 0.5, step=1, device=device)
        if self.weight_type == 'distance':
            weighting_matrix = torch.abs(x_coords).view(1, 1, 1, width, 1)
        elif self.weight_type == 'squared_distance':  # Terrible option. Do not use.
            weighting_matrix = (x_coords ** 2).view(1, 1, 1, width, 1)
        # elif self.weight_type == 'exponential_distance':  # Actually the exponent minus one.  # Total failure.
        #     distance = torch.abs(x_coords).view(1, 1, 1, width, 1)
        #     weighting_matrix = torch.expm1(distance)
        else:
            raise NotImplementedError('Unknown weighting type')

        return weighting_matrix
