import torch

import math

from data.data_transforms import apply_info_mask, ifft2, complex_center_crop, fft2, complex_abs, kspace_to_nchw


class PreProcessXNet:
    def __init__(self, mask_func, challenge, device, augment_data=False,
                 use_seed=True, crop_center=True, resolution=320):
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

            # Complex image made from down-sampled k-space.
            complex_image = ifft2(masked_kspace)

            if self.crop_center:
                complex_image = complex_center_crop(complex_image, shape=(self.resolution, self.resolution))

            # Recall that the Fourier transform is a linear transform.
            cmg_target = ifft2(kspace_target)

            if self.crop_center:
                cmg_target = complex_center_crop(cmg_target, shape=(self.resolution, self.resolution))

            # Data augmentation by flipping images up-down and left-right.
            if self.augment_data:
                flip_lr = torch.rand(()) < 0.5
                flip_ud = torch.rand(()) < 0.5

                if flip_lr and flip_ud:
                    # Last dim is real/complex dimension for complex image and target.
                    complex_image = torch.flip(complex_image, dims=(-3, -2))
                    cmg_target = torch.flip(cmg_target, dims=(-3, -2))
                    target = torch.flip(target, dims=(-2, -1))  # Has only two dimensions, height and width.

                elif flip_ud:
                    complex_image = torch.flip(complex_image, dims=(-3,))
                    cmg_target = torch.flip(cmg_target, dims=(-3,))
                    target = torch.flip(target, dims=(-2,))

                elif flip_lr:
                    complex_image = torch.flip(complex_image, dims=(-2,))
                    cmg_target = torch.flip(cmg_target, dims=(-2,))
                    target = torch.flip(target, dims=(-1,))

            img_input = complex_abs(complex_image)
            img_scale = torch.std(img_input)
            img_input /= img_scale

            # Adding pi to angles so that the phase is in the [0, 2pi] range for better learning.
            phase_input = torch.atan2(complex_image[..., 1], complex_image[..., 0])
            phase_input += math.pi  # Don't forget to remove the pi in the output transform!

            cmg_target /= img_scale
            img_target = complex_abs(cmg_target)
            kspace_target = fft2(cmg_target)  # Reconstruct k-space target after cropping and image augmentation.
            phase_target = torch.atan2(cmg_target[..., 1], cmg_target[..., 0])

            extra_params = {'img_scales': img_scale, 'masks': mask}
            extra_params.update(info)
            extra_params.update(attrs)

            # Use plurals as keys to reduce confusion.
            targets = {'kspace_targets': kspace_target, 'cmg_targets': cmg_target,
                       'img_targets': img_target, 'phase_targets': phase_target, 'img_inputs': img_input}

            if self.challenge == 'multicoil':
                targets['rss_targets'] = target

            # Converting to NCHW format for CNN. Also adding phase input.
            inputs = (img_input, phase_input)

        return inputs, targets, extra_params
