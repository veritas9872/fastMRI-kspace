import torch
from data.data_transforms import apply_info_mask, ifft2, complex_center_crop, fft2, center_crop, complex_abs


class PreProcessComplex:
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
            cmg_target = ifft2(kspace_target)

            if self.crop_center:
                complex_image = complex_center_crop(complex_image, shape=(self.resolution, self.resolution))
                cmg_target = complex_center_crop(cmg_target, shape=(self.resolution, self.resolution))

            cmg_scale = torch.std(complex_image)
            complex_image /= cmg_scale
            cmg_target /= cmg_scale

            extra_params = {'cmg_scales': cmg_scale, 'masks': mask}
            extra_params.update(info)
            extra_params.update(attrs)

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

            kspace_target = fft2(cmg_target)  # Reconstruct k-space target after scaling, cropping, and flipping.

            # The image target is obtained after flipping the complex image.
            # This removes the need to flip the image target.
            img_target = complex_abs(cmg_target)
            img_inputs = complex_abs(complex_image)

            # Use plurals as keys to reduce confusion.
            targets = {'kspace_targets': kspace_target, 'cmg_targets': cmg_target,
                       'img_targets': img_target, 'img_inputs': img_inputs}

            if self.challenge == 'multicoil':
                targets['rss_targets'] = target

            # Converting to N2CHW format for Complex CNN.
            inputs = complex_image.permute(dims=(0, 4, 1, 2, 3))

        return inputs, targets, extra_params
