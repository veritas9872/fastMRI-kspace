import torch

from data.data_transforms import apply_info_mask, complex_abs, ifft2, center_crop, root_sum_of_squares


class PreProcessRSS:
    """
    Rather hack filled implementation of input transform for RSS inputs.
    """
    def __init__(self, mask_func, challenge, device, augment_data=False, use_seed=True, resolution=320, fat_info=False):
        assert callable(mask_func), '`mask_func` must be a callable function.'
        assert challenge == 'multicoil', 'Challenge must be multicoil for this.'

        self.mask_func = mask_func
        self.challenge = challenge
        self.device = device
        self.augment_data = augment_data
        self.use_seed = use_seed
        self.resolution = resolution  # Only has effect when center_crop is True.
        self.fat_info = fat_info

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

            input_image = complex_abs(ifft2(masked_kspace))
            # Cropping is mandatory if RSS means the 320 version. Not so if a larger image is used.
            # However, I think that removing target processing is worth the loss of flexibility.
            input_image = center_crop(input_image, shape=(self.resolution, self.resolution))
            img_scale = torch.std(input_image)
            input_image /= img_scale

            extra_params = {'img_scales': img_scale, 'masks': mask}
            extra_params.update(info)
            extra_params.update(attrs)

            if target is 0:
                target = center_crop(root_sum_of_squares(complex_abs(ifft2(kspace_target)), dim=1),
                                     shape=(self.resolution, self.resolution)).squeeze()

            # Data augmentation by flipping images up-down and left-right.
            if self.augment_data:
                flip_lr = torch.rand(()) < 0.5
                flip_ud = torch.rand(()) < 0.5

                if flip_lr and flip_ud:
                    input_image = torch.flip(input_image, dims=(-2, -1))
                    target = torch.flip(target, dims=(-2, -1))

                elif flip_ud:
                    input_image = torch.flip(input_image, dims=(-2,))
                    target = torch.flip(target, dims=(-2,))

                elif flip_lr:
                    input_image = torch.flip(input_image, dims=(-1,))
                    target = torch.flip(target, dims=(-1,))

            # Use plurals as keys to reduce confusion.
            input_rss = root_sum_of_squares(input_image, dim=1).squeeze()
            targets = {'img_inputs': input_image, 'rss_targets': target, 'rss_inputs': input_rss}

            if self.fat_info:
                fat_supp = extra_params['acquisition'] == 'CORPDFS_FBK'  # Fat suppressed acquisition.
                batch, _, height, width = input_image.shape
                fat_info = torch.ones(size=(batch, 1, height, width), device=self.device) * fat_supp
                input_image = torch.cat([input_image, fat_info], dim=1)

        return input_image, targets, extra_params
