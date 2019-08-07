import torch
import torch.nn.functional as F

from data.data_transforms import apply_info_mask, complex_abs, ifft2, center_crop


class PreProcessIMG:
    """
    Pre-processing for image-to-image learning.
    """
    def __init__(self, mask_func, challenge, device, augment_data=False,
                 use_seed=True, center_crop=True, resolution=320, divisor=1):
        assert callable(mask_func), '`mask_func` must be a callable function.'
        if challenge not in ('singlecoil', 'multicoil'):
            raise ValueError(f'Challenge should either be "singlecoil" or "multicoil"')

        self.mask_func = mask_func
        self.challenge = challenge
        self.device = device
        self.augment_data = augment_data
        self.use_seed = use_seed
        self.center_crop = center_crop
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

            if self.center_crop:
                image = center_crop(image, shape=(self.resolution, self.resolution))

            img_scale = torch.std(image)
            image /= img_scale

            extra_params = {'img_scales': img_scale, 'masks': mask}
            extra_params.update(info)
            extra_params.update(attrs)

            img_target = complex_abs(ifft2(kspace_target))
            img_target /= img_scale

            if self.center_crop:
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
