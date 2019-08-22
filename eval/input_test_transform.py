import torch
import torch.nn.functional as F

from data.data_transforms import complex_abs, ifft2, center_crop, apply_info_mask


class PreProcessValIMG:
    def __init__(self, mask_func, challenge, device, crop_center=True, resolution=320, divisor=1):
        assert callable(mask_func), '`mask_func` must be a callable function.'
        if challenge not in ('singlecoil', 'multicoil'):
            raise ValueError(f'Challenge should either be "singlecoil" or "multicoil"')

        self.mask_func = mask_func
        self.challenge = challenge
        self.device = device
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
            seed = tuple(map(ord, file_name))
            masked_kspace, mask, info = apply_info_mask(kspace_target, self.mask_func, seed)

            image = complex_abs(ifft2(masked_kspace))
            if self.crop_center:
                image = center_crop(image, shape=(self.resolution, self.resolution))
            else:  # Super-hackish temporary line.  # TODO: Fix this thing later!
                image = center_crop(image, shape=(352, image.size(-1)))

            margin = image.size(-1) % self.divisor
            if margin > 0:
                pad = [(self.divisor - margin) // 2, (1 + self.divisor - margin) // 2]
            else:  # This is a fix to prevent padding by half the divisor when margin=0.
                pad = [0, 0]

            # This pads at the last dimension of a tensor with 0.
            image = F.pad(image, pad=pad, value=0)

            img_scale = torch.std(center_crop(image, shape=(self.resolution, self.resolution)))  # Also a hack!
            image /= img_scale

            extra_params = {'img_scales': img_scale, 'masks': mask}
            extra_params.update(info)
            extra_params.update(attrs)

            # Use plurals as keys to reduce confusion.
            targets = {'img_inputs': image}

        return image, targets, extra_params


class PreProcessTestIMG:
    """
    Pre-processing for image-to-image learning.
    """
    def __init__(self, challenge, device, crop_center=True, resolution=320):

        if challenge not in ('singlecoil', 'multicoil'):
            raise ValueError(f'Challenge should either be "singlecoil" or "multicoil"')

        self.challenge = challenge
        self.device = device
        self.crop_center = crop_center
        self.resolution = resolution  # Only has effect when center_crop is True.
        # self.divisor = divisor

    def __call__(self, masked_kspace, target, attrs, file_name, slice_num):
        assert isinstance(masked_kspace, torch.Tensor), 'k-space target was expected to be a Pytorch Tensor.'
        if masked_kspace.dim() == 3:  # If the collate function does not expand dimensions for single-coil.
            masked_kspace = masked_kspace.expand(1, 1, -1, -1, -1)
        elif masked_kspace.dim() == 4:  # If the collate function does not expand dimensions for multi-coil.
            masked_kspace = masked_kspace.expand(1, -1, -1, -1, -1)
        elif masked_kspace.dim() != 5:  # Expanded k-space should have 5 dimensions.
            raise RuntimeError('k-space target has invalid shape!')

        if masked_kspace.size(0) != 1:
            raise NotImplementedError('Batch size should be 1 for now.')

        with torch.no_grad():
            image = complex_abs(ifft2(masked_kspace))

            if self.crop_center:
                image = center_crop(image, shape=(self.resolution, self.resolution))

            img_scale = torch.std(image)
            image /= img_scale

            extra_params = {'img_scales': img_scale}
            extra_params.update(attrs)

            # Use plurals as keys to reduce confusion.
            targets = {'img_inputs': image}

            # margin = image.size(-1) % self.divisor
            # if margin > 0:
            #     pad = [(self.divisor - margin) // 2, (1 + self.divisor - margin) // 2]
            # else:  # This is a fix to prevent padding by half the divisor when margin=0.
            #     pad = [0, 0]
            #
            # # This pads at the last dimension of a tensor with 0.
            # inputs = F.pad(image, pad=pad, value=0)

        return image, targets, extra_params
