from torch import nn, Tensor

from data.data_transforms import complex_abs, fft2, ifft2, center_crop, root_sum_of_squares, fft1, ifft1


class PostProcessComplex(nn.Module):
    def __init__(self, challenge: str, replace_kspace=False, resolution=320):
        super().__init__()
        if challenge not in ('singlecoil', 'multicoil'):
            raise ValueError(f'Challenge should either be "singlecoil" or "multicoil"')
        self.challenge = challenge
        self.replace_kspace = replace_kspace
        self.resolution = resolution

    def forward(self, cmg_output: Tensor, targets: dict, extra_params: dict):
        assert cmg_output.dim() == 5 and cmg_output.size(1) == 2, 'Invalid shape!'
        if cmg_output.size(0) > 1:
            raise NotImplementedError('Only one at a time for now.')

        kspace_target = targets['kspace_targets']
        cmg_recon = cmg_output.permute(dims=(0, 2, 3, 4, 1))  # Convert back into NCHW2

        if cmg_recon.shape != kspace_target.shape:  # Cropping recon left-right.
            left = (cmg_recon.size(-2) - kspace_target.size(-2)) // 2
            cmg_recon = cmg_recon[..., left:left+kspace_target.size(-2), :]

        assert cmg_recon.shape == kspace_target.shape, 'Reconstruction and target sizes are different.'
        assert (cmg_recon.size(-3) % 2 == 0) and (cmg_recon.size(-2) % 2 == 0), \
            'Not impossible but not expected to have sides with odd lengths.'

        kspace_recon = fft2(cmg_recon)

        if self.replace_kspace:
            mask = extra_params['masks']
            kspace_recon = kspace_target * mask + (1 - mask) * kspace_recon
            cmg_recon = ifft2(kspace_recon)

        img_recon = complex_abs(cmg_recon)

        # recons = {'kspace_recons': kspace_recon, 'cmg_recons': cmg_recon, 'img_recons': img_recon}
        recons = dict()

        if self.challenge == 'multicoil':
            rss_recon = center_crop(img_recon, (self.resolution, self.resolution)) * extra_params['cmg_scales']
            rss_recon = root_sum_of_squares(rss_recon, dim=1).squeeze()
            recons['rss_recons'] = rss_recon

        return recons  # recons are not rescaled except rss_recons.


class PostProcessComplexWSK(nn.Module):
    def __init__(self, challenge: str, replace=False, weighted=True, resolution=320):
        super().__init__()
        if challenge not in ('singlecoil', 'multicoil'):
            raise ValueError(f'Challenge should either be "singlecoil" or "multicoil"')
        self.challenge = challenge
        self.replace = replace
        self.weighted = weighted
        self.resolution = resolution

    def forward(self, semi_kspace_output: Tensor, targets: dict, extra_params: dict):
        assert semi_kspace_output.dim() == 5 and semi_kspace_output.size(1) == 2, 'Invalid shape!'
        if semi_kspace_output.size(0) > 1:
            raise NotImplementedError('Only one at a time for now.')

        semi_kspace_target = targets['semi_kspace_targets']
        semi_kspace_recon = semi_kspace_output.permute(dims=(0, 2, 3, 4, 1))  # Convert back into NCHW2

        if semi_kspace_recon.shape != semi_kspace_target.shape:  # Cropping recon left-right.
            left = (semi_kspace_recon.size(-2) - semi_kspace_target.size(-2)) // 2
            semi_kspace_recon = semi_kspace_recon[..., left:left+semi_kspace_target.size(-2), :]

        assert semi_kspace_recon.shape == semi_kspace_target.shape, 'Reconstruction and target sizes are different.'
        assert (semi_kspace_recon.size(-3) % 2 == 0) and (semi_kspace_recon.size(-2) % 2 == 0), \
            'Not impossible but not expected to have sides with odd lengths.'

        if self.weighted:
            semi_kspace_recon = semi_kspace_recon / extra_params['weightings']

        if self.replace:
            mask = extra_params['masks']
            semi_kspace_recon = semi_kspace_target * mask + (1 - mask) * semi_kspace_recon

        # kspace_recon = fft1(semi_kspace_recon, direction='height')
        cmg_recon = ifft1(semi_kspace_recon, direction='width')
        img_recon = complex_abs(cmg_recon)

        recons = {'semi_kspace_recons': semi_kspace_recon, 'cmg_recons': cmg_recon, 'img_recons': img_recon}

        if self.challenge == 'multicoil':
            rss_recon = center_crop(img_recon, (self.resolution, self.resolution)) * extra_params['sk_scales']
            rss_recon = root_sum_of_squares(rss_recon, dim=1).squeeze()
            recons['rss_recons'] = rss_recon

        return recons  # recons are not rescaled except rss_recons.

