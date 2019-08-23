from torch import nn, Tensor

from data.data_transforms import complex_abs, fft2, center_crop, root_sum_of_squares


class PostProcessComplex(nn.Module):
    def __init__(self, challenge, resolution=320):
        super().__init__()
        if challenge not in ('singlecoil', 'multicoil'):
            raise ValueError(f'Challenge should either be "singlecoil" or "multicoil"')
        self.challenge = challenge
        self.resolution = resolution

    def forward(self, cmg_output: Tensor, targets, extra_params):
        assert cmg_output.dim() == 5 and cmg_output.size(1) == 2, 'Invalid shape!'
        if cmg_output.size(0) > 1:
            raise NotImplementedError('Only one at a time for now.')

        cmg_target = targets['cmg_targets']
        cmg_recon = cmg_output.permute(dims=(0, 2, 3, 4, 1))  # Convert back into NCHW2

        if cmg_recon.shape != cmg_target.shape:  # Cropping recon left-right.
            left = (cmg_recon.size(-2) - cmg_target.size(-2)) // 2
            cmg_recon = cmg_recon[..., left:left+cmg_target.size(-2), :]

        assert cmg_recon.shape == cmg_target.shape, 'Reconstruction and target sizes are different.'
        assert (cmg_recon.size(-3) % 2 == 0) and (cmg_recon.size(-2) % 2 == 0), \
            'Not impossible but not expected to have sides with odd lengths.'

        kspace_recon = fft2(cmg_recon)
        img_recon = complex_abs(cmg_recon)

        recons = {'kspace_recons': kspace_recon, 'cmg_recons': cmg_recon, 'img_recons': img_recon}

        if self.challenge == 'multicoil':
            rss_recon = center_crop(img_recon, (self.resolution, self.resolution)) * extra_params['cmg_scales']
            rss_recon = root_sum_of_squares(rss_recon, dim=1).squeeze()
            recons['rss_recons'] = rss_recon

        return recons  # recons are not rescaled except rss_recons.
