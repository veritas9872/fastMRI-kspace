import torch
from torch import nn

from data.data_transforms import nchw_to_kspace, ifft2, complex_abs, \
    fft1, fft2, ifft1, kspace_to_nchw, complex_width_crop


class OutputTransform(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, cmg_outputs, targets, extra_params):

        cmg_recons_toim = nchw_to_kspace(cmg_outputs)
        img_recons = complex_abs(cmg_recons_toim)
        kspace_recons = fft2(cmg_recons_toim)

        recons = {'kspace_recons': kspace_recons, 'cmg_recons': cmg_outputs, 'img_recons': img_recons}

        return recons  # Returning scaled reconstructions.


class OutputTransformCC(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, cmg_outputs, targets, extra_params):

        cmg_recons_toim = nchw_to_kspace(cmg_outputs)
        cmg_wc_kspace = complex_width_crop(cmg_recons_toim, 320)
        cmg_wc = kspace_to_nchw(cmg_wc_kspace)
        img_recons = complex_abs(cmg_recons_toim)
        img_wc = complex_abs(cmg_wc_kspace)

        kspace_recons = fft2(cmg_recons_toim)

        recons = {'kspace_recons': kspace_recons, 'cmg_recons': cmg_outputs, 'img_recons': img_recons,
                  'cmg_wc': cmg_wc, 'img_wc': img_wc}

        return recons  # Returning scaled reconstructions.


class OutputTransformCCRSS(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, cmg_outputs, targets, extra_params):

        cmg_recons_toim = nchw_to_kspace(cmg_outputs)
        cmg_wc_kspace = complex_width_crop(cmg_recons_toim, 320)
        cmg_wc = kspace_to_nchw(cmg_wc_kspace)
        img_recons = complex_abs(cmg_recons_toim)
        img_wc = complex_abs(cmg_wc_kspace)

        kspace_recons = fft2(cmg_recons_toim)

        recons = {'kspace_recons': kspace_recons, 'cmg_recons': cmg_outputs, 'img_recons': img_recons,
                  'cmg_wc': cmg_wc, 'img_wc': img_wc}

        img_recons_rss = (img_recons ** 2).sum(dim=1).sqrt().squeeze()

        return recons, img_recons_rss  # Returning scaled reconstructions.


class OutputReplaceCCRSS(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, cmg_outputs, targets, extra_params):

        cmg_recons_toim = nchw_to_kspace(cmg_outputs)
        kspace_recons = fft2(cmg_recons_toim)

        kspace_recons = kspace_recons * (1 - extra_params['masks']) + targets['kspace_targets'] * extra_params['masks']
        cmg_recons = ifft2(kspace_recons)
        cmg_wc_kspace = complex_width_crop(cmg_recons, 320)
        cmg_wc = kspace_to_nchw(cmg_wc_kspace)
        img_recons = complex_abs(cmg_recons)
        img_wc = complex_abs(cmg_wc_kspace)
        img_recons_rss = (img_recons ** 2).sum(dim=1).sqrt().squeeze()

        recons = {'kspace_recons': kspace_recons, 'cmg_recons': cmg_outputs, 'img_recons': img_recons,
                  'cmg_wc': cmg_wc, 'img_wc': img_wc}

        return recons, img_recons_rss


class OutputTransformWC(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, cmg_outputs, targets, extra_params):
        cmg_recons_toim = nchw_to_kspace(cmg_outputs)
        cmg_wc = complex_width_crop(cmg_recons_toim, 320)
        img_wc = complex_abs(cmg_wc)

        kspace_wc = fft2(cmg_wc)

        recons = {'kspace_recons': kspace_wc, 'cmg_recons': cmg_wc, 'img_recons': img_wc}

        return recons


class OutputTransformWCTest(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, cmg_outputs, extra_params):

        cmg_recons_toim = nchw_to_kspace(cmg_outputs)
        cmg_wc = complex_width_crop(cmg_recons_toim, 320)
        img_wc = complex_abs(cmg_wc)

        k_scale = extra_params['k_scales']
        recons = img_wc / k_scale

        # recons = img_wc

        rss_recons = (recons ** 2).sum(dim=1).sqrt().squeeze()

        return rss_recons


class OutputTransformIK(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, k_outputs, targets, extra_params):


        rs_k_outputs = nchw_to_kspace(k_outputs)
        rs_img_recons = ifft2(rs_k_outputs)
        cmg_recons = kspace_to_nchw(rs_img_recons)
        img_recons = complex_abs(rs_img_recons)

        recons = {'kspace_recons': rs_k_outputs, 'cmg_recons': cmg_recons, 'img_recons': img_recons}

        return recons


class OutputTransformIKCrop(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, k_outputs, targets, extra_params):


        rs_k_outputs = nchw_to_kspace(k_outputs)
        rs_k_outputs_wc = complex_width_crop(rs_k_outputs, 320)
        rs_img_recons = ifft2(rs_k_outputs_wc)
        cmg_recons = kspace_to_nchw(rs_img_recons)
        img_recons = complex_abs(rs_img_recons)

        recons = {'kspace_recons': rs_k_outputs_wc, 'cmg_recons': cmg_recons, 'img_recons': img_recons}

        return recons


class OutputTransformCCTest(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, cmg_outputs, extra_params):

        cmg_recons_toim = nchw_to_kspace(cmg_outputs)
        img_recons = complex_abs(cmg_recons_toim)

        kspace_recons = fft2(cmg_recons_toim)

        recons = {'kspace_recons': kspace_recons, 'cmg_recons': cmg_outputs, 'img_recons': img_recons}
        k_scale = extra_params['k_scales']

        recons = recons['img_recons'] / k_scale
        rss_recons = (recons ** 2).sum(dim=1).sqrt()

        return rss_recons.squeeze()  # Returning scaled reconstructions. Not rescaled.


class OutputTransformIKTest(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, k_outputs, extra_params):

        k_outputs_torch = nchw_to_kspace(k_outputs)
        cmg_recons_toim = ifft2(k_outputs_torch)
        img_recons = complex_abs(cmg_recons_toim)

        k_scale = extra_params['k_scales']

        recons = img_recons / k_scale
        rss_recons = (recons ** 2).sum(dim=1).sqrt()

        return rss_recons.squeeze()



class OutputReplaceTransformK(nn.Module):
    """
    Outputs are expected to be k-space data that need reshaping and conversion to the image domain.
    Inputs and targets are expected to be scaled already.
    Currently, the implementation expects only 1 batch.
    Also, I removed rescaling as the loss needs to be calculated on the standardized values for data scale invariance.
    Final reconstructions can be obtained by dividing by the k_scale value since
    the Fourier transform and its relatives are all linear functions.
    """
    def __init__(self):
        super().__init__()

    def forward(self, kspace_outputs, targets, extra_params, weight_map):

        if kspace_outputs.size(0) > 1:
            raise NotImplementedError('Only one batch at a time for now.')

        # For removing width dimension padding. Recall that k-space form has 2 as last dim size.
        left = (kspace_outputs.size(-1) - targets['kspace_targets'].size(-2)) // 2
        right = left + targets['kspace_targets'].size(-2)

        # Cropping width dimension by pad. # Multiply by scales to restore the original scaling. --> Not rescaling!!!
        kspace_outputs = kspace_outputs[..., left:right]  # * extra_params['k_scales']

        kspace_recons = nchw_to_kspace(kspace_outputs)

        kspace_recons = kspace_recons / weight_map

        assert kspace_recons.shape == targets['kspace_targets'].shape, 'Reconstruction and target sizes are different.'

        kspace_recons = kspace_recons * (1 - extra_params['masks']) + targets['kspace_targets'] * extra_params['masks']

        cmg_recons = ifft2(kspace_recons)

        img_recons = complex_abs(cmg_recons)

        recons = {'kspace_recons': kspace_recons, 'cmg_recons': cmg_recons, 'img_recons': img_recons}

        return recons  # Returning scaled reconstructions. Not rescaled.


class OutputWeightTransformK(nn.Module):
    """
    Processing of output to inverse k2wgt multiplied to k-space input
    """
    def __init__(self):
        super().__init__()

    def forward(self, kspace_outputs, targets, extra_params):

        if kspace_outputs.size(0) > 1:
            raise NotImplementedError('Only one batch at a time for now.')

        kspace_recons = nchw_to_kspace(kspace_outputs)

        kspace_recons = kspace_recons / extra_params['weight_map']

        assert kspace_recons.shape == targets['kspace_targets'].shape, 'Reconstruction and target sizes are different.'

        cmg_recons = ifft2(kspace_recons)

        img_recons = complex_abs(cmg_recons)

        recons = {'kspace_recons': kspace_recons, 'cmg_recons': cmg_recons, 'img_recons': img_recons}

        return recons  # Returning scaled reconstructions. Not rescaled.


class OutputTransformK(nn.Module):
    """
    Processing of output to inverse k2wgt multiplied to k-space input
    """
    def __init__(self):
        super().__init__()

    def forward(self, kspace_outputs, targets, extra_params):

        if kspace_outputs.size(0) > 1:
            raise NotImplementedError('Only one batch at a time for now.')

        kspace_recons = nchw_to_kspace(kspace_outputs)

        assert kspace_recons.shape == targets['kspace_targets'].shape, 'Reconstruction and target sizes are different.'

        cmg_recons = ifft2(kspace_recons)

        img_recons = complex_abs(cmg_recons)

        recons = {'kspace_recons': kspace_recons, 'cmg_recons': cmg_recons, 'img_recons': img_recons}

        return recons  # Returning scaled reconstructions. Not rescaled.


class OutputStackTransformK(nn.Module):
    """
    Processing of output to inverse k2wgt multiplied to k-space input
    """
    def __init__(self):
        super().__init__()

    def forward(self, kspace_outputs, targets, extra_params):

        if kspace_outputs.size(0) > 1:
            raise NotImplementedError('Only one batch at a time for now.')

        kspace_recons = nchw_to_kspace(kspace_outputs)

        assert kspace_recons.shape == targets['kspace_targets'].shape, 'Reconstruction and target sizes are different.'

        cmg_recons = ifft2(kspace_recons)

        img_recons = complex_abs(cmg_recons)

        recons = {'kspace_recons': kspace_recons, 'cmg_recons': cmg_recons, 'img_recons': img_recons}

        return recons  # Returning scaled reconstructions. Not rescaled.


class OutputTransformNormK(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, kspace_outputs, targets, extra_params):

        if kspace_outputs.size(0) > 1:
            raise NotImplementedError('Only one batch at a time for now.')

        kspace_recons = nchw_to_kspace(kspace_outputs)

        assert kspace_recons.shape == targets['kspace_targets'].shape, 'Reconstruction and target sizes are different.'

        cmg_recons = ifft2(kspace_recons * extra_params['norm_mask'])

        img_recons = complex_abs(cmg_recons)

        recons = {'kspace_recons': kspace_recons, 'cmg_recons': cmg_recons, 'img_recons': img_recons}

        return recons  # Returning scaled reconstructions. Not rescaled.


class OutputReplaceTransformCroppedK(nn.Module):
    """
    Outputs are expected to be k-space data that need reshaping and conversion to the image domain.
    Inputs and targets are expected to be scaled already.
    Currently, the implementation expects only 1 batch.
    Also, I removed rescaling as the loss needs to be calculated on the standardized values for data scale invariance.
    Final reconstructions can be obtained by dividing by the k_scale value since
    the Fourier transform and its relatives are all linear functions.
    """
    def __init__(self):
        super().__init__()

    def forward(self, kspace_outputs, targets, extra_params):

        if kspace_outputs.size(0) > 1:
            raise NotImplementedError('Only one batch at a time for now.')

        # For removing width dimension padding. Recall that k-space form has 2 as last dim size.
        left = (kspace_outputs.size(-1) - targets['kspace_targets'].size(-2)) // 2
        right = left + targets['kspace_targets'].size(-2)

        # Cropping width dimension by pad. # Multiply by scales to restore the original scaling. --> Not rescaling!!!
        kspace_outputs = kspace_outputs[..., left:right]  # * extra_params['k_scales']

        kspace_recons = nchw_to_kspace(kspace_outputs)

        assert kspace_recons.shape == targets['kspace_targets'].shape, 'Reconstruction and target sizes are different.'

        # kspace_recons = kspace_recons * (1 - extra_params['masks']) + targets['kspace_targets'] * extra_params['masks']

        cmg_recons = ifft1(kspace_recons, direction='width')

        img_recons = complex_abs(cmg_recons)

        kspace_recons = fft1(kspace_recons, direction='height')

        recons = {'kspace_recons': kspace_recons, 'cmg_recons': cmg_recons, 'img_recons': img_recons}

        return recons  # Returning scaled reconstructions. Not rescaled.


class MidTransformK(nn.Module):
    """
    Outputs are expected to be k-space data that need reshaping and conversion to the image domain.
    Inputs and targets are expected to be scaled already.
    Currently, the implementation expects only 1 batch.
    Also, I removed rescaling as the loss needs to be calculated on the standardized values for data scale invariance.
    Final reconstructions can be obtained by dividing by the k_scale value since
    the Fourier transform and its relatives are all linear functions.
    """
    def __init__(self):
        super().__init__()

    def forward(self, image_outputs, targets, extra_params):


        rs_image_outputs = nchw_to_kspace(image_outputs)
        rs_kspace_inputs = fft2(rs_image_outputs)
        kspace_inputs = kspace_to_nchw(rs_kspace_inputs)

        # kspace_recons = kspace_inputs * (1 - extra_params['masks']) + targets['kspace_targets'] * extra_params['masks']

        mid_outputs = {'kspace_recons': kspace_inputs}

        return mid_outputs  # Returning scaled reconstructions. Not rescaled.


class MidReplaceTransformK(nn.Module):
    """
    Outputs are expected to be k-space data that need reshaping and conversion to the image domain.
    Inputs and targets are expected to be scaled already.
    Currently, the implementation expects only 1 batch.
    Also, I removed rescaling as the loss needs to be calculated on the standardized values for data scale invariance.
    Final reconstructions can be obtained by dividing by the k_scale value since
    the Fourier transform and its relatives are all linear functions.
    """
    def __init__(self):
        super().__init__()

    def forward(self, image_outputs, targets, extra_params):

        rs_image_outputs = nchw_to_kspace(image_outputs)
        rs_kspace_inputs = fft2(rs_image_outputs)
        kspace_inputs = kspace_to_nchw(rs_kspace_inputs)

        assert extra_params['masks'].shape[-1] == targets['kspace_targets'].shape[-1]
        kspace_recons = kspace_inputs * (1 - extra_params['masks']) + targets['kspace_targets'] * extra_params['masks']

        mid_outputs = {'kspace_recons': kspace_recons}

        return mid_outputs  # Returning scaled reconstructions. Not rescaled.