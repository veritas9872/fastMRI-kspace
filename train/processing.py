import torch.nn as nn
from data.data_transforms import nchw_to_kspace, complex_abs, ifft2, chw_to_k_slice

# Maybe change file name later.


class SingleBatchOutputTransform(nn.Module):
    """
    This class is expected to be paired with KInputSliceTransform in data/pre_processing.py as its input transform.
    Hence, the disregard for params_dict.
    """
    def __init__(self):
        super().__init__()

    def forward(self, k_output, targets, scales):
        """
            Output post-processing for output k-space tensor with batch size of 1.
            This is experimental and is subject to change.
            Planning on taking k-space outputs from CNNs, then transforming them into original k-space shapes.
            No batch size planned yet.

            Args:
                k_output (torch.Tensor): CNN output of k-space. Expected to have batch-size of 1.
                targets (torch.Tensor): Target image domain data.
                scales (torch.Tensor): scaling factor used to divide the input k-space slice data.
            Returns:
                kspace (torch.Tensor): kspace in original shape with batch dimension in-place.
        """

        # For removing width dimension padding. Recall that k-space form has 2 as last dim size.
        left = (k_output.size(-1) - targets.size(-1)) // 2  # This depends on mini-batch size being 1 to work.
        right = left + targets.size(-1)

        # Previously, cropping was done by  [pad:-pad]. However, this fails catastrophically when pad == 0 as
        # all values are wiped out in those cases where [0:0] creates an empty tensor.

        # Cropping width dimension by pad. Multiply by scales to restore the original scaling.
        k_output = k_output[..., left:right] * scales

        # Processing to k-space form. This is where the batch_size == 1 is important.
        kspace_recons = nchw_to_kspace(k_output)

        # Convert to image.
        image_recons = complex_abs(ifft2(kspace_recons))

        assert image_recons.size() == targets.size(), 'Reconstruction and target sizes do not match.'

        return image_recons, kspace_recons


class OutputBatchTransform(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, k_output, targets, scales):
        """

        Args:
            k_output (torch.Tensor):
            targets (list):
            scales (list):

        Returns:
            image_recons (list):
            kspace_recons (list):

        """
        assert k_output.size(0) == len(targets) == len(scales)
        image_recons = list()
        kspace_recons = list()

        for k_slice, target, scaling in zip(k_output, targets, scales):

            left = (k_slice.size(-1) - target.size(-1)) // 2
            right = left + target.size(-1)

            k_slice_recon = chw_to_k_slice(k_slice[..., left:right] * scaling)
            i_slice_recon = complex_abs(ifft2(k_slice_recon))

            assert i_slice_recon.shape == target.shape
            image_recons.append(i_slice_recon)
            kspace_recons.append(k_slice_recon)

        return image_recons, kspace_recons


class OutputBatchMaskTransform(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, k_output, k_input, targets, extra_params):
        """

        Args:
            k_output (torch.Tensor):
            k_input (torch.Tensor):
            targets (list):
            extra_params (list):

        Returns:
            image_recons (list):
            kspace_recons (list):

        """
        assert k_output.shape == k_input.shape
        assert k_output.size(0) == len(targets) == len(extra_params)
        image_recons = list()
        kspace_recons = list()

        for k_output_slice, k_input_slice, target, extra_param in zip(k_output, k_input, targets, extra_params):

            left = (k_output_slice.size(-1) - target.size(-1)) // 2
            right = left + target.size(-1)

            k_slice_recon = chw_to_k_slice(k_output_slice[..., left:right] * extra_param['scaling']) * (1 - extra_param['mask']) + extra_param['masked_kspace']
            i_slice_recon = complex_abs(ifft2(k_slice_recon))

            assert i_slice_recon.shape == target.shape
            image_recons.append(i_slice_recon)
            kspace_recons.append(k_slice_recon)

        return image_recons, kspace_recons




