import torch
from data.data_transforms import ifftshift, fftshift, fft2, ifft2, tensor_to_complex_np

import numpy as np


def fft1(data, direction):
    """
    Apply centered, normalized 1 dimensional Fast Fourier Transform along the height axis.
    Super-inefficient implementation where the Inverse Fourier Transform is applied to the last (width) axis again.
    This is because there is no Pytorch native implementation for controlling FFT axes.
    Also, this is (probably) faster than permuting the tensor repeatedly.

    Args:
        data (torch.Tensor): Complex valued input data containing at least 3 dimensions: dimensions
            -3 & -2 are spatial dimensions and dimension -1 has size 2. All other dimensions are
            assumed to be batch dimensions.
        direction (str): Direction that the FFT is to be performed.
            Not using `dim` or `axis` as keyword to reduce confusion.
            Unfortunately, Pytorch has no complex number data type for fft, so axis dims are different.

    Returns:
        torch.Tensor: The FFT of the input.
    """
    assert data.size(-1) == 2
    assert direction in ('height', 'width'), 'direction must be either height or width.'

    # Push height dimension to last meaningful axis for FFT.
    if direction == 'height':
        data = data.transpose(dim0=-3, dim1=-2)

    data = ifftshift(data, dim=-2)
    data = torch.fft(data, signal_ndim=1, normalized=True)
    data = fftshift(data, dim=-2)

    # Restore height dimension to its original location.
    if direction == 'height':
        data = data.transpose(dim0=-3, dim1=-2)

    return data


def ifft1(data, direction):
    """
    Apply centered, normalized 1 dimensional Inverse Fast Fourier Transform along the height axis.
    Super-inefficient implementation where the Fourier Transform is applied to the last (width) axis again.
    This is because there is no Pytorch native implementation for controlling IFFT axes.
    Also, this is (probably) faster than permuting the tensor repeatedly.

    Args:
        data (torch.Tensor): Complex valued input data containing at least 3 dimensions: dimensions
            -3 & -2 are spatial dimensions and dimension -1 has size 2. All other dimensions are
            assumed to be batch dimensions.
        direction (str): Direction that the IFFT is to be performed.
            Not using `dim` or `axis` as keyword to reduce confusion.
            Unfortunately, Pytorch has no complex number data type for fft, so axis dims are different.
    Returns:
        torch.Tensor: The IFFT of the input.
    """
    assert data.size(-1) == 2
    assert direction in ('height', 'width'), 'direction must be either height or width.'

    if direction == 'height':  # Push height dimension to last meaningful axis for IFFT.
        data = data.transpose(dim0=-3, dim1=-2)

    data = ifftshift(data, dim=-2)
    data = torch.ifft(data, signal_ndim=1, normalized=True)
    data = fftshift(data, dim=-2)

    if direction == 'height':  # Restore height dimension to its original location.
        data = data.transpose(dim0=-3, dim1=-2)

    return data


# Tests
def create_tensor(shape):
    tensor = np.arange(np.product(shape)).reshape(shape)
    tensor = torch.from_numpy(tensor).float()
    return tensor


def test_fft1(shape):
    shape = shape + [2]
    tensor = create_tensor(shape)
    out_torch = fft1(tensor).numpy()
    out_torch = out_torch[..., 0] + 1j * out_torch[..., 1]

    tensor_numpy = tensor_to_complex_np(tensor)
    tensor_numpy = np.fft.ifftshift(tensor_numpy, axes=-2)
    out_numpy = np.fft.fft(tensor_numpy, axis=-2, norm='ortho')
    out_numpy = np.fft.fftshift(out_numpy, axes=-2)

    assert np.allclose(out_torch, out_numpy)


def test_ifft1(shape):
    shape = shape + [2]
    tensor = create_tensor(shape)
    out_torch = ifft1(tensor).numpy()
    out_torch = out_torch[..., 0] + 1j * out_torch[..., 1]

    tensor_numpy = tensor_to_complex_np(tensor)
    tensor_numpy = np.fft.ifftshift(tensor_numpy, axes=-2)
    out_numpy = np.fft.ifft(tensor_numpy, axis=-2, norm='ortho')
    out_numpy = np.fft.fftshift(out_numpy, axes=-2)

    assert np.allclose(out_torch, out_numpy)

