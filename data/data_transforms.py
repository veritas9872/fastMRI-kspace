import torch
import numpy as np


def to_tensor(data):
    """
    Convert numpy array to PyTorch tensor. For complex arrays, the real and imaginary parts
    are stacked along the last dimension.
    Args:
        data (np.array): Input numpy array
    Returns:
        torch.Tensor: PyTorch version of data
    """
    if np.iscomplexobj(data):
        data = np.stack((data.real, data.imag), axis=-1)
    return torch.from_numpy(data)


def apply_mask(data, mask_func, seed=None):
    """
    Subsample given k-space by multiplying with a mask.
    Args:
        data (torch.Tensor): The input k-space data. This should have at least 3 dimensions, where
            dimensions -3 and -2 are the spatial dimensions, and the final dimension has size
            2 (for complex values).
        mask_func (callable): A function that takes a shape (tuple of ints) and a random
            number seed and returns a mask.
        seed (int or 1-d array_like, optional): Seed for the random number generator.
    Returns:
        (tuple): tuple containing:
            masked data (torch.Tensor): Subsampled k-space data
            mask (torch.Tensor): The generated mask
    """
    shape = np.array(data.shape)
    shape[:-3] = 1
    mask = mask_func(shape, seed)
    return torch.where(mask == 0, torch.Tensor([0]), data), mask


def apply_info_mask(data, mask_func, seed=None):
    """
    Subsample given k-space by multiplying with a mask.
    Args:
        data (torch.Tensor): The input k-space data. This should have at least 3 dimensions, where
            dimensions -3 and -2 are the spatial dimensions, and the final dimension has size
            2 (for complex values).
        mask_func (callable): A function that takes a shape (tuple of ints) and a random
            number seed and returns a mask and a dictionary containing information about the masking.
        seed (int or 1-d array_like, optional): Seed for the random number generator.
    Returns:
        (tuple): tuple containing:
            masked data (torch.Tensor): Sub-sampled k-space data
            mask (torch.Tensor): The generated mask
            info (dict): A dictionary containing information about the mask.
    """
    shape = np.array(data.shape)
    shape[:-3] = 1
    mask, info = mask_func(shape, seed)
    mask = mask.to(data.device)
    # Checked that this version also removes negative 0 values as well.
    return torch.where(mask == 0, torch.tensor(0, dtype=data.dtype, device=data.device), data), mask, info


def fft2(data):
    """
    Apply centered 2 dimensional Fast Fourier Transform.
    Args:
        data (torch.Tensor): Complex valued input data containing at least 3 dimensions: dimensions
            -3 & -2 are spatial dimensions and dimension -1 has size 2. All other dimensions are
            assumed to be batch dimensions.
    Returns:
        torch.Tensor: The FFT of the input.
    """
    assert data.size(-1) == 2
    data = ifftshift(data, dim=(-3, -2))
    data = torch.fft(data, 2, normalized=True)
    data = fftshift(data, dim=(-3, -2))
    return data


def ifft2(data):
    """
    Apply centered 2-dimensional Inverse Fast Fourier Transform.
    Args:
        data (torch.Tensor): Complex valued input data containing at least 3 dimensions: dimensions
            -3 & -2 are spatial dimensions and dimension -1 has size 2. All other dimensions are
            assumed to be batch dimensions.
    Returns:
        torch.Tensor: The IFFT of the input.
    """
    assert data.size(-1) == 2
    data = ifftshift(data, dim=(-3, -2))
    data = torch.ifft(data, 2, normalized=True)
    data = fftshift(data, dim=(-3, -2))
    return data


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


def complex_abs(data):
    """
    Compute the absolute value of a complex valued input tensor.
    Args:
        data (torch.Tensor): A complex valued tensor, where the size of the final dimension
            should be 2.
    Returns:
        torch.Tensor: Absolute value of data
    """
    assert data.size(-1) == 2
    return (data ** 2).sum(dim=-1).sqrt()


def root_sum_of_squares(data, dim=0):
    """
    Compute the Root Sum of Squares (RSS) transform along a given dimension of a tensor.
    Args:
        data (torch.Tensor): The input tensor
        dim (int): The dimensions along which to apply the RSS transform
    Returns:
        torch.Tensor: The RSS value
    """
    return torch.sqrt((data ** 2).sum(dim))


def center_crop(data, shape):
    """
    Apply a center crop to the input real image or batch of real images.
    Args:
        data (torch.Tensor): The input tensor to be center cropped. It should have at
            least 2 dimensions and the cropping is applied along the last two dimensions.
        shape (int, int): The output shape. The shape should be smaller than the
            corresponding dimensions of data.
    Returns:
        torch.Tensor: The center cropped image
    """
    assert 0 < shape[0] <= data.shape[-2]
    assert 0 < shape[1] <= data.shape[-1]
    w_from = (data.shape[-2] - shape[0]) // 2
    h_from = (data.shape[-1] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]
    return data[..., w_from:w_to, h_from:h_to]


def complex_center_crop(data, shape):
    """
    Apply a center crop to the input image or batch of complex images.
    Args:
        data (torch.Tensor): The complex input tensor to be center cropped. It should
            have at least 3 dimensions and the cropping is applied along dimensions
            -3 and -2 and the last dimensions should have a size of 2.
        shape (int, int): The output shape. The shape should be smaller than the
            corresponding dimensions of data.
    Returns:
        torch.Tensor: The center cropped image
    """
    assert 0 < shape[0] <= data.shape[-3]
    assert 0 < shape[1] <= data.shape[-2]
    w_from = (data.shape[-3] - shape[0]) // 2
    h_from = (data.shape[-2] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]
    return data[..., w_from:w_to, h_from:h_to, :]


def normalize(data, mean, stddev, eps=0.):
    """
    Normalize the given tensor using:
        (data - mean) / (stddev + eps)
    Args:
        data (torch.Tensor): Input data to be normalized
        mean (float): Mean value
        stddev (float): Standard deviation
        eps (float): Added to stddev to prevent dividing by zero
    Returns:
        torch.Tensor: Normalized tensor
    """
    return (data - mean) / (stddev + eps)


def normalize_instance(data, eps=0.):
    """
        Normalize the given tensor using:
            (data - mean) / (stddev + eps)
        where mean and stddev are computed from the data itself.
        Args:
            data (torch.Tensor): Input data to be normalized
            eps (float): Added to stddev to prevent dividing by zero
        Returns:
            torch.Tensor: Normalized tensor
        """
    mean = data.mean()
    std = data.std()
    return normalize(data, mean, std, eps), mean, std


# Helper functions

def roll(x, shift, dim):
    """
    Similar to np.roll but applies to PyTorch Tensors
    """
    if isinstance(shift, (tuple, list)):
        assert len(shift) == len(dim)
        for s, d in zip(shift, dim):
            x = roll(x, s, d)
        return x
    shift = shift % x.size(dim)
    if shift == 0:
        return x
    left = x.narrow(dim, 0, x.size(dim) - shift)
    right = x.narrow(dim, x.size(dim) - shift, shift)
    return torch.cat((right, left), dim=dim)


def fftshift(x, dim=None):
    """
    Similar to np.fft.fftshift but applies to PyTorch Tensors
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [dim // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = x.shape[dim] // 2
    else:
        shift = [x.shape[i] // 2 for i in dim]
    return roll(x, shift, dim)


def ifftshift(x, dim=None):
    """
    Similar to np.fft.ifftshift but applies to PyTorch Tensors
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [(dim + 1) // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = (x.shape[dim] + 1) // 2
    else:
        shift = [(x.shape[i] + 1) // 2 for i in dim]
    return roll(x, shift, dim)


def tensor_to_complex_np(data):
    """
    Converts a complex torch tensor to numpy array.
    Args:
        data (torch.Tensor): Input data to be converted to numpy.
    Returns:
        np.array: Complex numpy version of data
    """
    data = data.numpy()
    return data[..., 0] + 1j * data[..., 1]


# My k-space transforms
def k_slice_to_chw(tensor):
    """
    Convert torch tensor in (Coil, Height, Width, Complex) 4D k-slice format to
    (C, H, W) 3D format for processing by 2D CNNs.

    `Complex` indicates (real, imag) as 2 channels, the complex data format for Pytorch.

    C is the coils interleaved with real and imaginary values as separate channels.
    C is therefore always 2 * Coil.

    Singlecoil data is assumed to be in the 4D format with Coil = 1

    Args:
        tensor (torch.Tensor): Input data in 4D k-slice tensor format.
    Returns:
        tensor (torch.Tensor): tensor in 3D CHW format to be fed into a CNN.
    """
    assert isinstance(tensor, torch.Tensor)
    assert tensor.dim() == 4
    s = tensor.shape
    assert s[-1] == 2
    tensor = tensor.permute(dims=(0, 3, 1, 2)).reshape(shape=(2 * s[0], s[1], s[2]))
    return tensor


def chw_to_k_slice(tensor):
    """
    Convert a torch tensor in (C, H, W) format to the (Coil, Height, Width, Complex) format.

    This assumes that the real and imaginary values of a coil are always adjacent to one another in C.
    """
    assert isinstance(tensor, torch.Tensor)
    assert tensor.dim() == 3
    s = tensor.shape
    assert s[0] % 2 == 0
    tensor = tensor.view(size=(s[0] // 2, 2, s[1], s[2])).permute(dims=(0, 2, 3, 1))
    return tensor


def kspace_to_nchw(tensor):
    """
    Convert torch tensor in (Slice, Coil, Height, Width, Complex) 5D format to
    (N, C, H, W) 4D format for processing by 2D CNNs.

    Complex indicates (real, imag) as 2 channels, the complex data format for Pytorch.

    C is the coils interleaved with real and imaginary values as separate channels.
    C is therefore always 2 * Coil.

    Singlecoil data is assumed to be in the 5D format with Coil = 1

    Args:
        tensor (torch.Tensor): Input data in 5D kspace tensor format.
    Returns:
        tensor (torch.Tensor): tensor in 4D NCHW format to be fed into a CNN.
    """
    assert isinstance(tensor, torch.Tensor)
    assert tensor.dim() == 5
    s = tensor.shape
    assert (s[-1] == 2) or (s[-1] == 3)
    tensor = tensor.permute(dims=(0, 1, 4, 2, 3)).reshape(shape=(s[0], s[-1] * s[1], s[2], s[3]))
    return tensor


def nchw_to_kspace(tensor):
    """
    Convert a torch tensor in (N, C, H, W) format to the (Slice, Coil, Height, Width, Complex) format.

    This function assumes that the real and imaginary values of a coil are always adjacent to one another in C.
    """
    assert isinstance(tensor, torch.Tensor)
    assert tensor.dim() == 4
    s = tensor.shape
    assert s[1] % 2 == 0
    tensor = tensor.view(size=(s[0], s[1] // 2, 2, s[2], s[3])).permute(dims=(0, 1, 3, 4, 2))
    return tensor


def log_weighting(tensor, scale=1):
    assert scale > 0, '`scale` must be a positive value.'
    return scale * torch.sign(tensor) * torch.log1p(torch.abs(tensor))


def exp_weighting(tensor, scale=1):
    assert scale > 0, '`scale` must be a positive value.'
    return torch.sign(tensor) * torch.expm1(torch.abs(tensor) * (1 / scale))
