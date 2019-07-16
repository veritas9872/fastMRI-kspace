import numpy as np
import pytest
import torch

from train.subsample import MaskFunc, UniformMaskFunc
from data import data_transforms


def create_tensor(shape):
    tensor = np.arange(np.product(shape)).reshape(shape)
    tensor = torch.from_numpy(tensor).float()
    return tensor


@pytest.mark.parametrize('shape, center_fractions, accelerations', [
    ([4, 32, 32, 2], [0.08], [4]),
    ([2, 64, 64, 2], [0.04, 0.08], [8, 4]),
])
def test_apply_mask(shape, center_fractions, accelerations):
    mask_func = MaskFunc(center_fractions, accelerations)
    expected_mask = mask_func(shape, seed=123)
    tensor = create_tensor(shape)
    output, mask = data_transforms.apply_mask(tensor, mask_func, seed=123)
    assert output.shape == tensor.shape
    assert mask.shape == expected_mask.shape
    assert np.all(expected_mask.numpy() == mask.numpy())
    assert np.all((output * mask).numpy() == output.numpy())


@pytest.mark.parametrize('shape, center_fractions, accelerations', [
    ([4, 32, 32, 2], [0.08], [4]),
    ([2, 64, 64, 2], [0.04, 0.08], [8, 4]),
])
def test_apply_uniform_mask(shape, center_fractions, accelerations):
    mask_func = UniformMaskFunc(center_fractions, accelerations)
    expected_mask, expected_info = mask_func(shape, seed=123)
    tensor = create_tensor(shape)
    output, mask, info = data_transforms.apply_info_mask(tensor, mask_func, seed=123)
    assert isinstance(info, dict)
    assert output.shape == tensor.shape
    assert mask.shape == expected_mask.shape
    assert np.all(expected_mask.numpy() == mask.numpy())
    assert np.all((output * mask).numpy() == output.numpy())
    assert expected_info == info


@pytest.mark.parametrize('shape', [
    [3, 3],
    [4, 6],
    [10, 8, 4],
])
def test_fft2(shape):
    shape = shape + [2]
    tensor = create_tensor(shape)
    out_torch = data_transforms.fft2(tensor).numpy()
    out_torch = out_torch[..., 0] + 1j * out_torch[..., 1]

    tensor_numpy = data_transforms.tensor_to_complex_np(tensor)
    tensor_numpy = np.fft.ifftshift(tensor_numpy, (-2, -1))
    out_numpy = np.fft.fft2(tensor_numpy, norm='ortho')
    out_numpy = np.fft.fftshift(out_numpy, (-2, -1))
    assert np.allclose(out_torch, out_numpy)


@pytest.mark.parametrize('shape', [
    [3, 3],
    [4, 6],
    [10, 8, 4],
])
def test_ifft2(shape):
    shape = shape + [2]
    tensor = create_tensor(shape)
    out_torch = data_transforms.ifft2(tensor).numpy()
    out_torch = out_torch[..., 0] + 1j * out_torch[..., 1]

    tensor_numpy = data_transforms.tensor_to_complex_np(tensor)
    tensor_numpy = np.fft.ifftshift(tensor_numpy, (-2, -1))
    out_numpy = np.fft.ifft2(tensor_numpy, norm='ortho')
    out_numpy = np.fft.fftshift(out_numpy, (-2, -1))
    assert np.allclose(out_torch, out_numpy)


@pytest.mark.parametrize('shape', [
    [3, 3],
    [4, 6],
    [10, 8, 4],
])
def test_fft1_height(shape):
    shape = shape + [2]
    tensor = create_tensor(shape)
    out_torch = data_transforms.fft1(tensor, direction='height').numpy()
    out_torch = out_torch[..., 0] + 1j * out_torch[..., 1]

    tensor_numpy = data_transforms.tensor_to_complex_np(tensor)
    tensor_numpy = np.fft.ifftshift(tensor_numpy, axes=-2)
    out_numpy = np.fft.fft(tensor_numpy, axis=-2, norm='ortho')
    out_numpy = np.fft.fftshift(out_numpy, axes=-2)

    assert np.allclose(out_torch, out_numpy)


@pytest.mark.parametrize('shape', [
    [3, 3],
    [4, 6],
    [10, 8, 4],
])
def test_fft1_width(shape):
    shape = shape + [2]
    tensor = create_tensor(shape)
    out_torch = data_transforms.fft1(tensor, direction='width').numpy()
    out_torch = out_torch[..., 0] + 1j * out_torch[..., 1]

    tensor_numpy = data_transforms.tensor_to_complex_np(tensor)
    tensor_numpy = np.fft.ifftshift(tensor_numpy, axes=-1)
    out_numpy = np.fft.fft(tensor_numpy, axis=-1, norm='ortho')
    out_numpy = np.fft.fftshift(out_numpy, axes=-1)

    assert np.allclose(out_torch, out_numpy)


@pytest.mark.parametrize('shape', [
    [3, 3],
    [4, 6],
    [10, 8, 4],
])
def test_ifft1_height(shape):
    shape = shape + [2]
    tensor = create_tensor(shape)
    out_torch = data_transforms.ifft1(tensor, direction='height').numpy()
    out_torch = out_torch[..., 0] + 1j * out_torch[..., 1]

    tensor_numpy = data_transforms.tensor_to_complex_np(tensor)
    tensor_numpy = np.fft.ifftshift(tensor_numpy, axes=-2)
    out_numpy = np.fft.ifft(tensor_numpy, axis=-2, norm='ortho')
    out_numpy = np.fft.fftshift(out_numpy, axes=-2)

    assert np.allclose(out_torch, out_numpy)


@pytest.mark.parametrize('shape', [
    [3, 3],
    [4, 6],
    [10, 8, 4],
])
def test_ifft1_width(shape):
    shape = shape + [2]
    tensor = create_tensor(shape)
    out_torch = data_transforms.ifft1(tensor, direction='width').numpy()
    out_torch = out_torch[..., 0] + 1j * out_torch[..., 1]

    tensor_numpy = data_transforms.tensor_to_complex_np(tensor)
    tensor_numpy = np.fft.ifftshift(tensor_numpy, axes=-1)
    out_numpy = np.fft.ifft(tensor_numpy, axis=-1, norm='ortho')
    out_numpy = np.fft.fftshift(out_numpy, axes=-1)

    assert np.allclose(out_torch, out_numpy)


@pytest.mark.parametrize('shape', [
    [3, 3],
    [4, 6],
    [10, 8, 4],
])
def test_complex_abs(shape):
    shape = shape + [2]
    tensor = create_tensor(shape)
    out_torch = data_transforms.complex_abs(tensor).numpy()
    tensor_numpy = data_transforms.tensor_to_complex_np(tensor)
    out_numpy = np.abs(tensor_numpy)
    assert np.allclose(out_torch, out_numpy)


@pytest.mark.parametrize('shape, dim', [
    [[3, 3], 0],
    [[4, 6], 1],
    [[10, 8, 4], 2],
])
def test_root_sum_of_squares(shape, dim):
    tensor = create_tensor(shape)
    out_torch = data_transforms.root_sum_of_squares(tensor, dim).numpy()
    out_numpy = np.sqrt(np.sum(tensor.numpy() ** 2, dim))
    assert np.allclose(out_torch, out_numpy)


@pytest.mark.parametrize('shape, target_shape', [
    [[10, 10], [4, 4]],
    [[4, 6], [2, 4]],
    [[8, 4], [4, 4]],
])
def test_center_crop(shape, target_shape):
    tensor = create_tensor(shape)
    out_torch = data_transforms.center_crop(tensor, target_shape).numpy()
    assert list(out_torch.shape) == target_shape


@pytest.mark.parametrize('shape, target_shape', [
    [[10, 10], [4, 4]],
    [[4, 6], [2, 4]],
    [[8, 4], [4, 4]],
])
def test_complex_center_crop(shape, target_shape):
    shape = shape + [2]
    tensor = create_tensor(shape)
    out_torch = data_transforms.complex_center_crop(tensor, target_shape).numpy()
    assert list(out_torch.shape) == target_shape + [2, ]


@pytest.mark.parametrize('shape, mean, stddev', [
    [[10, 10], 0, 1],
    [[4, 6], 4, 10],
    [[8, 4], 2, 3],
])
def test_normalize(shape, mean, stddev):
    tensor = create_tensor(shape)
    output = data_transforms.normalize(tensor, mean, stddev).numpy()
    assert np.isclose(output.mean(), (tensor.numpy().mean() - mean) / stddev)
    assert np.isclose(output.std(), tensor.numpy().std() / stddev)


@pytest.mark.parametrize('shape', [
    [10, 10],
    [20, 40, 30],
])
def test_normalize_instance(shape):
    tensor = create_tensor(shape)
    output, mean, stddev = data_transforms.normalize_instance(tensor)
    output = output.numpy()
    assert np.isclose(tensor.numpy().mean(), mean, rtol=1e-2)
    assert np.isclose(tensor.numpy().std(), stddev, rtol=1e-2)
    assert np.isclose(output.mean(), 0, rtol=1e-2, atol=1e-3)
    assert np.isclose(output.std(), 1, rtol=1e-2, atol=1e-3)


@pytest.mark.parametrize('shift, dim', [
    (0, 0),
    (1, 0),
    (-1, 0),
    (100, 0),
    ((1, 2), (1, 2)),
])
@pytest.mark.parametrize('shape', [
    [5, 6, 2],
    [3, 4, 5],
])
def test_roll(shift, dim, shape):
    tensor = np.arange(np.product(shape)).reshape(shape)
    out_torch = data_transforms.roll(torch.from_numpy(tensor), shift, dim).numpy()
    out_numpy = np.roll(tensor, shift, dim)
    assert np.allclose(out_torch, out_numpy)


@pytest.mark.parametrize('shape', [
    [5, 3],
    [2, 4, 6],
])
def test_fftshift(shape):
    tensor = np.arange(np.product(shape)).reshape(shape)
    out_torch = data_transforms.fftshift(torch.from_numpy(tensor)).numpy()
    out_numpy = np.fft.fftshift(tensor)
    assert np.allclose(out_torch, out_numpy)


@pytest.mark.parametrize('shape', [
    [5, 3],
    [2, 4, 5],
    [2, 7, 5],
])
def test_ifftshift(shape):
    tensor = np.arange(np.product(shape)).reshape(shape)
    out_torch = data_transforms.ifftshift(torch.from_numpy(tensor)).numpy()
    out_numpy = np.fft.ifftshift(tensor)
    assert np.allclose(out_torch, out_numpy)


@pytest.mark.parametrize('shape', [
    [1, 640, 368, 2],
    [15, 640, 384, 2]
])
@pytest.mark.parametrize('scale', [1, 3.7, 0.12])
def test_reversibility(shape, scale):
    tensor = torch.rand(shape) * 20 - 10
    new = data_transforms.exp_weighting(data_transforms.log_weighting(tensor, scale), scale)
    assert torch.allclose(tensor, new)


if __name__ == '__main__':
    pass  # Run all tests
