import pytest

import torch

from metrics.new_1d_ssim import SSIM, MSSSIM


def test_ssim_symmetry():
    inputs = torch.rand(3, 3, 256, 256)
    target = torch.rand(3, 3, 256, 256)
    loss1 = SSIM(reduction='none', max_val=1.)(inputs, target)
    loss2 = SSIM(reduction='none', max_val=1.)(target, inputs)
    assert torch.all(torch.eq(loss1, loss2))


def test_ms_ssim_symmetry():
    inputs = torch.rand(3, 3, 256, 256)
    target = torch.rand(3, 3, 256, 256)
    loss1 = MSSSIM(reduction='none', max_val=1.)(inputs, target)
    loss2 = MSSSIM(reduction='none', max_val=1.)(target, inputs)
    assert torch.all(torch.eq(loss1, loss2))


def test_ssim_equality():
    inputs = torch.rand(3, 3, 256, 256)
    target = inputs
    loss = SSIM(reduction='none', max_val=1.)(inputs, target)
    assert torch.all(torch.eq(loss, 1))


def test_ms_ssim_equality():
    inputs = torch.rand(3, 3, 256, 256)
    target = inputs
    loss = MSSSIM(reduction='none', max_val=1.)(inputs, target)
    assert torch.all(torch.eq(loss, 1))


@pytest.mark.parametrize(
    'scale', [2, 4, 7]
)
def test_ssim_scale_invariance(scale):
    inputs = torch.rand(3, 3, 256, 256)
    target = torch.rand(3, 3, 256, 256)
    loss1 = SSIM(reduction='none', max_val=1.)(inputs, target)
    loss2 = SSIM(reduction='none', max_val=scale)(inputs * scale, target * scale)
    assert torch.allclose(loss1, loss2)


@pytest.mark.parametrize(
    'scale', [2, 4, 7]
)
def test_ms_ssim_scale_invariance(scale):
    inputs = torch.rand(3, 3, 256, 256)
    target = torch.rand(3, 3, 256, 256)
    loss1 = MSSSIM(reduction='none', max_val=1.)(inputs, target)
    loss2 = MSSSIM(reduction='none', max_val=scale)(inputs * scale, target * scale)
    assert torch.allclose(loss1, loss2)


if __name__ == '__main__':
    pass
