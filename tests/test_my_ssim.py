import torch
from metrics.my_ssim import SSIMLoss, MultiScaleSSIMLoss


def test_ssim_loss_always_le_one(self):
    target = torch.rand(3, 3, 256, 256)
    input = torch.rand(3, 3, 256, 256)
    loss = SSIMLoss(reduction='none')(input, target, max_val=1.)
    m = loss > 1
    self.assertEqual(m.sum(), 0)


def test_ms_ssim_loss_always_le_one(self):
    target = torch.rand(3, 3, 256, 256)
    input = torch.rand(3, 3, 256, 256)
    loss = MultiScaleSSIMLoss(reduction='none')(input, target, max_val=1.)
    m = loss > 1
    self.assertEqual(m.sum(), 0)


def test_ssim_symmetry(self):
    target = torch.rand(3, 3, 256, 256)
    input = torch.rand(3, 3, 256, 256)
    l1 = SSIMLoss(reduction='none')(input, target, max_val=1.)
    l2 = SSIMLoss(reduction='none')(target, input, max_val=1.)
    self.assertEqual(l1, l2)


def test_ms_ssim_symmetry(self):
    target = torch.rand(3, 3, 256, 256)
    input = torch.rand(3, 3, 256, 256)
    l1 = MultiScaleSSIMLoss(reduction='none')(input, target, max_val=1.)
    l2 = MultiScaleSSIMLoss(reduction='none')(target, input, max_val=1.)
    self.assertEqual(l1, l2)


def test_ssim_equality(self):
    target = torch.rand(3, 3, 256, 256)
    input = target
    loss = SSIMLoss(reduction='none')(input, target, max_val=1.)
    loss = loss - 1.
    self.assertEqual(loss.sum(), 0)


def test_ms_ssim_equality(self):
    target = torch.rand(3, 3, 256, 256)
    input = target
    loss = MultiScaleSSIMLoss(reduction='none')(input, target, max_val=1.)
    loss = loss - 1.
    self.assertEqual(loss.sum(), 0)


def test_ssim_raises_if_target_and_input_are_different_size(self):
    target = torch.rand(5, 3, 128, 128)
    input = torch.rand(3, 2, 64, 64)
    with self.assertRaises(ValueError):
        SSIMLoss()(input, target, max_val=1.)


def test_ms_ssim_raises_if_target_and_input_are_different_size(self):
    target = torch.rand(5, 3, 128, 128)
    input = torch.rand(3, 2, 64, 64)
    with self.assertRaises(ValueError):
        MultiScaleSSIMLoss()(input, target, max_val=1.)


if __name__ == '__main__':
    pass
