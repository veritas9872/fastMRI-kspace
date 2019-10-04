import torch

from data.data_transforms import fft2, ifft2


def check_invertible():
    orig = torch.rand(4, 6, 8, 12, 2, dtype=torch.float64) * 1024 - 64
    trans = fft2(orig) * 100
    trans = ifft2(trans) / 100
    print(torch.allclose(orig, trans))


if __name__ == '__main__':
    check_invertible()
