import torch

from skimage.measure import compare_ssim
import numpy as np
import tensorflow as tf

from metrics.new_1d_ssim import SSIM
from metrics.ssim1d import SSIM as SSIM1D

import h5py


def check_same():

    eps = 0.5
    device = 'cpu'

    noise1 = torch.rand(10, 20, 30, 40, device=device)
    noise2 = torch.rand(10, 20, 30, 40, device=device)
    inputs = torch.ones(10, 20, 30, 40, device=device)
    target = torch.ones(10, 20, 30, 40, device=device) - eps * noise2

    my_ssim_func = SSIM(filter_size=7, max_val=eps * 10).to(device=device)
    orig_1d_ssim_func = SSIM1D(win_size=7, data_range=eps * 10, channel=20).to(device=device)

    inputs_cpu = inputs.view(-1, 30, 40).permute(1, 2, 0).cpu().numpy()
    target_cpu = target.view(-1, 30, 40).permute(1, 2, 0).cpu().numpy()
    other_ssim = compare_ssim(inputs_cpu, target_cpu, win_size=7, multichannel=True, sigma=1.5,
                              gaussian_weights=True, data_range=eps * 10, use_sample_covariance=False)

    assert callable(my_ssim_func) and callable(orig_1d_ssim_func)
    my_ssim_val = my_ssim_func(inputs, target)
    ssim_1d_val = orig_1d_ssim_func(inputs, target)

    print(my_ssim_val.item(), ssim_1d_val.item(), other_ssim)


def compare_images():
    tf.enable_eager_execution()
    file1 = '/media/veritas/D/FastMRI/multicoil_val/file1000263.h5'
    file2 = '/media/veritas/D/FastMRI/multicoil_val/file1000925.h5'

    eps = 0.000435262

    my_ssim_func = SSIM(filter_size=7, max_val=eps)
    orig_ssim_func = SSIM1D(win_size=7, data_range=eps, channel=1)

    with h5py.File(file1, 'r') as hf1, h5py.File(file2, 'r') as hf2:
        block1 = torch.from_numpy(hf1['reconstruction_rss'][0:30]) - torch.rand(30, 320, 320) * eps * 0.1
        block2 = torch.from_numpy(hf2['reconstruction_rss'][0:30])

    # print((block2.max() - block2.min()).item())
    my_ssim_val = my_ssim_func(block1, block2)
    ssim_1d_val = orig_ssim_func(block1.unsqueeze(dim=1), block2.unsqueeze(dim=1))

    new_block1 = block1.transpose(0, 2).numpy()
    new_block2 = block2.transpose(0, 2).numpy()

    tf_block1 = tf.convert_to_tensor(new_block1)
    tf_block2 = tf.convert_to_tensor(new_block2)

    tf_ssim = tf.image.ssim(tf_block1, tf_block2, max_val=eps, filter_size=7)

    other_ssim = compare_ssim(new_block1, new_block2, win_size=7, multichannel=True, sigma=1.5,
                              gaussian_weights=True, data_range=eps, use_sample_covariance=False)

    print(my_ssim_val.item(), ssim_1d_val.item(), other_ssim, float(tf_ssim))


if __name__ == '__main__':
    compare_images()
    # Conclusion, My implemen
