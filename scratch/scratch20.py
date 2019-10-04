import torch
import h5py
import numpy as np

from data.data_transforms import root_sum_of_squares, center_crop, ifft2, to_tensor, complex_abs


file = '/media/veritas/D/FastMRI/multicoil_val/file1000229.h5'
with h5py.File(file, 'r') as hf:
    kspace = hf['kspace'][()]
    rss = hf['reconstruction_rss'][()]

kspace = to_tensor(kspace)
image = center_crop(root_sum_of_squares(complex_abs(ifft2(kspace)), dim=1), shape=(320, 320)).squeeze().numpy()
print(np.allclose(image, rss))

