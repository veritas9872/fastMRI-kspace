import torch
import h5py

from data.data_transforms import ifft2, to_tensor, root_sum_of_squares, center_crop, complex_center_crop, complex_abs

file = '/media/veritas/D/FastMRI/multicoil_val/file1001798.h5'
sdx = 10
with h5py.File(file, mode='r') as hf:
    kspace = hf['kspace'][sdx]
    target = hf['reconstruction_rss'][sdx]

cmg_scale = 2E-5
recon = complex_center_crop(ifft2(to_tensor(kspace) / cmg_scale), shape=(320, 320)) * cmg_scale
recon = root_sum_of_squares(complex_abs(recon))
target = to_tensor(target)

print(torch.allclose(recon, target))
