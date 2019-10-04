import torch
from data.data_transforms import ifft2, fft2, complex_abs

image = torch.rand(10, 20, 30, 2)
lr_flip = torch.flip(image, dims=[-2])
ud_flip = torch.flip(image, dims=[-3])
all_flip = torch.flip(image, dims=[-3, -2])

kspace = fft2(image)
lr_kspace = fft2(lr_flip)
ud_kspace = fft2(ud_flip)
all_kspace = fft2(all_flip)

absolute = torch.sum(complex_abs(kspace))
lr_abs = torch.sum(complex_abs(lr_kspace))
ud_abs = torch.sum(complex_abs(ud_kspace))
all_abs = torch.sum(complex_abs(all_kspace))

a = torch.allclose(absolute, lr_abs)
b = torch.allclose(absolute, ud_abs)
c = torch.allclose(absolute, all_abs)

print(a, b, c)


