import torch
from data.data_transforms import complex_abs


complex_image = torch.rand(10, 20, 30, 2, dtype=torch.float64)
absolute_image = complex_abs(complex_image)
print(absolute_image.shape)
angle_image = torch.atan2(complex_image[..., 1], complex_image[..., 0])
recon_real = absolute_image * torch.cos(angle_image)
recon_imag = absolute_image * torch.sin(angle_image)
recon_image = torch.stack([recon_real, recon_imag], dim=-1)
print(torch.allclose(complex_image, recon_image))
