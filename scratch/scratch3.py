import torch
import numpy as np
from data.data_transforms import ifft2, fft2


a = torch.rand(20, 40, 60, 92, 2, device='cuda:1')
b = ifft2(a * 1E8)
c = fft2(b) * 1E-8

# print(torch.all(a == c))
# print(torch.allclose(a, c, rtol=0.01))
eps = np.finfo(np.float64).eps

print(torch.max(c / (a + eps)))
print(torch.min(c / (a + eps)))
print(torch.mean(c / (a + eps)).cpu().numpy())

# print(torch.sum(a != c) / (20 * 40 * 60 * 92 * 2))

