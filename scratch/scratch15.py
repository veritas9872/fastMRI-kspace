import torch

z1 = torch.zeros(1)
z2 = torch.zeros(1)
# print(z1 / z2)
a = torch.atan2(z1, z2)
n = torch.atan(z1 / z2)
print(a.item())
print(n.item())
