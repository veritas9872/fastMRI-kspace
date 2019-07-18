import torch
from torch import nn
from time import perf_counter


class Net(nn.Module):
    def __init__(self, pool_type):
        super().__init__()
        self.layer = nn.Conv2d(in_channels=100, out_channels=10, kernel_size=3)
        if pool_type == 'max':
            self.pool = nn.AdaptiveMaxPool2d(1)
        elif pool_type == 'avg':
            self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, tensor):
        output = self.layer(tensor)
        output = self.pool(output)
        return output


def time_check():
    torch.autograd.set_grad_enabled(True)

    device = torch.device('cuda:0')
    max_model = Net(pool_type='max').to(device=device, non_blocking=True)
    avg_model = Net(pool_type='avg').to(device=device, non_blocking=True)

    dummy1 = torch.rand(10, 100, 64, 64, device=device)
    dummy2 = torch.rand(10, 100, 64, 64, device=device)

    max_model(dummy1)

    tic1 = perf_counter()
    for _ in range(100):
        max_model(dummy1)
    toc1 = perf_counter() - tic1

    avg_model(dummy2)

    tic2 = perf_counter()
    for _ in range(100):
        avg_model(dummy2)
    toc2 = perf_counter() - tic2

    print(toc1, toc2)
    print(dummy1.max(), dummy2.max())


if __name__ == '__main__':
    time_check()
