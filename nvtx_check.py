import torch
from torch.cuda import nvtx


def main():
    nvtx.range_push('Test')
    a = torch.ones(10, 20, 30, device='cuda')
    b = a.to('cpu', non_blocking=True)
    print('Hello World!')
    nvtx.range_pop()
    print('Finished!')


if __name__ == '__main__':
    main()
