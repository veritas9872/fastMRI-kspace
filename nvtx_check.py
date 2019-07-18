import torch
from torch.cuda import nvtx
from torch.autograd.profiler import emit_nvtx, load_nvprof, profile


def main():
    nvtx.range_push('Test')
    a = torch.ones(10, 20, 30, device='cuda')
    b = a.to('cpu', non_blocking=True)
    print('Hello World!')
    nvtx.range_pop()
    print('Finished!')


if __name__ == '__main__':
    with profile(use_cuda=True) as prof:
        main()
    print(prof)
    # load_nvprof('check.qdstem.qdrep')

