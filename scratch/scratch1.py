from data.data_transforms import kspace_to_nchw, nchw_to_kspace, to_tensor, k_slice_to_chw
import numpy as np
import torch
from time import time

k1 = np.random.uniform(size=(32, 15, 640, 328))
k2 = np.random.uniform(size=(32, 15, 640, 328))

k = k1 + k2 * 1j

kt = to_tensor(k)

tic = time()
ncwh = kspace_to_nchw(kt)
mid = kt.shape[1]


for idx, kts in enumerate(kt):
    temp = k_slice_to_chw(kts)
    print(idx, torch.eq(ncwh[idx], temp).all())


chan = 17
ri = chan % 2
sli = chan // 2

print(torch.eq(torch.squeeze(ncwh[3, chan, ...]), torch.squeeze(kt[3, sli, ..., ri])).all())
kspace = nchw_to_kspace(ncwh)
toc = time() - tic

print(torch.eq(kt, kspace).all(), toc)
