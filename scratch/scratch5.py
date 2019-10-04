from h5py import File
from pathlib import Path
import numpy as np


data_root = '/media/veritas/F/compFastMRI/multicoil_val'

data_path = Path(data_root)

for path in sorted(data_path.glob('*.h5'), reverse=True):
    with File(name=path, mode='r') as file:
        print(str(path))
        for idx in range(len(file['kspace'])):
            # print(idx)
            f = file['kspace'][idx]


# I found out that for the compressed dataset, the bottleneck is not data I/O but decompression by the CPU.
# However, if the uncompressed dataset is used, I/O becomes the bottleneck.
# For cases with 2 processes and no SWMR, I still do not know why it is so slow.
# However, the CPU does not bottleneck and data I/O does.
# Therefore, I believe that the multiple processes trying to access data on disk at once are still the problem.
# I don't know why they can't access data in single file even without SWMR activated.
