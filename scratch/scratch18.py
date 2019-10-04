from h5py import File
import numpy as np

file1 = '/media/veritas/D/FastMRI/multicoil_test/file1000082.h5'
file2 = '/media/veritas/D/FastMRI/multicoil_test_v2/file1000082_v2.h5'

with File(file1, 'r') as hf1, File(file2, 'r') as hf2:
    print(hf1.keys())
    print(hf2.keys())
    k1 = hf1['kspace'][()]
    k2 = hf2['kspace'][()]
    print(np.all(k1 == k2))
