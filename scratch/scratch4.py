import h5py
# import numpy as np

level = 0
compression = 'lzf'


comp = dict(compression=compression, shuffle=True, fletcher32=True)


old_name = '/media/veritas/E/fastMRI/multicoil_train/file1000001.h5'

with h5py.File(name=old_name, mode='r') as old_file, h5py.File(f'{compression}.h5', mode='x') as new_file:
    old_data = old_file['kspace']
    last_dim = old_data.shape[-1]
    chunks = (1, 15, 640, last_dim)
    new_file.create_dataset(name='kspace', data=old_data, chunks=chunks, **comp)
