import h5py
import numpy as np
from pathlib import Path
import warnings


"""
Code to transform the original dataset into a form that has faster data IO.
This is especially important for non-SSD storage.
For most systems, File I/O is the bottleneck for training speed.
"""


def check_chunk_size(data, chunk, file):
    mb = 2 ** 20  # megabyte
    chunk_bytes = np.prod(chunk) * data.itemsize
    if chunk_bytes > mb:
        warnings.warn(f'kspace chunk size for {file} is greater than 1MB. '
                      f'Specified chunk size is {chunk_bytes} for chunk configuration of {chunk}'
                      f'Please reconsider chunk size configurations. '
                      f'A chunk size greater than 1MB cannot utilize HDF5 caching by default.')


def make_compressed_dataset(data_folder, save_dir, **save_params):
    data_path = Path(data_folder)
    files = data_path.glob('*.h5')

    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    save_path = save_path / data_path.stem
    save_path.mkdir()

    for file in sorted(files):
        print(f'Processing {file}')
        with h5py.File(file, mode='r') as old_hf:
            attrs = dict(old_hf.attrs)
            kspace = np.asarray(old_hf['kspace'])

            # Chunk size should be below 1M for cache utilization. Complex data is 8 bytes.
            if kspace.ndim == 3:  # Single-coil case
                chunk = (1, kspace.shape[-2] // 4, kspace.shape[-1])  # dim=-2 is always 640 for fastMRI.
                # chunk = (1, 640, kspace.shape[-1])
                recons_key = 'reconstruction_esc'

            elif kspace.ndim == 4:
                chunk = (1, 1, kspace.shape[-2] // 4, kspace.shape[-1])
                # chunk = (1, 15, 640, kspace.shape[-1])
                recons_key = 'reconstruction_rss'
            else:
                raise TypeError('Invalid dimensions of input k-space data')

            test_set = recons_key not in old_hf.keys()
            labels = np.asarray(old_hf[recons_key]) if not test_set else None

        check_chunk_size(kspace, chunk, file)

        with h5py.File(save_path / file.name, mode='x', libver='latest') as new_hf:
            new_hf.attrs.update(attrs)
            new_hf.create_dataset('kspace', data=kspace, chunks=chunk, **save_params)
            if not test_set:
                new_hf.create_dataset(recons_key, data=labels, chunks=(1, 320, 320), **save_params)


def check_same(old_folder, new_folder):
    old_path = Path(old_folder)
    new_path = Path(new_folder)

    old_paths = list(old_path.glob('*.h5'))
    old_paths.sort()
    new_paths = list(new_path.glob('*.h5'))
    new_paths.sort()

    assert len(old_paths) == len(new_paths)

    for old, new in zip(old_paths, new_paths):
        assert old.name == new.name, 'Name is not the same.'
        print(f'Checking {new}')
        with h5py.File(old, mode='r') as old_hf, h5py.File(new, mode='r') as new_hf:
            assert dict(new_hf.attrs) == dict(old_hf.attrs)

            for key in new_hf.keys():
                assert np.all(np.asarray(old_hf[key]) == np.asarray(new_hf[key]))
    else:
        print('All is well!')


if __name__ == '__main__':
    train_dir = '/media/veritas/E/fastMRI/multicoil_train'
    val_dir = '/media/veritas/E/fastMRI/multicoil_val'
    test_dir = '/media/veritas/E/fastMRI/multicoil_test'

    data_root = '/media/veritas/F/lzfCompFastMRI'  # Compressed Fast MRI Dataset
    data_path_ = Path(data_root)

    # For floating point values, I have found that gzip level 1 and 9 give almost the same compression.
    # I have not checked whether this is also true for complex numbers but I presume this here.

    # I have found that gzip with level 1 is almost the same as gzip level 9 for complex data
    # when used with the shuffle filter. They both reduce the data by about half.
    # The differences are not great enough to justify the extra computational cost of higher gzip levels.
    # The differences do justify using gzip over lzf, however.
    # kwargs = dict(compression='gzip', compression_opts=1, shuffle=True, fletcher32=True)
    kwargs = dict(compression='lzf', shuffle=True)

    # Use compression if storing on hard drive, not SSD.
    make_compressed_dataset(train_dir, data_root, **kwargs)
    make_compressed_dataset(val_dir, data_root, **kwargs)
    make_compressed_dataset(test_dir, data_root, **kwargs)

    # check_same(train_dir, data_path_ / 'new_singlecoil_train')
    # check_same(val_dir, data_path_ / 'new_singlecoil_val')
    # check_same(test_dir, data_path_ / 'new_singlecoil_test')


