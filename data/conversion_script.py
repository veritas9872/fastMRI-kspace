import h5py
import numpy as np
from pathlib import Path


"""
Code to transform the original dataset into a form that has faster data IO.
This is especially important for non-SSD storage.
For most systems, File I/O is the bottleneck for training speed.
"""


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
                recons_key = 'reconstruction_esc'

            elif kspace.ndim == 4:
                chunk = (1, 1, kspace.shape[-2] // 4, kspace.shape[-1])
                recons_key = 'reconstruction_rss'
            else:
                raise TypeError('Invalid dimensions of input k-space data')

            test_set = recons_key not in old_hf.keys()
            labels = np.asarray(old_hf[recons_key]) if not test_set else None

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

    data_root = '/media/veritas/F/compFastMRI'  # Compressed Fast MRI Dataset
    data_path_ = Path(data_root)

    # For floating point values, I have found that gzip level 1 and 9 give almost the same compression.
    # I have not checked whether this is also true for complex numbers but I presume this here.
    gzip = dict(compression='gzip', compression_opts=1, shuffle=True, fletcher32=False)

    # Use compression if storing on hard drive, not SSD.
    make_compressed_dataset(train_dir, data_root, **gzip)
    make_compressed_dataset(val_dir, data_root, **gzip)
    make_compressed_dataset(test_dir, data_root, **gzip)

    # check_same(train_dir, data_path_ / 'new_singlecoil_train')
    # check_same(val_dir, data_path_ / 'new_singlecoil_val')
    # check_same(test_dir, data_path_ / 'new_singlecoil_test')


