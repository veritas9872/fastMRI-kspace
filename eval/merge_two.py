import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm


def concat_outputs(folder1, folder2, out_folder, index: int, key='reconstruction'):
    """

    Args:
        folder1: Folder to get the front data
        folder2: Folder to get the back data, the major part.
        out_folder: Folder to send outputs to.
        index: Index to cut outputs.
        key: Key for reconstructions.

    Returns:

    """
    path1 = Path(folder1)
    path2 = Path(folder2)
    out_path = Path(out_folder)
    out_path.mkdir(exist_ok=False)  # Prevent accidental over-writing on existing folder.
    assert path1.exists() and path2.exists()
    files1 = list(path1.glob('*.h5'))
    files2 = list(path2.glob('*.h5'))
    files1.sort()
    files2.sort()
    gzip = dict(compression='gzip', compression_opts=1, shuffle=True, fletcher32=True)
    for file1, file2 in tqdm(zip(files1, files2)):
        assert file1.name == file2.name
        with h5py.File(name=file1, mode='r') as hf1, h5py.File(name=file2, mode='r') as hf2:
            start = hf1[key][:index]
            end = hf2[key][index:]

        combine = np.concatenate([start, end], axis=0)
        with h5py.File(name=out_path / file1.name, mode='x') as hf:  # Prevent accidental over-writing on existing file.
            hf.create_dataset(name=key, data=combine, **gzip)


def concat_outputs_with_reference(folder1, folder2, out_folder, info_folder,
                                  index: int, acceleration: int, acc_key='acceleration', key='reconstruction'):
    path1 = Path(folder1)
    path2 = Path(folder2)
    info_path = Path(info_folder)
    out_path = Path(out_folder)
    out_path.mkdir(exist_ok=True)  # Prevent accidental over-writing on existing folder.
    assert path1.exists() and path2.exists() and info_path.exists()

    files1 = list(path1.glob('*.h5'))
    files2 = list(path2.glob('*.h5'))
    files_info = list(info_path.glob('*.h5'))
    assert len(files1) == len(files2) == len(files_info)

    files1.sort()
    files2.sort()
    files_info.sort()

    gzip = dict(compression='gzip', compression_opts=1, shuffle=True, fletcher32=True)

    for file1, file2, file_info in tqdm(zip(files1, files2, files_info)):
        assert file1.name == file2.name == file_info.name
        with h5py.File(file1, 'r') as hf1, h5py.File(file2, 'r') as hf2, h5py.File(file_info, 'r') as hfi:
            acc = hfi.attrs[acc_key]
            if acc == acceleration:
                start = hf1[key][:index]
                end = hf2[key][index:]
                combine = np.concatenate([start, end], axis=0)

                with h5py.File(out_path / file1.name, mode='x') as hf:
                    hf.create_dataset(name=key, data=combine, **gzip)


def check_correct(folder1, folder2, out_folder, index: int, key='reconstruction'):
    path1 = Path(folder1)
    path2 = Path(folder2)
    out_path = Path(out_folder)
    files1 = list(path1.glob('*.h5'))
    files2 = list(path2.glob('*.h5'))
    files_out = list(out_path.glob('*.h5'))
    files1.sort()
    files2.sort()
    files_out.sort()

    check = list()
    for file1, file2, file_out in zip(files1, files2, files_out):
        with h5py.File(name=file1, mode='r') as hf1, h5py.File(name=file2, mode='r') as hf2, h5py.File(name=file_out, mode='r') as hf:
            # print(hf1[key].shape, hf2[key].shape, hf[key].shape)
            start = hf1[key][0:index, :, :]
            front = hf[key][0:index, :, :]
            end = hf2[key][index:, :, :]
            back = hf[key][index:, :, :]
            a = np.all(start == front)
            b = np.all(end == back)
            check += [a, b]
    print(all(check))


if __name__ == '__main__':
    f1 = 'recons/JH_full_challenge'
    f2 = 'recons/JH_challenge'
    out = 'recons/acc4_challenge'
    info = '/media/user/Data/compFastMRI/multicoil_challenge'
    # concat_outputs(folder1=f1, folder2=f2, out_folder=out, index=12, key='reconstruction')
    # check_correct(folder1=f1, folder2=f2, out_folder=out, index=12, key='reconstruction')
    concat_outputs_with_reference(folder1=f1, folder2=f2, out_folder=out, info_folder=info,
                                  index=12, acceleration=4, acc_key='acceleration', key='reconstruction')