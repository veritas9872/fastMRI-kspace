import numpy as np
import h5py


def ifft(array, ax):
    array = np.fft.ifftshift(array, axes=ax)
    array = np.fft.ifftn(array, axes=ax, norm='ortho')
    array = np.fft.fftshift(array, axes=ax)
    return array


def fft(array, ax):
    array = np.fft.ifftshift(array, axes=ax)
    array = np.fft.fftn(array, axis=ax, norm='ortho')
    array = np.fft.fftshift(array, axes=ax)
    return array


def main():
    file = '/media/veritas/D/FastMRI/multicoil_train/file1000048.h5'
    with h5py.File(file, mode='r') as hf:
        kspace = hf['kspace'][18, 8]

    lr_flip = np.flip(kspace, axis=-1)
    # ud_flip = np.flip(kspace, axis=-2)
    # flipped = np.flip(kspace, axis=(-2, -1))

    orig_img = np.abs(ifft(kspace, ax=(-1,)))
    lr_img = np.abs(ifft(lr_flip, ax=(-1,)))

    # print(np.allclose(orig_img, lr_img.conjugate(), rtol=1E-3, atol=1E-5))

    print(orig_img.shape)

    print(orig_img[318:322, 182:186])
    print()
    print(lr_img[318:322, 182:186])


if __name__ == '__main__':
    main()

