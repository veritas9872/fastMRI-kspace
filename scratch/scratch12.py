import matplotlib.pyplot as plt
import numpy as np
import h5py


if __name__ == '__main__':
    file = '/media/veritas/D/FastMRI/multicoil_val/file1001689.h5'

    with h5py.File(file, mode='r') as hf:
        kspace = hf['kspace'][15, 6]

    image = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace)))
    abs_img = np.abs(image)
    semi = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(kspace, axes=-2), axis=-2), axes=-2)
    angle = np.angle(image)
    absolute = np.abs(image)

    plt.gray()

    plt.figure(1)
    plt.imshow(angle)
    plt.colorbar()

    plt.figure(2)
    plt.imshow(absolute)
    plt.colorbar()

    plt.figure(3)
    plt.imshow(abs_img)
    plt.colorbar()

    plt.show()
