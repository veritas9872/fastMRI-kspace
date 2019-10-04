import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import h5py


def make_weighting_matrix(array, squared_weighting=False):
    *_, height, width = array.shape

    assert (height % 2 == 0) and (width % 2 == 0), 'Not absolutely necessary but odd sizes are unexpected.'
    mid_height = height / 2
    mid_width = width / 2

    # The indexing might be a bit confusing.
    x_coords = np.arange(start=-mid_width + 0.5, stop=mid_width + 0.5, step=1).reshape(1, width)
    x_coords = np.repeat(x_coords, repeats=height, axis=0)
    y_coords = np.arange(start=-mid_height + 0.5, stop=mid_height + 0.5, step=1).reshape(height, 1)
    y_coords = np.repeat(y_coords, repeats=width, axis=1)

    if squared_weighting:
        weighting_matrix = (x_coords ** 2) + (y_coords ** 2)
    else:
        weighting_matrix = (np.sqrt((x_coords ** 2) + (y_coords ** 2)))

    # weighting_matrix = weighting_matrix.reshape(height, width)

    return weighting_matrix, x_coords, y_coords


def semi_weight(array, weight_type):
    *_, height, width = array.shape
    assert width % 2 == 0, 'Not absolutely necessary but odd sizes are unexpected.'
    mid_width = width / 2
    mid_height = height / 2

    x_coords = np.arange(start=-mid_width + 0.5, stop=mid_width + 0.5, step=1).reshape(1, width)
    x_coords = np.repeat(x_coords, repeats=height, axis=0)
    y_coords = np.arange(start=-mid_height + 0.5, stop=mid_height + 0.5, step=1).reshape(height, 1)
    y_coords = np.repeat(y_coords, repeats=width, axis=1)

    if weight_type == 'distance':
        weighting_matrix = np.abs(x_coords)
    elif weight_type == 'root_distance':
        weighting_matrix = np.sqrt(np.abs(x_coords))
    elif weight_type == 'log_distance':
        weighting_matrix = np.log1p(np.abs(x_coords))
    elif weight_type == 'harmonic_distance':
        weighting_matrix = 1 / (0 + np.abs(x_coords))
    elif weight_type == 'exp_distance':
        weighting_matrix = np.exp(np.abs(x_coords) / (width * np.pi))
    else:
        raise NotImplementedError('Invalid weighting type.')

    return weighting_matrix, x_coords, y_coords


def main():
    file = '/media/veritas/D/FastMRI/multicoil_val/file1001689.h5'

    # with h5py.File(file, mode='r') as hf:
    #     kspace = hf['kspace'][18, 7]

    original = np.ones((320, 320))
    kspace = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(original, axes=(-2, -1)), axes=(-2, -1)), axes=(-2, -1))

    semi = np.fft.ifftshift(np.fft.ifft(np.fft.fftshift(kspace, axes=-2), axis=-2), axes=-2)
    # absolute = np.abs(semi)
    absolute = semi.imag

    weighting, x_coords, y_coords = semi_weight(absolute, weight_type='distance')

    weighted = absolute * weighting
    print(np.min(weighted), np.max(weighted), np.std(weighted))

    plt.gray()
    plt.figure(1)
    ax = plt.axes(projection='3d')
    ax.plot_surface(x_coords, y_coords, weighted)
    plt.show()


if __name__ == '__main__':
    main()

