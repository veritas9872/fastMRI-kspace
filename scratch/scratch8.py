import torch

import numpy as np
import matplotlib.pyplot as plt


def trial(tensor, slope=0.1):
    assert isinstance(tensor, torch.Tensor), '`tensor` must be a tensor.'
    assert tensor.dim() == 5, '`tensor` is expected to be in the k-space format.'
    device = tensor.device
    height = tensor.size(-3)
    width = tensor.size(-2)
    assert (height % 2 == 0) and (width % 2 == 0), 'Not absolutely necessary but odd sizes are unexpected.'
    mid_height = height / 2
    mid_width = width / 2

    # The indexing might be a bit confusing.
    x_coords = torch.arange(start=-mid_width + 0.5, end=mid_width + 0.5, step=1,
                            device=device).view(1, width).expand(height, width)

    y_coords = torch.arange(start=-mid_height + 0.5, end=mid_height + 0.5, step=1,
                            device=device).view(height, 1).expand(height, width)

    weighting_matrix = slope * torch.sqrt((x_coords ** 2) + (y_coords ** 2))

    weighting_matrix = weighting_matrix.view(1, 1, height, width, 1)

    return weighting_matrix


if __name__ == '__main__':
    tester = torch.zeros(1, 2, 640, 368, 2)
    matrix = trial(tester, slope=1)
    multiplier = tester * matrix
    matrix = matrix.numpy().squeeze()
    plt.gray()
    plt.imshow(matrix)
    plt.colorbar()
    plt.show()
    print(matrix[0, 0])
