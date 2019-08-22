import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn.init import _calculate_fan_in_and_fan_out

import numpy as np


class ComplexInitializer:
    """
    Implementation according to github page for Deep Complex Networks.
    I am not certain whether the mode is correct, the paper seems to suggest that it should be
    1 / sqrt(fan_in + fan_out) and 1 / sqrt(fan_in).
    However, I will follow my instincts and diverge from the code.
    """
    def __init__(self, method='kaiming'):
        assert method in ('kaiming', 'xavier'), 'Invalid initialization method.'
        self.method = method

    def get_weight_inits(self, weight_shape):
        fan_in, fan_out = _calculate_fan_in_and_fan_out(torch.zeros(size=weight_shape))
        if self.method == 'xavier':
            mode = 1 / np.sqrt(fan_in + fan_out)
        elif self.method == 'kaiming':
            mode = 1 / np.sqrt(fan_in)
        else:
            raise NotImplementedError('Invalid initialization method.')

        magnitude = np.random.rayleigh(scale=mode, size=weight_shape)
        phase = np.random.uniform(low=-np.pi, high=np.pi, size=weight_shape)
        weight_real = torch.from_numpy((magnitude * np.cos(phase)).astype(np.float32))
        weight_imag = torch.from_numpy((magnitude * np.sin(phase)).astype(np.float32))
        return weight_real, weight_imag


# class ComplexIndependentInitializer:
#     """
#     Due to the incomprehensible shape changing in the original code,
#     I am not sure if this is correct in shape.
#     This needs fixing later on.
#     """
#     def __init__(self, criterion='kaiming'):
#         assert criterion in ('kaiming', 'xavier')
#         self.criterion = criterion
#
#     def get_weight_inits(self, weight_shape):
#         num_rows = weight_shape[0] * weight_shape[1]  # C_out * C_in
#         num_cols = weight_shape[2] * weight_shape[3]  # K_height * K_width
#
#         flat_shape = (num_rows, num_cols)
#         real = np.random.uniform(size=flat_shape)
#         imag = np.random.uniform(size=flat_shape)
#         z = real + 1j * imag
#         u, _, v = np.linalg.svd(z)
#
#         # The columns of unitary_z are independent of one another.
#         unitary_z = np.dot(u, np.dot(np.eye(num_rows, num_cols), np.conjugate(v).T))
#
#         # reshape_z = np.reshape(unitary_z.T, newshape=(weight_shape[2], weight_shape[3], num_rows))
#         # reshape_z = np.reshape(reshape_z, )
#         #
#         # # I think that this will make the kernels independent of one another???
#         # independent_real = np.reshape(unitary_z.T.real, newshape=new_shape)
#         # independent_imag = np.reshape(unitary_z.T.imag, newshape=new_shape)
#         #
#         # fan_in, fan_out = _calculate_fan_in_and_fan_out(torch.zeros(size=weight_shape))
#         # if self.criterion == 'xavier':
#         #     desired_var = 1 / (fan_in + fan_out)
#         # elif self.criterion == 'kaiming':
#         #     desired_var = 1 / fan_in
#         # else:
#         #     raise NotImplementedError('Invalid criterion.')
#         #
#         # real_multiple = np.sqrt(desired_var / np.var(independent_real))
#         # imag_multiple = np.sqrt(desired_var / np.var(independent_imag))
#         #
#         # scaled_real = real_multiple * independent_real
#         # scaled_imag = imag_multiple * independent_imag
#         #
#         # weight_real = np.transpose(scaled_real, axes=(1, 2, 0)).reshape()
#         #
#         # return weight_real, weight_imag  # This will blow up as is right now.


class ComplexConv2d(nn.Module):
    """
    Complex convolution in 2D.
    Expects the real and imaginary data to be in the second (dim=1) dimension.
    Thus, the input and output tensors are all 5D.

    Please note that this layer does not implement gradient clipping at norm of 1, as was the original implementation
    set out in DEEP COMPLEX NETWORKS (Trabelsi et al.).
    This is just an imitation with the bare minimum necessary to get complex convolutions functioning.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super().__init__()
        kwargs = dict(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)

        self.conv_real = nn.Conv2d(**kwargs)
        self.conv_imag = nn.Conv2d(**kwargs)

        self.oc = out_channels

        # Weight initialization. Somewhat inelegant in style but it works (I think...).
        init = ComplexInitializer(method='kaiming')
        weight_real, weight_imag = init.get_weight_inits(weight_shape=self.conv_real.weight.shape)

        new_weights = {'conv_real.weight': weight_real, 'conv_imag.weight': weight_imag}
        self.load_state_dict(state_dict=new_weights, strict=False)

    def forward(self, tensor: Tensor) -> Tensor:
        assert tensor.dim() == 5, 'Expected (N,2,C,H,W) format.'
        assert tensor.size(1) == 2, 'Expected real/imag to be represented in the second dimension, dim=1.'
        r = tensor[:, 0]  # Separating the tensor before convolution increases speed for some reason.
        i = tensor[:, 1]  # Maybe slice indexing forces copying of memory.
        real = self.conv_real(r) - self.conv_imag(i)
        imag = self.conv_real(i) + self.conv_imag(r)
        return torch.stack([real, imag], dim=1)


class ComplexSpatialDropout2d(nn.Module):
    def __init__(self, p=0.):
        super().__init__()
        self.drop = nn.Dropout3d(p=p)

    def forward(self, tensor: Tensor) -> Tensor:
        assert tensor.dim() == 5, 'Expected (N,2,C,H,W) format.'
        assert tensor.size(1) == 2, 'Expected real/imag to be represented in the second dimension, dim=1.'
        output = tensor.transpose(1, 2)  # Put the channels on dim=1. The 2 is on the depth dimension to stay together.
        output = self.drop(output)  # 3D conv keeps the complex values together.
        output = output.transpose(2, 1)  # Return to original shape. Transpose does not copy memory.
        return output


# # Checking out code.
# class ComplexIndependentFilters:
#     # This initialization constructs complex-valued kernels
#     # that are independent as much as possible from each other
#     # while respecting either the He or the Glorot criterion.
#     def __init__(self, kernel_size, input_dim,
#                  weight_dim, nb_filters=None,
#                  criterion='kaiming', seed=None):
#
#         # `weight_dim` is used as a parameter for sanity check
#         # as we should not pass an integer as kernel_size when
#         # the weight dimension is >= 2.
#         # nb_filters == 0 if weights are not convolutional (matrix instead of filters)
#         # then in such a case, weight_dim = 2.
#         # (in case of 2D input):
#         #     nb_filters == None and len(kernel_size) == 2 and_weight_dim == 2
#         # conv1D: len(kernel_size) == 1 and weight_dim == 1
#         # conv2D: len(kernel_size) == 2 and weight_dim == 2
#         # conv3d: len(kernel_size) == 3 and weight_dim == 3
#
#         assert len(kernel_size) == weight_dim and weight_dim in {0, 1, 2, 3}
#         self.nb_filters = nb_filters
#         self.kernel_size = kernel_size
#         self.input_dim = input_dim
#         self.weight_dim = weight_dim
#         self.criterion = criterion
#         self.seed = 1337 if seed is None else seed
#
#     def __call__(self, weight_shape):
#
#         if self.nb_filters is not None:
#             num_rows = self.nb_filters * self.input_dim
#             num_cols = np.prod(self.kernel_size)
#         else:
#             # in case it is the kernel is a matrix and not a filter
#             # which is the case of 2D input (No feature maps).
#             num_rows = self.input_dim
#             num_cols = self.kernel_size[-1]
#
#         flat_shape = (int(num_rows), int(num_cols))
#         rng = np.random.RandomState(self.seed)
#         r = rng.uniform(size=flat_shape)
#         i = rng.uniform(size=flat_shape)
#         z = r + 1j * i
#         u, _, v = np.linalg.svd(z)
#         unitary_z = np.dot(u, np.dot(np.eye(int(num_rows), int(num_cols)), np.conjugate(v).T))
#         print(1, unitary_z.shape)
#         real_unitary = unitary_z.real
#         imag_unitary = unitary_z.imag
#         if self.nb_filters is not None:
#             indep_real = np.reshape(real_unitary, (num_rows,) + tuple(self.kernel_size))
#             indep_imag = np.reshape(imag_unitary, (num_rows,) + tuple(self.kernel_size))
#             fan_in, fan_out = _calculate_fan_in_and_fan_out(torch.zeros(size=weight_shape))
#         else:
#             indep_real = real_unitary
#             indep_imag = imag_unitary
#             fan_in, fan_out = (int(self.input_dim), self.kernel_size[-1])
#
#         print(2, indep_real.shape)
#
#         if self.criterion == 'glorot':
#             desired_var = 1. / (fan_in + fan_out)
#         elif self.criterion == 'kaiming':
#             desired_var = 1. / (fan_in)
#         else:
#             raise ValueError('Invalid criterion: ' + self.criterion)
#
#         multip_real = np.sqrt(desired_var / np.var(indep_real))
#         multip_imag = np.sqrt(desired_var / np.var(indep_imag))
#         scaled_real = multip_real * indep_real
#         scaled_imag = multip_imag * indep_imag
#
#         if self.weight_dim == 2 and self.nb_filters is None:
#             weight_real = scaled_real
#             weight_imag = scaled_imag
#         else:
#             kernel_shape = tuple(self.kernel_size) + (int(self.input_dim), self.nb_filters)
#             if self.weight_dim == 1:
#                 transpose_shape = (1, 0)
#             elif self.weight_dim == 2 and self.nb_filters is not None:
#                 transpose_shape = (1, 2, 0)
#             elif self.weight_dim == 3 and self.nb_filters is not None:
#                 transpose_shape = (1, 2, 3, 0)
#             else:
#                 raise ValueError()
#
#             weight_real = np.transpose(scaled_real, transpose_shape)
#             weight_imag = np.transpose(scaled_imag, transpose_shape)
#             weight_real = np.reshape(weight_real, kernel_shape)
#             weight_imag = np.reshape(weight_imag, kernel_shape)
#         weight = np.concatenate([weight_real, weight_imag], axis=-1)
#
#         return weight


if __name__ == '__main__':
    # torch.autograd.set_grad_enabled(False)
    tensor = torch.rand(1, 2, 15, 320, 320)
    conv_real = nn.Conv2d(in_channels=15, out_channels=4, kernel_size=3, padding=1)
    conv_imag = nn.Conv2d(in_channels=15, out_channels=4, kernel_size=3, padding=1)

    def forward(tensor: Tensor) -> Tensor:
        assert tensor.dim() == 5, 'Expected (N,2,C,H,W) format.'
        assert tensor.size(1) == 2, 'Expected real/imag to be represented in the second dimension, dim=1.'

        r = tensor[:, 0]
        i = tensor[:, 1]
        print('Why?')
        print(r.shape)
        real = conv_real(r) - conv_imag(i)
        print('?')
        imag = conv_real(i) + conv_imag(r)
        return torch.stack([real, imag], dim=1)

    def _other_forward(tensor: Tensor) -> Tensor:
        assert tensor.dim() == 5, 'Expected (N,2,C,H,W) format.'
        assert tensor.size(1) == 2, 'Expected real/imag to be represented in the second dimension, dim=1.'
        n, _, c, h, w = tensor.shape
        folded = tensor.view(n * 2, c, h, w)  # Merge to batch dimension for efficient calculation.
        real_conv = conv_real(folded).view(tensor.shape)
        imag_conv = conv_imag(folded).view(tensor.shape)
        real = real_conv[:, 0] - imag_conv[:, 1]
        imag = real_conv[:, 1] + imag_conv[:, 0]
        return torch.stack([real, imag], dim=1)

    x = forward(tensor)
    y = _other_forward(tensor)

    print(torch.all(x == y))
