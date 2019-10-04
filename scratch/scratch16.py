import matplotlib.pyplot as plt
import h5py

plt.gray()
file = r'/home/veritas/PycharmProjects/fastMRI-kspace/eval/i2i_5/file1000254.h5'
with h5py.File(file, mode='r') as hf:
    image = hf['reconstruction'][16, ...]

plt.figure(1)
plt.imshow(image)
plt.show()
