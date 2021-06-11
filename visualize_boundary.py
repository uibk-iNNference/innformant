import os
import matplotlib.pyplot as plt
import numpy as np
from math import log10, sqrt

from resources import data, utils

if __name__ == "__main__":
    sample_index = 70
    sample, label = data.get_single_cifar10_test_sample(index=sample_index, include_label=True)
    plt.imshow(sample[0])
    plt.show()

    save_path = os.path.join('boundaries', 'amdrome-intelsandybridge', f'cifar10_{sample_index}.npy')
    boundary = np.load(save_path)
    plt.imshow(boundary[0])
    plt.show()

    print(f'PSNR of sample {sample_index} is {utils.calculate_psnr(sample, boundary)}')
