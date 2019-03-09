"""
@File: RareCategory.py
@Author: HoshinoTouko
@License: (C) Copyright 2014 - 2019, HoshinoTouko
@Contact: i@insky.jp
@Website: https://touko.moe/
@Created at: 3/10/2019 0:21
@Desc: 
"""
import random

import numpy as np
import matplotlib.pyplot as plt


def _test():
    dot_size = 1

    mean = np.array([1, 1])
    cov = np.array(([3, 0], [0, 3]))
    dots_large_a = np.random.multivariate_normal(mean, cov, 4450)

    mean = np.array([6, 6])
    cov = np.array(([1.5, 0], [0, 1.5]))
    dots_large_b = np.random.multivariate_normal(mean, cov, 4450)

    mean = np.array([2, 4])
    cov = np.array(([0.1, 0], [0, 0.1]))
    dots_small = np.random.multivariate_normal(mean, cov, 100)

    dots = dots_large_a.tolist() + \
           dots_large_b.tolist() + \
           dots_small.tolist()

    plt.scatter(dots_large_a[:, 0], dots_large_a[:, 1], color='blue', s=dot_size)
    plt.scatter(dots_large_b[:, 0], dots_large_b[:, 1], color='green', s=dot_size)
    plt.scatter(dots_small[:, 0], dots_small[:, 1], color='red', s=dot_size)
    plt.show()

    plt.close()

    test_sampling = random.sample(dots, 1000)
    test_sampling = np.array(test_sampling)
    plt.scatter(test_sampling[:, 0], test_sampling[:, 1], color='blue', s=dot_size)
    plt.show()


if __name__ == '__main__':
    _test()
