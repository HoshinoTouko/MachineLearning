"""
@File: three_dimension.py
@Author: HoshinoTouko
@License: (C) Copyright 2014 - 2019, HoshinoTouko
@Contact: i@insky.jp
@Website: https://touko.moe/
@Created at: 3/17/2019 16:44
@Desc: 
"""
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import matplotlib.pyplot as plt


def generate_data():
    mean = np.array([1, 2, 3])
    cov = np.array(([2.5, 1, 0], [1, 2.5, 0], [0, 0, 0.2]))
    dots = np.random.multivariate_normal(mean, cov, 400)

    return dots


def show_data(dots):
    plt.subplot(111, projection='3d')
    plt.title('Three dimensions data')
    plt.scatter(dots[:, 0], dots[:, 1], dots[:, 2], c='b')
    plt.show()
    plt.close()


def main():
    dots = generate_data()

    show_data(dots)


if __name__ == '__main__':
    main()
