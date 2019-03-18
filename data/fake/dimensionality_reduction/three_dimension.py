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
    dots_a = np.random.multivariate_normal(mean, cov, 400)

    mean = np.array([8, 4, 3])
    cov = np.array(([2.5, 1, 0], [1, 2.5, 0], [0, 0, 0.2]))
    dots_b = np.random.multivariate_normal(mean, cov, 400)

    return np.concatenate((dots_a, dots_b)), \
           np.array([0 for i in range(400)] + [1 for i in range(400)])


def show_data(dots, labels):
    plt.subplot(111, projection='3d')
    plt.title('Three dimensions data')
    plt.scatter(dots[np.where(labels == 0), 0], dots[np.where(labels == 0), 1], dots[:, 2], c='b')
    plt.scatter(dots[np.where(labels == 1), 0], dots[np.where(labels == 1), 1], dots[:, 2], c='g')
    plt.show()
    plt.close()


def main():
    features, labels = generate_data()

    show_data(features, labels)


if __name__ == '__main__':
    main()
