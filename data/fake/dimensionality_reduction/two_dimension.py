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
    mean = np.array([1, 2])
    cov = np.array(([3.1, 0.5], [0.5, 0.1]))
    dots_a = np.random.multivariate_normal(mean, cov, 200)

    mean = np.array([8, 4])
    cov = np.array(([3.1, 0.5], [0.5, 0.1]))
    dots_b = np.random.multivariate_normal(mean, cov, 200)

    return np.concatenate((dots_a, dots_b)), \
           np.array([0 for i in range(200)] + [1 for i in range(200)])


def show_data(dots, labels):
    plt.title('Two dimensions data')
    plt.scatter(dots[np.where(labels == 0), 0], dots[np.where(labels == 0), 1], c='b')
    plt.scatter(dots[np.where(labels == 1), 0], dots[np.where(labels == 1), 1], c='g')
    plt.show()
    plt.close()


def main():
    features, labels = generate_data()

    show_data(features, labels)


if __name__ == '__main__':
    main()
