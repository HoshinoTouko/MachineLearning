"""
@File: simple_data.py
@Author: HoshinoTouko
@License: (C) Copyright 2014 - 2019, HoshinoTouko
@Contact: i@insky.jp
@Website: https://touko.moe/
@Created at: 3/12/2019 20:29
@Desc: 
"""
import numpy as np
import matplotlib.pyplot as plt


def generate_data(number=700):
    dots_a, dots_b, dots_c = _generate_data(number=1000)
    labels = []
    labels += [0 for _ in dots_a]
    labels += [1 for _ in dots_b]
    labels += [2 for _ in dots_c]
    return dots_a.tolist() + dots_b.tolist() + dots_c.tolist(), labels


def _generate_data(number=1000):
    mean = np.array([1, 2])
    cov = np.array(([2.5, 0], [0, 2.5]))
    dots_a = np.random.multivariate_normal(mean, cov, number)

    mean = np.array([5, 5])
    cov = np.array(([1.5, 0], [0, 1.5]))
    dots_b = np.random.multivariate_normal(mean, cov, number)

    mean = np.array([4, -3])
    cov = np.array(([2, 0], [0, 2]))
    dots_c = np.random.multivariate_normal(mean, cov, number)

    return dots_a, dots_b, dots_c


def show_plot():
    dot_size = 3

    dots_a, dots_b, dots_c = _generate_data()
    dots_a = np.array(dots_a)
    dots_b = np.array(dots_b)
    dots_c = np.array(dots_c)

    plt.scatter(dots_a[:, 0], dots_a[:, 1], color='blue', s=dot_size)
    plt.scatter(dots_b[:, 0], dots_b[:, 1], color='green', s=dot_size)
    plt.scatter(dots_c[:, 0], dots_c[:, 1], color='red', s=dot_size)
    plt.show()


if __name__ == '__main__':
    show_plot()
