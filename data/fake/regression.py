'''
File: regression.py
Created Date: 03 Sep, 2018
Author: Touko Hoshino
-----
Copyright (c) 2018 Hoshino Touko
'''
import numpy as np
import random
import matplotlib.pyplot as plt


def rand(w):
    res = (random.random() - 0.5) * (max(w) - min(w))
    if res == 0:
        res = (random.random() - 0.5) * max(w)
    return res


def generate_linear(w, num):
    dimensions = len(w)
    res = []
    for _ in range(num):
        x = []
        for _ in range(dimensions):
            x.append(rand(w) * 10)
        res.append((x, np.dot(x, w) + rand(w) * 35))
    return res


def gen_random_linear_regression_data(dimension, num=20):
    w = []
    for _ in range(dimension):
        w.append(random.randint(-20, 20))
    return generate_linear(w, num), w


def _test_linear_regression_data():
    data, w = gen_random_linear_regression_data(1, 100)
    print(w)
    x, y = list(map(lambda x: x[0], data)), list(map(lambda x: x[1], data))
    plt.plot(x, y, 'bo')
    plt.plot([min(x), max(x)], [np.dot(w, min(x)), np.dot(w, max(x))], 'r')
    plt.show()


if __name__ == '__main__':
    _test_linear_regression_data()
