"""
@File: dataset.py
@Author: HoshinoTouko
@License: (C) Copyright 2014 - 2018, HoshinoTouko
@Contact: i@insky.jp
@Website: https://touko.moe/
@Created at: 2018-06-06 19:14
@Desc: 
"""
import matplotlib.pyplot as plt
import random
import math


def generate_sin_data(times, center_x, rand):
    res = []
    for _ in range(times):
        rand_x = center_x + 3 * math.pi * (random.random() - 1)
        res.append((
            rand_x,
            math.sin(rand_x) + rand * (random.random() + 0.2)
        ))
    return res


def generate_sin_data_for_test():
    res = []
    res.append(generate_sin_data(30, 10, 2))
    res.append(generate_sin_data(30, 10, -2))
    return res


def generate_two_dimensional_linear_data(center_x, center_y, center_range, k, times, up_or_down):
    res = []

    def func(x):
        return k * (x - center_x) + center_y

    for _ in range(times):
        rand_x = 5 * random.random()
        rand_y = func(rand_x)
        if up_or_down > 0:
            res.append((
                rand_x,
                rand_y + 5 * random.random() + 1))
        else:
            res.append((
                rand_x,
                rand_y - 5 * random.random() - 1))

    return res


def generate_two_dimensional_linear_data_for_test():
    center_range = 10
    center_x = 2 * center_range * random.random() - center_range
    center_y = 2 * center_range * random.random() - center_range
    k = 1 / random.uniform(-1, 1)
    res = []
    res.append(generate_two_dimensional_linear_data(
        center_x, center_y, center_range, k, 30, -1))
    res.append(generate_two_dimensional_linear_data(
        center_x, center_y, center_range, k, 30, 1))

    return res


def generate_circle(times, r, noise):
    res = []
    noise = math.log10(noise+1.8)
    for _ in range(times):
        new_r = r + 2 * noise * random.random() - noise

        x = 2 * new_r * random.random() - new_r
        y = math.sqrt(new_r * new_r - x * x)
        if random.random() >= 0.5:
            x = -x
        if random.random() >= 0.5:
            y = -y
        if random.random() >= 0.5:
            res.append((x, y))
        else:
            res.append((y, x))

    return res


def generate_concentric_circles():
    res = []
    res.append(generate_circle(50, 5, math.e))
    res.append(generate_circle(50, 3, math.e))
    return res


if __name__ == '__main__':
    data = generate_concentric_circles()
    data = generate_two_dimensional_linear_data_for_test()
    # data = generate_sin_data_for_test()
    plt.scatter(list(map(lambda x: x[0], data[0])), list(map(lambda x: x[1], data[0])))
    plt.scatter(list(map(lambda x: x[0], data[1])), list(map(lambda x: x[1], data[1])))
    plt.show()
