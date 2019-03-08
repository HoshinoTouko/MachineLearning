"""
@File: Sampling.py
@Author: HoshinoTouko
@License: (C) Copyright 2014 - 2019, HoshinoTouko
@Contact: i@insky.jp
@Website: https://touko.moe/
@Created at: 3/8/2019 14:52
@Desc: 
"""
import os
import random

import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


class UniformSampling:
    @classmethod
    def get_dist(cls, a, b):
        if len(a) != len(b):
            raise Exception('Vector dimension not equal.')

        a = np.array(a)
        b = np.array(b)
        return np.linalg.norm(a - b)

    @classmethod
    def sampling(cls, features, labels, test_size=0.3):
        count = len(features)
        if len(labels) != count:
            raise Exception('Numbers of features and labels are not matched.')

        select_points = []
        selected = random.randint(0, count-1)
        min_length = [
            float('INF')
            for _ in range(count)
        ]
        while len(select_points) < count * (1 - test_size):
            select_points.append(selected)
            min_length[selected] = 0
            for i in range(count):
                if i in select_points:
                    continue

                min_length[i] = min(
                    min_length[i],
                    cls.get_dist(features[i], features[selected])
                )
            selected = np.argmax(min_length)

        features_train, features_test, labels_train, labels_test = [], [], [], []
        for i in range(count):
            if i in select_points:
                features_train.append(features[i])
                labels_train.append(labels[i])
                continue
            features_test.append(features[i])
            labels_test.append(labels[i])

        return features_train, features_test, labels_train, labels_test


def main():
    marker_size = 3
    marker = 'x'
    features = []
    labels = []
    with open(os.path.abspath('../data/files/classification/1.csv'), 'r') as fi:
        for line in fi.readlines():
            _line = line.strip().split(',')
            features.append(list(map(float, _line[:2])))
            labels.append(_line[2])
        features_train, features_test, labels_train, labels_test = UniformSampling.sampling(features, labels)

        # Draw plot
        for i in range(len(features_train)):
            color = 'green'
            if labels_train[i] == 'A':
                color = 'red'
            plt.scatter(features_train[i][0], features_train[i][1], color=color, label='Marked', marker=marker, s=marker_size)
        for i in range(len(features_test)):
            color = 'blue'
            plt.scatter(features_test[i][0], features_test[i][1], color=color, label='Unmarked data', marker=marker, s=marker_size)

        red_patch = mpatches.Patch(color='red', label='Marked A')
        green_patch = mpatches.Patch(color='green', label='Marked B')
        blue_patch = mpatches.Patch(color='blue', label='Unmarked data')
        plt.legend(handles=[red_patch, green_patch, blue_patch])
        plt.show()


if __name__ == '__main__':
    main()
