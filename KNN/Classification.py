"""
@File: Classification.py
@Author: HoshinoTouko
@License: (C) Copyright 2014 - 2019, HoshinoTouko
@Contact: i@insky.jp
@Website: https://touko.moe/
@Created at: 3/6/2019 19:41
@Desc: 
"""
from collections import Counter
from sklearn.model_selection import train_test_split

import os
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import itertools
import data.fake.classification as dt
from Utils.Sampling import UniformSampling


class KNNClassifier:
    def __init__(self, features_train, labels_train):
        if len(features_train) < 20:
            raise Exception('Train data set too small.')
        self.features_train = features_train
        self.labels_train = labels_train
        self.dimension = len(features_train[0])

    def test(self, feature):
        res = []
        for i in range(len(self.features_train)):
            dist = 0
            for j in range(self.dimension):
                dist += pow(self.features_train[i][j] - feature[j], 2)
            res.append([dist, self.labels_train[i]])

        selects = sorted(res, key=lambda x: x[0])[:10]

        return Counter([x[1] for x in selects]).most_common(1)

    def analyse(self, features, labels):
        acc = 0.
        count = len(features)
        for i in range(len(features)):
            res = self.test(features[i])
            if res[0][0] == labels[i]:
                acc += 1
        print('Accuracy: %.2f%%, train %s features, test %s samples.' % (
            100 * acc / count, len(self.features_train), count
        ))


def main():
    # data = dt.generate_two_dimensional_linear_data_for_test(200)
    # features = [x for x in itertools.chain(data[0], data[1])]
    # labels = [-1 for x in range(len(data[0]))]
    # labels += [1 for x in range(len(data[1]))]
    features = []
    labels = []
    with open(os.path.abspath('../data/files/classification/1.csv'), 'r') as fi:
        for line in fi.readlines():
            _line = line.strip().split(',')
            features.append(list(map(float, _line[:2])))
            labels.append(_line[2])

    print('Sampling...')
    # features_train, features_test, \
    # labels_train, labels_test = train_test_split(
    #     features, labels, test_size=0.75, random_state=50)
    features_train, features_test, \
        labels_train, labels_test = UniformSampling.sampling(
            features, labels, test_size=0.75)

    _KNNClassifier = KNNClassifier(features_train, labels_train)

    print('Analysing...')
    _KNNClassifier.analyse(features_test, labels_test)

    # Draw
    print('Draw scatter graph...')
    for i in range(len(features_train)):
        color = 'green'
        if labels_train[i] == 'A':
            color = 'red'
        plt.scatter(features_train[i][0], features_train[i][1], color=color, label='Marked')
    for i in range(len(features_test)):
        color = 'blue'
        plt.scatter(features_test[i][0], features_test[i][1], color=color, label='Unmarked data')

    red_patch = mpatches.Patch(color='red', label='Marked A')
    green_patch = mpatches.Patch(color='green', label='Marked B')
    blue_patch = mpatches.Patch(color='blue', label='Unmarked data')
    plt.legend(handles=[red_patch, green_patch, blue_patch])
    plt.show()


if __name__ == '__main__':
    main()
