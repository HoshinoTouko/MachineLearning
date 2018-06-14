"""
@File: mnist.py
@Author: HoshinoTouko
@License: (C) Copyright 2014 - 2018, HoshinoTouko
@Contact: i@insky.jp
@Website: https://touko.moe/
@Created at: 2018-06-13 14:19
@Desc: 
"""
from SVM.GaussSVM import GaussSVM
from sklearn.model_selection import train_test_split
from data.mnist.load_mnist import load_datasets

import numpy as np


def pre_process(features):
    res = []
    for x in range(0, 16, 2):
        for y in range(0, 16, 2):
            tmp = features[x + y * 8]
            tmp += features[x + 1 + y * 8]
            tmp += features[x + (y + 1) * 8]
            tmp += features[x + 1 + (y + 1) * 8]
            res.append(tmp/4)
    return res


def main():
    mnist_features, mnist_label, mnist_test_features, mnist_test_label = load_datasets()

    features = []
    labels = []
    print('Data preprocessing...')
    # for i in range(mnist_features.shape[0]):
    #     features.append(mnist_features[i][0])
    #     if mnist_label[i] == 0 or mnist_label[i] == 1:
    #         labels.append(1)
    #     else:
    #         labels.append(-1)
    for i in range(mnist_test_features.shape[0]):
        features.append(
            pre_process(mnist_test_features[i][0]))
        if mnist_test_label[i] == 0 or mnist_test_label[i] == 1:
            labels.append(1)
        else:
            labels.append(-1)
    print('Split dataset...')
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.65, random_state=70)

    print(np.array(features).shape)
    print('Training...')
    print('%d samples' % len(y_train))
    svm = GaussSVM(x_train, y_train, c=2.5, o=1000)
    svm.smo_train(400, 0.9)
    print('Accuracy: %.2f%%' % (svm.analysis(x_test, y_test) * 100))


if __name__ == '__main__':
    main()
