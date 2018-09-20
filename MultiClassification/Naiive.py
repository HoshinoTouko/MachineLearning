'''
File: Naiive.py
Created Date: 19 Sep, 2018
Author: Touko Hoshino
-----
Copyright (c) 2018 Hoshino Touko
'''
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.datasets import load_digits
from data.mnist.load_mnist import load_datasets as load_mnist

from sklearn.preprocessing import scale, StandardScaler
from sklearn import svm

from sklearn.model_selection import GridSearchCV

import numpy as np


class Naiive:
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        # self.svm0 = svm.SVC(C=10.0, gamma=0.001)  Digit dataset
        self.svm0 = svm.SVC(cache_size=800, C=10.0, gamma=0.001)
        self.svm1 = svm.SVC(cache_size=800, C=10.0, gamma=0.001)
        self.svm2 = svm.SVC(cache_size=800, C=10.0, gamma=0.001)
        self.svm3 = svm.SVC(cache_size=800, C=10.0, gamma=0.001)

    def fit(self):
        for i in range(4):
            print('Training SVM %d' % i)
            self.__dict__['svm%d' % i].fit(self.features, self.labels[:,i])
            print('SVM %d trained' % i)
    
    def score(self, features, labels):
        acc = 0.
        res = np.array(np.asmatrix(list(map(
            lambda i: self.__dict__['svm%d' % i].predict(features),
            list(range(4))
        ))).T)
        
        for num in range(len(labels)):
            correct_number = int(''.join(list(map(str, labels[num]))), 2)
            predict_number = int(''.join(list(map(str, res[num]))), 2)
            if correct_number == predict_number:
                acc += 1
            else:
                print('Correct: %d, predict: %d' % (correct_number, predict_number))
        return acc / len(features)


def pre_process(features):
    res = []
    for x in range(0, 28, 2):
        for y in range(0, 28, 2):
            tmp = features[x + y * 28]
            tmp += features[x + 1 + y * 28]
            tmp += features[x + (y + 1) * 28]
            tmp += features[x + 1 + (y + 1) * 28]
            res.append(tmp)
    return res


def main():
    # digits = load_digits()
    # print(digits.data.shape)
    # print(digits.target.shape)

    # features = []
    # labels = []
    # for i in range(digits.target.shape[0]):
    #     features.append(digits.data[i])
    #     labels.append(
    #         list(map(
    #             int, 
    #             list('000' + bin(digits.target[i]).replace('0b', ''))[-4:]
    #     )))

    mnist_features, mnist_label, mnist_test_features, mnist_test_label = load_mnist()

    features = []
    labels = []
    print('Data preprocessing...')
    for i in range(mnist_test_features.shape[0]):
        # features.append(pre_process(mnist_test_features[i][0]))
        features.append(mnist_test_features[i][0])
        labels.append(
            list(map(
                int, 
                list('000' + bin(mnist_test_label[i][0]).replace('0b', ''))[-4:]
        )))
    features = StandardScaler().fit_transform(features)
    print(len(features[0]))
    print('Split dataset...')

    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.4)
    
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    svc = svm.SVC

    # grid = GridSearchCV(
    #     svc(cache_size=500, verbose=True), 
    #     param_grid={"C":[0.1, 1, 10, 15, 20], "gamma": [0.003, 0.004, 0.005, 0.001]}, cv=4
    # )
    # grid.fit(x_train, y_train[:,0])
    # print("The best parameters are %s with a score of %0.2f"
    #   % (grid.best_params_, grid.best_score_))
    
    naiive = Naiive(x_train, y_train)
    naiive.fit()

    print('Get an accuracy of %.4f%%' % 
        (naiive.score(x_test, y_test) * 100))


if __name__ == '__main__':
    main()

