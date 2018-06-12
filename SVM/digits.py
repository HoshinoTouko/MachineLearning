"""
@File: test_usps.py
@Author: HoshinoTouko
@License: (C) Copyright 2014 - 2018, HoshinoTouko
@Contact: i@insky.jp
@Website: https://touko.moe/
@Created at: 2018-06-12 22:40
@Desc: 
"""
from sklearn.datasets import load_digits, load_iris
from sklearn.model_selection import train_test_split

from SVM.GaussSVM import GaussSVM


def main():
    iris = load_iris()
    digits = load_digits()
    print(digits.data.shape)
    print(digits.target.shape)

    features = []
    labels = []
    for i in range(digits.target.shape[0]):
        features.append(digits.data[i])
        if digits.target[i] == 0 or digits.target[i] == 1:
            labels.append(1)
        else:
            labels.append(-1)

    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.3, random_state=70)

    svm = GaussSVM(x_train, y_train, c=10)
    svm.smo_train(1000, 0.98)
    print('Accuracy: %.2f%%' % (svm.analysis(x_test, y_test) * 100))


if __name__ == '__main__':
    main()
