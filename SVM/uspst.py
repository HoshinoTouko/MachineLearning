"""
@File: some_datasets.py
@Author: HoshinoTouko
@License: (C) Copyright 2014 - 2018, HoshinoTouko
@Contact: i@insky.jp
@Website: https://touko.moe/
@Created at: 2018-06-20 14:30
@Desc: 
"""
from SVM.GaussSVM import GaussSVM
from sklearn.model_selection import train_test_split


def main():
    with open('../data/some_datasets/uspst_uni.txt') as fi:
        dataset = fi.readlines()
    with open('../data/some_datasets/uspst_uni_label.txt') as fi:
        labelset = fi.readlines()
    features = []
    labels = []
    for i in range(len(dataset)):
        line = dataset[i]
        data = line.split()
        features.append(list(map(float, data)))
        label = float(labelset[i])
        if label > 3:
            labels.append(1)
        else:
            labels.append(-1)
    # print(features)
    # print(labels)
    features_train, features_test, \
    labels_train, labels_test = train_test_split(
        features, labels, test_size=0.25, random_state=50)

    svm = GaussSVM(features_train, labels_train, o=17)
    # svm.kernel = lambda x, y: np.dot(x, y)
    svm.smo_train(500)
    svm.analysis()
    acc = svm.analysis(features_test, labels_test)
    print('Result of uspst dataset')
    print('Size of train dataset: %d' % len(features_train))
    print('Size of test dataset: %d' % len(features_test))
    print('Test dataset accuracy: %.2f%%' % (acc * 100))


if __name__ == '__main__':
    main()