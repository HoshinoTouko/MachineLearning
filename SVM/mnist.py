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
        features.append(mnist_test_features[i][0])
        if mnist_test_label[i] == 0 or mnist_test_label[i] == 1:
            labels.append(1)
        else:
            labels.append(-1)
    print('Split dataset...')
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.65, random_state=70)

    print('Training...')
    print('%d samples' % len(y_train))
    svm = GaussSVM(x_train, y_train, c=5, o=100000)
    svm.smo_train(2000, 0.95)
    print('Accuracy: %.2f%%' % (svm.analysis(x_test, y_test) * 100))


if __name__ == '__main__':
    main()


