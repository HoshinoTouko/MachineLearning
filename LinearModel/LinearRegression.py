'''
File: LinearRegression.py
Created Date: 03 Sep, 2018
Author: Touko Hoshino
-----
Copyright (c) 2018 Hoshino Touko
'''
import numpy as np

from data.fake.regression import gen_random_linear_regression_data


class SimpleLinearRegression:
    def __init__(self, features, labels):
        self._features = features
        self.features = np.c_[features, np.ones([len(features)])]
        self.labels = labels

    @classmethod
    def is_full_rank(cls, matrix):
        matrix = np.array(matrix)
        try:
            tmp_matrix = np.dot(matrix.T, matrix)
            rank = np.linalg.matrix_rank(tmp_matrix)
            # print("Rank is %d" % rank)
            return rank == len(tmp_matrix)
        except Exception as e:
            print(e)
            return False
    

    def train(self):
        features = np.array(self.features)
        labels = np.array(self.labels)
        if self.is_full_rank(features):
            # Calculate w
            _tmp = np.asmatrix(np.dot(features.T, features))
            w_res = np.array(np.dot(np.dot(_tmp.I, features.T), labels))[0]
            w_res = w_res[:-1]
            # Calculate b
            b_res = np.mean(self.labels) - np.dot(w_res, np.mean(self._features, axis=0))
            # Return
            return w_res, b_res
        raise Exception('Not full rank matrix not support')


def test_simple_linear_regression():
    data, w, b = gen_random_linear_regression_data(10, 50)
    features, labels = list(map(lambda x: x[0], data)), list(map(lambda x: x[1], data))
    linearInstance = SimpleLinearRegression(features, labels)
    # Check if full rank
    # print(linearInstance.is_full_rank(features))
    train_w, train_b = linearInstance.train()

    # Analysis
    w = np.array(w, dtype=np.float)
    print(train_w, w)
    print(train_b, b)

    print('Error(RMSE): %.6f' % np.sqrt((train_w - w) ** 2).mean())
    print('Train''s b: %.6f, correct b: %.6f' % (train_b, b))


if __name__ == '__main__':
    test_simple_linear_regression()
