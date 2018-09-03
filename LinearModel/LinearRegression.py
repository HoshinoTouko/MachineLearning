'''
File: LinearRegression.py
Created Date: 03 Sep, 2018
Author: Touko Hoshino
-----
Copyright (c) 2018 Hoshino Touko
'''
import numpy as np

from data.fake.regression import gen_random_linear_regression_data


class LinearRegression:
    def __init__(self, features, labels):
        self.features = features
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
            _tmp = np.asmatrix(np.dot(features.T, features))
            res = np.dot(np.dot(_tmp.I, features.T), labels)
            return np.array(res)
        return False


def main():
    data, w = gen_random_linear_regression_data(10, 50)
    features, labels = list(map(lambda x: x[0], data)), list(map(lambda x: x[1], data))
    linearInstance = LinearRegression(features, labels)
    # Check if full rank
    # print(linearInstance.is_full_rank(features))
    train_res = linearInstance.train()[0]

    # Analysis
    w = np.array(w, dtype=np.float)
    print(train_res, w)
    print('Error(RMSE): %.6f' % np.sqrt((train_res - w) ** 2).mean())


if __name__ == '__main__':
    main()
