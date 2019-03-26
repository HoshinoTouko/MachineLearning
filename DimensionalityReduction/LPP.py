"""
@File: DNE.py
@Author: HoshinoTouko
@License: (C) Copyright 2014 - 2019, HoshinoTouko
@Contact: i@insky.jp
@Website: https://touko.moe/
@Created at: 3/18/2019 23:57
@Desc: 
"""
from data.fake.dimensionality_reduction import three_dimension

from scipy.spatial import distance_matrix
import numpy as np
import matplotlib.pyplot as plt


class LPP:
    def __init__(self, features, target_dimension, k=5, t=10):
        self.features = features
        self.target_dimension = target_dimension
        self.k = k
        self.t = t

    def kernel(self, x1, x2):
        dist = np.linalg.norm(x1 - x2)
        return np.exp(-pow(dist, 2)/self.t)

    def process(self):
        n = len(self.features)
        n_dist = distance_matrix(self.features, self.features)

        # Get KNN arg matrix
        knn_arg_mat = list(map(lambda x: np.argsort(x)[1:self.k + 1].tolist(), n_dist))

        _wm = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1):
                if i in knn_arg_mat[j] or j in knn_arg_mat[i]:
                    x1 = self.features[i]
                    x2 = self.features[j]
                    _wm[i][j] = _wm[j][i] = self.kernel(x1, x2)

        _dm = np.matrix(np.diag(np.sum(_wm, axis=0)))
        _lm = _dm - _wm
        x = np.matrix(self.features).T

        eigenvalue, feature_vector = np.linalg.eig(
            x * _lm * _dm.I * x.I
        )
        print(eigenvalue, feature_vector)

        reduce_matrix = feature_vector[np.argsort(eigenvalue)]

        return x.T * reduce_matrix.T


def main():
    features, labels = three_dimension.generate_data()
    three_dimension.show_data(features, labels)

    lpp = LPP(features, 2)
    res = np.array(lpp.process())
    # print(res)

    plt.title('Reduced three dimensions data.')
    plt.scatter(res[np.argwhere(labels == 0), 0], res[np.argwhere(labels == 0), 1], c='b')
    plt.scatter(res[np.argwhere(labels == 1), 0], res[np.argwhere(labels == 1), 1], c='g')
    plt.show()
    plt.close()


if __name__ == '__main__':
    main()
