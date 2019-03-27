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


class LDNE:
    def __init__(self, features, labels, target_dimension, k=5, t=10):
        self.features = features
        self.labels = labels
        self.target_dimension = target_dimension
        self.k = k
        self.t = t

    def kernel(self, x1, x2):
        dist = np.linalg.norm(x1 - x2)
        return np.exp(-pow(dist, 2)/self.t)

    def process(self):
        n_dist = distance_matrix(self.features, self.features)

        # Get KNN arg matrix
        knn_arg_mat = list(map(lambda x: np.argsort(x)[1:self.k + 1].tolist(), n_dist))

        # Get k_nearest_graphic (W)
        n = len(self.features)
        w = np.zeros((n, n), dtype=np.short)
        for i in range(n):
            for j in range(n):
                if i in knn_arg_mat[j] or j in knn_arg_mat[i]:
                    x1 = self.features[i]
                    x2 = self.features[j]
                    exp = self.kernel(x1, x2)
                    if self.labels[i] == self.labels[j]:
                        w[i][j] = exp
                        continue
                    w[i][j] = -exp
                    continue

        d = np.diag(np.sum(w, axis=0))
        l = d - w
        x = np.matrix(self.features).T

        eigenvalue, feature_vector = np.linalg.eig(
            x * l * x.T
        )
        print(eigenvalue, feature_vector)

        reduce_matrix = feature_vector[np.argsort(eigenvalue)]

        return x.T * reduce_matrix.T


def main():
    features, labels = three_dimension.generate_data()
    three_dimension.show_data(features, labels)

    ldne = LDNE(features, labels, 2)
    res = np.array(ldne.process())
    # print(res)

    plt.title('Reduced three dimensions data.')
    plt.scatter(res[np.argwhere(labels == 0), 0], res[np.argwhere(labels == 0), 1], c='b')
    plt.scatter(res[np.argwhere(labels == 1), 0], res[np.argwhere(labels == 1), 1], c='g')
    plt.show()
    plt.close()


if __name__ == '__main__':
    main()
