"""
@File: PCA.py
@Author: HoshinoTouko
@License: (C) Copyright 2014 - 2019, HoshinoTouko
@Contact: i@insky.jp
@Website: https://touko.moe/
@Created at: 3/17/2019 17:49
@Desc: 
"""
from data.fake.dimensionality_reduction import three_dimension, two_dimension

import numpy as np
import matplotlib.pyplot as plt


class PCA:
    def __init__(self, features, target_dimension):
        self.features = features
        self.target_dimension = target_dimension

    def process(self):
        dots = self.features

        cov = np.cov(np.matrix(dots).T)
        # print('Cov')
        # print(cov)

        eigenvalue, feature_vector = np.linalg.eig(cov)
        # print('eigenvalue')
        # print(eigenvalue)
        # print('feature_vector')
        # print(feature_vector)

        reduce_matrix = feature_vector[np.argsort(-eigenvalue)[:self.target_dimension]]
        print('%s dimension reduce_matrix.' % self.target_dimension)
        print(reduce_matrix)

        return np.matrix(self.features) * reduce_matrix.T


def main():
    dots = three_dimension.generate_data()
    three_dimension.show_data(dots)

    pca = PCA(dots, 2)
    res = np.array(pca.process())
    # print(res)

    plt.title('Reduced three dimensions data.')
    plt.scatter(res[:, 0], res[:, 1], c='b')
    plt.show()
    plt.close()

    dots = two_dimension.generate_data()
    two_dimension.show_data(dots)

    pca = PCA(dots, 1)
    res = np.array(pca.process())
    # print(res)

    plt.title('Reduced two dimensions data.')
    plt.scatter(res[:, 0], np.zeros(shape=res[:, 0].shape), c='b')
    plt.show()
    plt.close()


if __name__ == '__main__':
    main()
