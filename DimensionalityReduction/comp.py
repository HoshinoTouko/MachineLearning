"""
@File: comp.py
@Author: HoshinoTouko
@License: (C) Copyright 2014 - 2019, HoshinoTouko
@Contact: i@insky.jp
@Website: https://touko.moe/
@Created at: 3/26/2019 19:25
@Desc: 
"""
from DimensionalityReduction.DNE import DNE
from DimensionalityReduction.LPP import LPP
from DimensionalityReduction.PCA import PCA
from data.fake.dimensionality_reduction import three_dimension

import copy
import numpy as np
import matplotlib.pyplot as plt


def main():
    features, labels = three_dimension.generate_data()
    three_dimension.show_data(features, labels)

    pca = PCA(copy.deepcopy(features), 2)
    res = np.array(pca.process())
    # print(res)

    plt.title('PCA')
    plt.scatter(res[np.argwhere(labels == 0), 0], res[np.argwhere(labels == 0), 1], c='b')
    plt.scatter(res[np.argwhere(labels == 1), 0], res[np.argwhere(labels == 1), 1], c='g')
    plt.show()
    plt.close()

    dne = DNE(copy.deepcopy(features), labels, 2)
    res = np.array(dne.process())
    # print(res)

    plt.title('DNE')
    plt.scatter(res[np.argwhere(labels == 0), 0], res[np.argwhere(labels == 0), 1], c='b')
    plt.scatter(res[np.argwhere(labels == 1), 0], res[np.argwhere(labels == 1), 1], c='g')
    plt.show()
    plt.close()

    lpp = LPP(copy.deepcopy(features), 2, t=200)
    res = np.array(lpp.process())
    # print(res)

    plt.title('LPP')
    plt.scatter(res[np.argwhere(labels == 0), 0], res[np.argwhere(labels == 0), 1], c='b')
    plt.scatter(res[np.argwhere(labels == 1), 0], res[np.argwhere(labels == 1), 1], c='g')
    plt.show()
    plt.close()


if __name__ == '__main__':
    main()
