"""
@File: LinearSVM.py
@Author: HoshinoTouko
@License: (C) Copyright 2014 - 2018, HoshinoTouko
@Contact: i@insky.jp
@Website: https://touko.moe/
@Created at: 2018-06-08 18:40
@Desc: 
"""
import itertools
import random

import SVM.dataset as dt
import numpy as np

import matplotlib.pyplot as plt


class LinearSVM:
    def __init__(self, features, labels):
        self.features = np.array(features)
        self.labels = np.array(labels)
        self.num_of_samples = self.features.shape[0]
        self.lagrange_multiplier = np.zeros(self.num_of_samples)
        self.b = 0.0  # 偏移量
        # Maintain a kkt list
        self.kkt_list = []
        self.rebuild_kkt_list()

    def kernel(self, a, b):
        return np.dot(a, b)

    def calculate_b(self):
        # Page 125 Formula 6.18
        b = 0.
        for i in range(self.num_of_samples):
            b += 1 / self.labels[i]
            tmp = 0
            for j in range(self.num_of_samples):
                if self.lagrange_multiplier[j] != 0:
                    tmp += self.lagrange_multiplier[j] * \
                        self.labels[j] * \
                        self.kernel(self.features[i], self.features[j])
            b -= tmp
        self.b = b / self.num_of_samples
        return self.b

    def update_kkt_list(self, update_list):
        for i in update_list:
            self.kkt_list[i] = self.check_kkt(
                self.features[i], self.labels[i], self.lagrange_multiplier[i])

    def rebuild_kkt_list(self):
        li = []
        for i in range(self.num_of_samples):
            li.append(self.check_kkt(self.features[i], self.labels[i], self.lagrange_multiplier[i]))
        self.kkt_list = li
        return li

    def calculate_e1_minus_e2(self, a1, a2):
        e1, e2 = self.predict(self.features[a1]) - self.labels[a1], \
                 self.predict(self.features[a2]) - self.labels[a2]
        return e1 - e2

    def smo_train(self, times):
        for _ in range(times):
            # Init a1 and a2
            a1 = -1
            a2 = -1
            # Select a1
            for i in range(self.num_of_samples):
                kkt, err = self.kkt_list[i]
                if not kkt:
                    a1 = i
                    break
            # If no a1, select a random number
            if a1 == -1:
                a1 = random.choice(range(self.num_of_samples))
            # Select a2 max(abs(e1 - e2))
            max_e1_minus_e2 = 0
            for i in range(self.num_of_samples):
                e1_minus_e2 = abs(self.calculate_e1_minus_e2(a1, i))
                if e1_minus_e2 > max_e1_minus_e2:
                    max_e1_minus_e2 = e1_minus_e2
                    a2 = i
            # If no choice, rand it
            while a2 == -1 or a2 == a1:
                a2 = random.choice(range(self.num_of_samples))
            # Page 125 Formula 6.16
            # Calculate const
            # const = self.lagrange_multiplier[a1] * self.labels[a1] + \
            #     self.lagrange_multiplier[a2] * self.labels[a2]
            # Calculate e1 and e2
            e1, e2 = self.predict(self.features[a1]) - self.labels[a1], \
                     self.predict(self.features[a2]) - self.labels[a2]
            # Calculate new a2
            g = self.kernel(self.features[a1], self.features[a1]) + \
                self.kernel(self.features[a2], self.features[a2]) - \
                self.kernel(self.features[a1], self.features[a2])
            a2_new_lm = self.lagrange_multiplier[a2] + self.labels[a2] * (e1 - e2) / g
            a1_new_lm = self.lagrange_multiplier[a1] + self.labels[a1] * self.labels[a2] * \
                        (self.lagrange_multiplier[a2] - a2_new_lm)
            # Update lagrange multiplier
            self.lagrange_multiplier[a1] = a1_new_lm
            self.lagrange_multiplier[a2] = a2_new_lm
            # print(a1_new_lm, a2_new_lm)
            self.update_kkt_list([a1, a2])
            self.calculate_b()
        return

    def check_kkt(self, feature, label, lm):
        # 检查 KKT 条件
        # lm: 拉格朗日乘子
        # Page 124
        # 此处未考虑松弛情况 C
        # 检查条件满足情况
        con1 = lm >= 0
        con2 = label * self.predict(feature) - 1 >= 0
        con3 = lm * (label * self.predict(feature) - 1) == 0
        # 检查偏移情况
        err = 0.
        if not con1:
            err += -lm
        if not con2:
            err += 1 - label * self.predict(feature)
        if not con3:
            err += abs(lm * (label * self.predict(feature) - 1))
        # (con1 and con2 and con3)
        # print(err)
        return con1 and con2 and con3, err

    def predict(self, feature):
        res = self.b
        for i in range(self.num_of_samples):
            if self.lagrange_multiplier[i] != 0:
                res += self.lagrange_multiplier[i] * \
                    self.labels[i] * self.kernel(self.features[i], feature)
        return res

    def analysis(self):
        correct = 0
        for i in range(self.num_of_samples):
            if self.predict(self.features[i]) * self.labels[i] > 0:
                correct += 1
        return correct / self.num_of_samples


def main():
    data = dt.generate_two_dimensional_linear_data_for_test()
    features = [x for x in itertools.chain(data[0], data[1])]
    labels = [-1 for x in range(len(data[0]))]
    labels += [1 for x in range(len(data[1]))]

    coll = [i for i in range(len(labels))]
    random.shuffle(coll)

    shuffle_features = []
    shuffle_labels = []
    for i in coll:
        shuffle_features.append(features[i])
        shuffle_labels.append(labels[i])
    linearSVM = LinearSVM(shuffle_features, shuffle_labels)

    linearSVM.smo_train(1000)
    print('Accuracy: %.2f%%' % (linearSVM.analysis() * 100))
    print(linearSVM.lagrange_multiplier)

    # Draw plot
    for i in range(len(shuffle_features)):
        color = 'green'
        if shuffle_labels[i] == 1:
            color = 'blue'
        if linearSVM.lagrange_multiplier[i] > 0:
            color = 'red'
        plt.scatter(shuffle_features[i][0], shuffle_features[i][1], color=color)
    plt.show()


if __name__ == '__main__':
    main()
