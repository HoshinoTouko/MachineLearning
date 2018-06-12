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
import math

import SVM.dataset as dt
import numpy as np

import matplotlib.pyplot as plt


class GaussSVM:
    def __init__(self, features, labels, c=10E100):
        self.features = np.array(features)
        self.labels = np.array(labels)
        self.num_of_samples = self.features.shape[0]
        self.lagrange_multiplier = np.zeros(self.num_of_samples)
        self.b = 0.  # 偏移量
        self.c = c
        # Maintain a kkt list
        self.kkt_list = []
        self.rebuild_kkt_list()
        # Maintain a pre-predict list
        self.pre_predict = []
        self.rebuild_pre_predict_list()

    def kernel(self, a, b, o=35):
        tmp = a - b
        norm = np.dot(tmp, tmp)
        return math.exp(- norm ** 2 / (2 * o * o))

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

    def calculate_e(self, a):
        return self.pre_predict[a] - self.labels[a]
        # return self.predict(self.features[a]) - self.labels[a]

    def rebuild_pre_predict_list(self):
        for i in range(self.num_of_samples):
            self.pre_predict.append(self.predict(self.features[i]))

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
        e1, e2 = self.calculate_e(a1), self.calculate_e(a2)
        return e1 - e2

    def smo_train(self, times=100, max_acc=0.9):
        last_acc = 0
        plot_list = []
        last_a2 = 0  # Prevent repeat a2
        for _time in range(times):
            # Init a1 and a2
            a1 = -1
            a2 = -1
            # Select a1
            for i in range(self.num_of_samples):
                kkt = self.kkt_list[i]
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
            while a2 == -1 or a2 == a1 or a2 == last_a2:
                a2 = random.choice(range(self.num_of_samples))
            last_a2 = a2
            # Page 125 Formula 6.16
            # Calculate const
            # const = self.lagrange_multiplier[a1] * self.labels[a1] + \
            #     self.lagrange_multiplier[a2] * self.labels[a2]
            # Calculate e1 and e2
            e1, e2 = self.pre_predict[a1] - self.labels[a1], \
                     self.pre_predict[a2] - self.labels[a2]
            # e1, e2 = self.predict(self.features[a1]) - self.labels[a1], \
            #          self.predict(self.features[a2]) - self.labels[a2]
            # Calculate new a2
            g = self.kernel(self.features[a1], self.features[a1]) + \
                self.kernel(self.features[a2], self.features[a2]) - \
                self.kernel(self.features[a1], self.features[a2])
            a2_new_lm = self.lagrange_multiplier[a2] + self.labels[a2] * (e1 - e2) / g
            # Calculate upper and lower
            if self.labels[a1] == self.labels[a2]:
                lower = max(0, self.lagrange_multiplier[a2] + self.lagrange_multiplier[a1] - self.c)
                upper = min(self.c, self.lagrange_multiplier[a2] + self.lagrange_multiplier[a1])
            else:
                lower = max(0, self.lagrange_multiplier[a2] - self.lagrange_multiplier[a1])
                upper = min(self.c, self.c + self.lagrange_multiplier[a2] - self.lagrange_multiplier[a1])
            if a2_new_lm > upper:
                a2_new_lm = upper
            elif a2_new_lm < lower:
                a2_new_lm = lower
            # Calculate a1
            a1_new_lm = self.lagrange_multiplier[a1] + self.labels[a1] * self.labels[a2] * \
                        (self.lagrange_multiplier[a2] - a2_new_lm)
            # Update new_b
            b1 = self.b - self.calculate_e(a1) - \
                self.labels[a1] * (a1_new_lm - self.lagrange_multiplier[a1]) * \
                self.kernel(self.features[a1], self.features[a1]) - \
                self.labels[a2] * (a2_new_lm - self.lagrange_multiplier[a2]) * \
                self.kernel(self.features[a1], self.features[a2])
            b2 = self.b - self.calculate_e(a2) - \
                self.labels[a1] * (a1_new_lm - self.lagrange_multiplier[a1]) * \
                self.kernel(self.features[a1], self.features[a2]) - \
                self.labels[a2] * (a2_new_lm - self.lagrange_multiplier[a2]) * \
                self.kernel(self.features[a2], self.features[a2])
            if 0 < a2_new_lm < self.c:
                new_b = b2
            elif 0 < a1_new_lm < self.c:
                new_b = b1
            else:
                new_b = (b1 + b2) / 2
            # self.calculate_b()
            # Update pre_predict list
            for i in range(len(self.pre_predict)):
                # Update b
                self.pre_predict[i] += (new_b - self.b)
                self.pre_predict[i] += (a1_new_lm - self.lagrange_multiplier[a1]) * \
                    self.labels[a1] * self.kernel(self.features[i], self.features[a1])
                self.pre_predict[i] += (a2_new_lm - self.lagrange_multiplier[a2]) * \
                    self.labels[a2] * self.kernel(self.features[i], self.features[a2])
            # Update lagrange multiplier
            self.lagrange_multiplier[a1] = a1_new_lm
            self.lagrange_multiplier[a2] = a2_new_lm
            # print(a1_new_lm, a2_new_lm)
            self.update_kkt_list([a1, a2])
            # Interrupt condition
            acc = self.analysis()
            # Record acc
            plot_list.append(acc)
            if acc > last_acc:
                last_acc = acc
            else:
                if acc > max_acc:
                    print('Ended prematurely, iter time: %d' % _time)
                    print('Iter time: %d, acc: %.2f%%' % (_time, acc * 100))
                    break
            # Show train time
            if _time % 10 == 0:
                print('Iter time: %d, acc: %.2f%%' % (_time, acc * 100))
        plt.plot(plot_list)
        plt.title('Accuracy plot.')
        plt.show()
        return

    def check_kkt(self, feature, label, lm):
        # 检查 KKT 条件
        # lm: 拉格朗日乘子
        # Page 124
        ui = self.predict(feature)
        con1 = 0 <= lm <= self.c
        con2 = label * ui - 1 >= 0
        con3 = lm * (label * ui - 1) == 0
        return con1 and con2 and con3

    def predict(self, feature):
        res = self.b
        for i in range(self.num_of_samples):
            if self.lagrange_multiplier[i] != 0:
                res += self.lagrange_multiplier[i] * \
                    self.labels[i] * self.kernel(self.features[i], feature)
        return res

    def analysis(self, features=None, labels=None):
        correct = 0
        # print(
        #     'Num of lagrange multiplier %d' %
        #     len(self.lagrange_multiplier[np.where(self.lagrange_multiplier > 0)])
        # )
        if features is None:
            for i in range(self.num_of_samples):
                if self.pre_predict[i] * self.labels[i] > 0:
                    correct += 1
            return correct / self.num_of_samples
        # test
        for i in range(len(labels)):
            if self.predict(features[i]) * labels[i] > 0:
                correct += 1
        print(
            'Num of lagrange multiplier %d' %
            len(self.lagrange_multiplier[np.where(self.lagrange_multiplier > 0)])
        )
        return correct / len(labels)


def main():
    data = dt.generate_concentric_circles()
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
    gaussSVM = GaussSVM(shuffle_features, shuffle_labels)

    gaussSVM.smo_train(500, 0.97)
    print('Accuracy: %.2f%%' % (gaussSVM.analysis() * 100))
    # print(gaussSVM.lagrange_multiplier)

    # Draw plot
    for i in range(len(shuffle_features)):
        color = 'green'
        if shuffle_labels[i] == 1:
            color = 'blue'
        if gaussSVM.lagrange_multiplier[i] > 0:
            color = 'red'
        plt.scatter(shuffle_features[i][0], shuffle_features[i][1], color=color)
    plt.show()


if __name__ == '__main__':
    main()
