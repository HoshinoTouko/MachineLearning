'''
File: LogisticRegression.py
Created Date: 04 Sep, 2018
Author: Touko Hoshino
-----
Copyright (c) 2018 Hoshino Touko
'''
import numpy as np

from data.fake.classification import generate_two_dimensional_linear_data_for_test

class LogisticRegression:
    def __init__(self, features, labels):
        self._features = np.array(features)
        self.features = np.c_[features, np.ones([len(features)])]
        self.labels = np.array(labels)

        self.w = np.zeros([len(features[0]) + 1])

    def predict(self, x, target=0, w=None):
        if w is None:
            w = self.w
        w = np.asmatrix(w)

        tmp = np.array(np.exp(np.dot(w, x)))[0][0]
        if target == 0:
            return 1 / (1 + tmp)
        return tmp / (1 + tmp)

    # Newton_method
    def calculate_first_derivative(self):
        sum = np.zeros([len(self.w)])
        for i in range(len(self.features)):
            sum += self.features[i] * (self.predict(self.features[i], 1) - self.labels[i])
        return sum
    
    def calculate_second_derivative(self):
        sum = 0
        for i in range(len(self.features)):
            mat_x = np.asmatrix(self.features[i])
            xxt = np.array(np.dot(mat_x, mat_x.T))[0]
            p1 = self.predict(self.features[i], 1)
            sum += (xxt * p1 * (1 - p1))
        return sum

    def train_by_newton_method(self, times=20):
        for _time in range(times):
            self.w = self.w - \
                self.calculate_first_derivative() / self.calculate_second_derivative()
            # print(self.w)
            if _time % 20 == 0:
                analysis_res = self.analysis() * 100
                print('Train %d times and get the accuracy %.4f%%' % 
                    (_time, analysis_res)
                )
                if analysis_res > 90:
                    print('End at %.4f' % analysis_res)
                    break

    # Analysis
    def analysis(self, features=None, labels=None, w=None):
        if features is None:
            features = self.features
        if labels is None:
            labels = self.labels
        if w is None:
            w = self.w
        
        acc = 0
        total = float(len(labels))
        for i in range(len(features)):
            p0 = self.predict(features[i], w=w)
            if p0 >= 0.5 and labels[i] == 0:
                acc += 1
            elif p0 < 0.5 and labels[i] == 1:
                acc += 1
            else:
                pass
        return acc / total


def main():
    pre_data = generate_two_dimensional_linear_data_for_test()
    features = []
    labels = []
    for t in pre_data[0]:
        features.append(t)
        labels.append(0)
    for t in pre_data[1]:
        features.append(t)
        labels.append(1)
    
    logisticRegression = LogisticRegression(features, labels)
    logisticRegression.train_by_newton_method(1000)
    logisticRegression.analysis()


if __name__ == '__main__':
    main()
