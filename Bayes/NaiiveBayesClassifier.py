'''
File: naiiveBayesClassifier.py
Created Date: 06 Sep, 2018
Author: Touko Hoshino
-----
Copyright (c) 2018 Hoshino Touko
'''
import numpy as np

from data.fake.bayes import generate_bayes_fake_data_for_test, run_solution


class NaiiveBayesClassifier:
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    
    def train(self):
        mat_features = np.asmatrix(self.features)
        mat_features_T = mat_features.T
        mat_labels = np.asmatrix(self.labels)
        mat_labels_T = mat_labels.T
        # First step: find all posibility in the features
        _tmp = list(np.array(mat_features.T))
        _feature_posibilities = list(map(lambda x: sorted(list(set(x))), _tmp))
        # print(_feature_posibilities)
        # Second step: maintaince a complex dictionary
        _conditional_probs = []
        for feature_num in range(len(_feature_posibilities)):
            _feature_coditional_prob = {}
            _features = np.array(mat_features_T)[feature_num]
            _labels = np.array(mat_labels_T).flatten()

            feature_choices = _feature_posibilities[feature_num]
            for _posibility in feature_choices:
                prob = _labels[np.where(_features==_posibility)].mean()
                _feature_coditional_prob[_posibility] = prob
            _conditional_probs.append(_feature_coditional_prob)
        print(_conditional_probs)


def main():
    data, solution = generate_bayes_fake_data_for_test()
    features = list(map(lambda x: x[0], data))
    labels = list(map(lambda x: x[1], data))

    bayes1 = NaiiveBayesClassifier(features, labels)
    bayes1.train()

if __name__ == '__main__':
    main()
