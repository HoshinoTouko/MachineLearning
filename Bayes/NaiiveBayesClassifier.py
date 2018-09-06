'''
File: naiiveBayesClassifier.py
Created Date: 06 Sep, 2018
Author: Touko Hoshino
-----
Copyright (c) 2018 Hoshino Touko
'''
import numpy as np

from data.fake.bayes import generate_bayes_discrete_fake_data_for_test, run_solution


class NaiiveBayesClassifier:
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        self.conditional_probs = None
    
    def discrete_train(self):
        mat_features = np.asmatrix(self.features)
        mat_features_T = mat_features.T
        mat_labels = np.asmatrix(self.labels)
        mat_labels_T = mat_labels.T
        # First step: find all posibility in the features
        _tmp = list(np.array(mat_features.T))
        _feature_posibilities = list(map(lambda x: sorted(list(set(x))), _tmp))
        # print(_feature_posibilities)
        # Second step: maintaince a complex dictionary to
        # represent the conditional probilities
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
        print('Finished')
        print(_conditional_probs)
        self.conditional_probs = _conditional_probs
    
    def predict(self, feature):
        if self.conditional_probs is None:
            raise Exception('No train')
        
        positive_prob = 1
        negetive_prob = 1
        for feature_num in range(len(feature)):
            _prob = self.conditional_probs[feature_num][feature[feature_num]]
            positive_prob *= _prob
            negetive_prob *= (1 - _prob)
        return int(positive_prob >= negetive_prob)
    
    def analysis(self, features, labels):
        total = len(features)
        acc = 0.
        for feature_num in range(len(features)):
            if self.predict(features[feature_num]) == labels[feature_num]:
                acc += 1
        print('Analysis result: %.4f%%' % (100 * acc / total))


def test_discrete():
    data, _ = generate_bayes_discrete_fake_data_for_test(100)
    features = list(map(lambda x: x[0], data))
    labels = list(map(lambda x: x[1], data))

    bayes1 = NaiiveBayesClassifier(features, labels)
    bayes1.discrete_train()

    # Analysis
    data, _ = generate_bayes_discrete_fake_data_for_test(100)
    features = list(map(lambda x: x[0], data))
    labels = list(map(lambda x: x[1], data))

    bayes1.analysis(features, labels)


def test_continuous():
    pass


def main():
    test_discrete()
    test_continuous()

if __name__ == '__main__':
    main()
