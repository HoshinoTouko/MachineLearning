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
        self.conditional_probs = None
    
    def train(self, discrete=True):
        mat_features = np.asmatrix(self.features)
        mat_features_T = mat_features.T
        mat_labels = np.asmatrix(self.labels)
        mat_labels_T = mat_labels.T
        _labels = np.array(mat_labels_T).flatten()
        # Find all posibility in features and labels
        _tmp = list(np.array(mat_features.T))
        _feature_posibilities = list(map(lambda x: sorted(list(set(x))), _tmp))
        # Count the numbers of labels
        label_choices, num_of_label = np.unique(_labels, return_counts=True)
        # Maintaince a complex dictionary to
        # represent the conditional probilities
        _conditional_probs = []
        for feature_num in range(len(_feature_posibilities)):
            _feature_coditional_prob = {}
            _features = np.array(mat_features_T)[feature_num]

            feature_choices = _feature_posibilities[feature_num]

            for _posibility in feature_choices:
                _feature_coditional_prob[_posibility] = {}
                for _label in label_choices:
                    match_features = np.where(_features==_posibility)[0]
                    match_labels = np.where(_labels==_label)[0]
                    _matches = set(match_features) & set(match_labels)
                    _feature_coditional_prob[_posibility][_label] = \
                        float(len(_matches)) / num_of_label[_label]

            _conditional_probs.append(_feature_coditional_prob)

        print('Finished')
        print(_conditional_probs)
        self.conditional_probs = _conditional_probs
    
    def predict(self, feature):
        if self.conditional_probs is None:
            raise Exception('Haven\'t been trained.')

        label_choices= np.unique(self.labels)
        prob_dict = {}
        for _label in label_choices:
            if _label not in prob_dict.keys():
                prob_dict[_label] = 1
            
            for feature_num in range(len(feature)):
                prob_dict[_label] *= \
                    self.conditional_probs[feature_num][feature[feature_num]][_label]

        return max(prob_dict, key=prob_dict.get)
    
    def analysis(self, features, labels):
        total = len(features)
        acc = 0.
        for feature_num in range(len(features)):
            if self.predict(features[feature_num]) == labels[feature_num]:
                acc += 1
        print('Analysis result: %.4f%%' % (100 * acc / total))


def test(discrete=True):
    data, _ = generate_bayes_fake_data_for_test(times=100, discrete=discrete)
    features = list(map(lambda x: x[0], data))
    labels = list(map(lambda x: x[1], data))

    bayes1 = NaiiveBayesClassifier(features, labels)
    bayes1.train(discrete=True)

    # Analysis
    data, _ = generate_bayes_fake_data_for_test(times=100, discrete=discrete)
    features = list(map(lambda x: x[0], data))
    labels = list(map(lambda x: x[1], data))

    bayes1.analysis(features, labels)


def test_continuous():
    test(discrete=False)


def test_discrete():
    test(discrete=True)

def main():
    test_discrete()
    test_continuous()

if __name__ == '__main__':
    main()
