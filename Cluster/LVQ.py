"""
@File: KMeans.py
@Author: HoshinoTouko
@License: (C) Copyright 2014 - 2019, HoshinoTouko
@Contact: i@insky.jp
@Website: https://touko.moe/
@Created at: 3/12/2019 21:02
@Desc: 
"""
from collections import Counter
from sklearn.model_selection import train_test_split

from data.fake.cluster import simple_data

import math
import random

import numpy as np


class LVQCluster:
    def __init__(self, features, labels, learn_rate):
        self.features = features
        self.labels = labels
        self.learn_rate = learn_rate
        self.category = len(list(set(labels)))
        self.prototype_vectors = None

    @classmethod
    def get_dist(cls, a, b):
        if len(a) != len(b):
            raise Exception('%s & %s length not equal' % (a, b))
        count = 0
        for i in range(len(a)):
            count += pow(a[i] - b[i], 2)
        return math.sqrt(count)

    def train(self, max_time=100):
        data_count = len(self.features)

        category_identifier = {}
        for label_id in range(len(self.labels)):
            label = self.labels[label_id]
            if label not in category_identifier.keys():
                category_identifier[label] = []
            category_identifier[label].append(label_id)

        prototype_vectors = {
            k: self.features[random.sample(v, 1)[0]]
            for k, v in category_identifier.items()
        }

        for time in range(max_time):
            selected_id = random.randint(0, data_count - 1)
            selected_feature = self.features[selected_id]
            selected_label = self.labels[selected_id]

            min_dist = float('inf')
            min_dist_label = 0
            for label, vector in prototype_vectors.items():
                dist = self.get_dist(vector, selected_feature)
                if dist < min_dist:
                    min_dist = dist
                    min_dist_label = label

            rate = self.learn_rate \
                if min_dist_label == selected_label else -self.learn_rate

            old_vector = np.array(prototype_vectors[min_dist_label])
            prototype_vectors[min_dist_label] = (
                old_vector + rate * (np.array(selected_feature) - old_vector)
            ).tolist()
        self.prototype_vectors = prototype_vectors

    def predict(self, feature):
        prototype_vectors = self.prototype_vectors
        min_dist = float('inf')
        min_dist_label = None
        for label, _feature in prototype_vectors.items():
            dist = self.get_dist(feature, _feature)
            if dist < min_dist:
                min_dist = dist
                min_dist_label = label
        return min_dist_label

    def analyse(self, features, labels):
        # Pre resolve data
        total = len(features)
        acc_count = 0

        for i in range(total):
            if self.predict(features[i]) == labels[i]:
                acc_count += 1

        print(
            'Test features: %s, accuracy rate: %.2f%%' %
            (total, acc_count / total * 100)
        )


def main():
    print('Generate data...')
    features, labels = simple_data.generate_data()

    features_train, features_test, labels_train, labels_test = train_test_split(
        features, labels, test_size=0.9)

    k_means = LVQCluster(features_train, labels_train, 0.1)
    print('Training...')
    k_means.train()
    print('Trained.')

    print('Testing...')
    k_means.analyse(features_test, labels_test)


if __name__ == '__main__':
    main()

