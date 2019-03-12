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


class KMeansCluster:
    def __init__(self, features, category):
        self.features = features
        self.category = category
        self.mean_vectors = None

    @classmethod
    def get_dist(cls, a, b):
        if len(a) != len(b):
            raise Exception('%s & %s length not equal' % (a, b))
        count = 0
        for i in range(len(a)):
            count += pow(a[i] - b[i], 2)
        return math.sqrt(count)

    def train(self, max_time=1000):
        mean_vectors = random.sample(self.features, self.category)

        for time in range(max_time):
            sorted_features = [[] for _ in range(self.category)]
            for feature_id in range(len(self.features)):
                min_dist = float('inf')
                min_dist_category = 0
                for i in range(self.category):
                    dist = self.get_dist(self.features[feature_id], mean_vectors[i])
                    if dist < min_dist:
                        min_dist = dist
                        min_dist_category = i
                sorted_features[min_dist_category].append(feature_id)

            # Calculate mean vector
            changed = False
            for i in range(self.category):
                all_vec = np.matrix([
                    self.features[x]
                    for x in sorted_features[i]
                ])
                mean_vector = (
                        np.sum(all_vec, axis=0) /
                        len(sorted_features[i])
                ).tolist()[0]
                if mean_vector != mean_vectors[i]:
                    changed = True
                    mean_vectors[i] = mean_vector
            if not changed:
                self.mean_vectors = mean_vectors
                print('Mean vector not changed, trained %s times.' % time)
                return

    def predict(self, feature):
        mean_vectors = self.mean_vectors
        min_dist = float('inf')
        min_dist_category = 0
        for i in range(self.category):
            dist = self.get_dist(feature, mean_vectors[i])
            if dist < min_dist:
                min_dist = dist
                min_dist_category = i
        return min_dist_category

    def analyse(self, features, labels):
        # Pre resolve data
        resolved_data = {}
        for i in range(len(features)):
            if labels[i] not in resolved_data.keys():
                resolved_data[labels[i]] = []
            resolved_data[labels[i]].append(features[i])

        total = 0
        acc_count = 0
        for label, features in resolved_data.items():
            res = [
                self.predict(feature)
                for feature in features
            ]
            total += len(features)
            acc_count += Counter(res).most_common(1)[0][1]

        print(
            'Test features: %s, accuracy rate: %.2f%%' %
            (total, acc_count / total * 100)
        )


def main():
    print('Generate data...')
    features, labels = simple_data.generate_data()

    features_train, features_test, labels_train, labels_test = train_test_split(
        features, labels, test_size=0.9)

    k_means = KMeansCluster(features_train, 3)
    print('Training...')
    k_means.train()
    print('Trained.')

    print('Testing...')
    k_means.analyse(features_test, labels_test)


if __name__ == '__main__':
    main()

