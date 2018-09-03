'''
@File: Utils.py
@Author: HoshinoTouko
@License: (C) Copyright 2014 - 2017, HoshinoTouko
@Contact: i@insky.jp
@Website: https://touko.moe/
@Create at: 2018-03-28 15:14
@Desc: 
'''
import numpy as np


def calculate_ent(target):
    items = np.unique(target)

    item_size = items.size
    count = np.zeros(item_size)

    for i in range(item_size):
        count[i] = target[target == items[i]].size

    p_i = np.divide(count, target.size)

    ent = 0
    for i in range(item_size):
        ent = ent - p_i[i] * np.log2(p_i[i])

    return ent


def calculate_condition_entropy(feature, condition, target):
    target_true = target[condition(feature)]
    target_reverse = target[condition(feature) == False]

    p_true = target_true.size / target.size
    p_false = 1 - p_true

    ent = p_true * calculate_ent(target_true) + \
          p_false * calculate_ent(target_reverse)

    return ent


def find_best_feature(feature, target):
    min_entropy = float('inf')
    min_point = 0
    points = generate_feature_points(feature, target)
    for p in points:
        entropy = calculate_condition_entropy(feature, lambda f: f < p, target)
        if entropy < min_entropy:
            min_entropy = entropy
            min_point = p

    if points.size == 0:
        min_entropy = 0

    return min_point, min_entropy


def generate_feature_points(feature, target):
    argsort = feature.argsort()

    feature_after_argsort = feature[argsort]
    target_after_argsort = target[argsort]

    # print((feature_after_argsort, target_after_argsort))

    last_value = target[0]
    split_value = []

    for i in range(target_after_argsort.size):
        #print(last_value, target_after_argsort)
        if last_value != target_after_argsort[i]:
            split_value.append((feature_after_argsort[i] + feature_after_argsort[i - 1]) / 2)
            last_value = target_after_argsort[i]

    return np.array(split_value)


def select_feature(features, target):

    min_entropy = float('inf')
    min_point = 0
    num_of_features = features.shape[1]

    index = 0
    for i in range(num_of_features):
        point, entropy = find_best_feature(features[:, i], target)
        if entropy <= min_entropy:
            index = i
            min_point = point
            min_entropy = entropy

    return index, min_point, min_entropy


if __name__ == '__main__':
    print(calculate_ent(np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])))