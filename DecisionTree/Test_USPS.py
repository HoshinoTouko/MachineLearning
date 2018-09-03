'''
@File: Test.py
@Author: HoshinoTouko
@License: (C) Copyright 2014 - 2017, HoshinoTouko
@Contact: i@insky.jp
@Website: https://touko.moe/
@Create at: 2018-03-28 14:31
@Desc: 
'''
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from IrisDecisionTree import SklearnDefaultDecisionTree
from IrisDecisionTree.HandwriteDecisionTree import build_tree, predict, predict_all

import os
import numpy as np

# Train data
fea_path = os.path.join(os.path.dirname(__file__), '../dataset/uspst_uni/train_features.txt')
labels_path = os.path.join(os.path.dirname(__file__), '../dataset/uspst_uni/train_label.txt')

usps_features_lines = open(fea_path).readlines()
usps_labels_lines = open(labels_path).readlines()

usps_features_list = []
usps_labels = np.zeros(len(usps_labels_lines))

for i in range(len(usps_features_lines)):
    usps_features_list.append(np.array(list(map(
        float,
        usps_features_lines[i].split('\t')
    ))))
    usps_labels[i] = int(usps_labels_lines[i].strip())

usps_features = np.array(usps_features_list)

# print(usps_features.shape)
# print(usps_labels.shape)

# Load test data
test_fea_path = os.path.join(os.path.dirname(__file__), '../dataset/uspst_uni/test_features.txt')
test_usps_features_lines = open(test_fea_path).readlines()
usps_test_dataset = []

for i in range(len(test_usps_features_lines)):
    usps_test_dataset.append(np.array(list(map(
        float,
        test_usps_features_lines[i].split('\t')
    ))))
usps_test_dataset = np.array(usps_test_dataset)

# Split train and test dataset
feature_train, feature_test, target_train, target_test = train_test_split(
    usps_features, usps_labels, test_size=0.3, random_state=42
)

if __name__ == '__main__':
    # Scikit-learn default decision tree.
    print('SklearnDefaultDecisionTree.predict')
    predict_results_skl = SklearnDefaultDecisionTree.predict(
        feature_train, target_train, feature_test
    )

    scores_skl = accuracy_score(predict_results_skl, target_test)
    print(scores_skl)

    # HandwriteDecisionTree.predict
    print('HandwriteDecisionTree.predict')
    print('Training.......')
    root_node = build_tree(
        feature_train, target_train, 1
    )
    print('Testing.......')
    predict_results_hdw = predict_all(root_node, feature_test)

    scores_hdw = accuracy_score(predict_results_hdw, target_test)
    print(scores_hdw)

    # Predict test dataset by hdw
    test_predict_results_hdw = predict_all(root_node, usps_test_dataset)
    test_res_path = os.path.join(os.path.dirname(__file__), '../dataset/uspst_uni/test_res.txt')
    with open(test_res_path, 'w+') as fi:
        fi.write(
            '\n'.join(map(lambda x: str(int(x)), test_predict_results_hdw))
        )

    # Predict test dataset by default decision tree
    predict_test_res_skl = SklearnDefaultDecisionTree.predict(
        feature_train, target_train, usps_test_dataset
    )
    test_sklearn_tree_res_path = os.path.join(
        os.path.dirname(__file__),
        '../dataset/uspst_uni/test_sklearn_tree_res.txt'
    )
    with open(test_sklearn_tree_res_path, 'w+') as fi:
        fi.write(
            '\n'.join(map(lambda x: str(int(x)), predict_test_res_skl))
        )

    # Difference rate between them
    print('Difference rate between them')
    print(accuracy_score(test_predict_results_hdw, predict_test_res_skl))
