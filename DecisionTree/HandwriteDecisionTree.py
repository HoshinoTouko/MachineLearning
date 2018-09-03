'''
@File: HandwriteDecisionTree.py
@Author: HoshinoTouko
@License: (C) Copyright 2014 - 2017, HoshinoTouko
@Contact: i@insky.jp
@Website: https://touko.moe/
@Create at: 2018-03-28 14:48
@Desc: 
'''
from IrisDecisionTree.Utils import *

import numpy as np


class TreeNode:
    idn = 0
    feature_index = ''
    feature_point = 0
    feature_entropy = 0
    target_label = ''
    true_node = None
    false_node = None

    def decision1(self, feature):
        return feature < self.feature_point

    @staticmethod
    def decision(feature, point):
        return feature < point


def build_tree(features, target, idn):
    # Root node 
    node = TreeNode()

    # Find minimum ent and the splitting point
    index, point, entropy = select_feature(features, target)
    node.idn = idn
    node.feature_index = index
    node.feature_point = point
    node.feature_entropy = entropy

    # Set most frequent number as the node's output
    try:
        node.target_label = np.argmax(np.bincount(target.astype(np.int64)))
    except Exception as e:
        print(e)
        node.target_label = -1

    '''
    print(
        'build tree node id %d, index %d, point %f, entropy %f, label %s ' %
        (idn, index, point, entropy, node.target_label)
    )
    '''

    # End create node when ent < 0.1
    if entropy < 0.2:
        # print('too low entropy : ', entropy)
        return node

    f_copy = features.copy()
    t_copy = target.copy()
    f = f_copy[:, index]
    selector = node.decision(f, point)

    '创建左右两个子节点'
    idn = idn + 1
    node.true_node = build_tree(f_copy[selector, :], t_copy[selector], idn)
    idn = node.true_node.idn + 1
    node.false_node = build_tree(f_copy[selector == False], t_copy[selector == False], idn)

    return node


def predict(node, features):
    tmp_node = node
    while True:
        if tmp_node.true_node and tmp_node.false_node:
            tmp_node = tmp_node.true_node \
                if tmp_node.decision1(features[tmp_node.feature_index]) \
                else tmp_node.false_node
        else:
            return tmp_node.target_label


def predict_all(node, features):
    result = []
    for feature in features:
        result.append(predict(node, feature))

    return np.array(result)
