'''
File: bayes.py
Created Date: 06 Sep, 2018
Author: Touko Hoshino
-----
Copyright (c) 2018 Hoshino Touko
'''
from prettytable import PrettyTable
import random
import numpy as np


def generate_bayes_fake_data(solution, times=10, discrete=True):
    # Resolve solution
    feature_list = solution['feature_list']
    # positive_list = solution['positive_list']
    # positive_prob = solution['positive_prob']

    len_of_data = len(feature_list)
    res = []
    for _ in range(times):
        _tmp_data = []
        for i in range(len_of_data):
            if discrete:
                choice = random.choice(feature_list[i])
            else:
                choice = random.uniform(min(feature_list[i]), max(feature_list[i]))
            _tmp_data.append(choice)
        res.append((_tmp_data, run_solution(solution, _tmp_data, discrete)))
    return res, solution


def run_solution(solution, data, discrete=True):
    # feature_list = solution['feature_list']
    positive_list = solution['positive_list']
    positive_prob = solution['positive_prob']

    good = 0.
    for i in range(len(data)):
        if discrete:
            if data[i] in positive_list[i]:
                good += 1
        else:
            if data[i] > np.average(positive_list[i]):
                good += 1
    return int(good / len(positive_list) > positive_prob)


def generate_bayes_fake_data_for_test(times=30, discrete=True):
    feature_list = [
        [1, 2, 3, 4],
        [1, 2, 3],
        [2, 3, 4, 5],
        [1, 2],
    ]
    positive_list = [
        [1, 2],
        [1],
        [3, 4, 5],
        [2],
    ]
    positive_prob = 0.5
    solution = {
        'feature_list': feature_list, 
        'positive_list': positive_list, 
        'positive_prob': positive_prob
    }
    return generate_bayes_fake_data(solution, times, discrete)


def main():
    data, solution = generate_bayes_fake_data_for_test(discrete=True)
    ptt = PrettyTable()
    ptt.field_names = ['Features', 'Labels']
    for d in data:
        ptt.add_row(d)
    print(ptt)
    print('Solution')
    print(solution)

if __name__ == '__main__':
    main()
