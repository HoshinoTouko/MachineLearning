'''
File: bayes.py
Created Date: 06 Sep, 2018
Author: Touko Hoshino
-----
Copyright (c) 2018 Hoshino Touko
'''
from prettytable import PrettyTable
import random


def generate_bayes_fake_data(feature_list, positive_list, positive_prob, times=10):
    if len(feature_list) != len(positive_list):
        raise Exception('Length of data not match.')

    len_of_data = len(feature_list)
    res = []
    for _ in range(times):
        good = 0.
        _tmp = []
        for i in range(len_of_data):
            choice = random.choice(feature_list[i])
            _tmp.append(choice)
            if choice in positive_list[i]:
                good += 1
        res.append((_tmp, int(good / len_of_data > positive_prob)))
    return res

def generate_bayes_fake_data_for_test():
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
    return generate_bayes_fake_data(feature_list, positive_list, 0.5, 30)


def main():
    data = generate_bayes_fake_data_for_test()
    ptt = PrettyTable()
    ptt.field_names = ['Features', 'Labels']
    for d in data:
        ptt.add_row(d)
    print(ptt)

if __name__ == '__main__':
    main()
