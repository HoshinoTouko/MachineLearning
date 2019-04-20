"""
@File: HiddenMarkovModel.py
@Author: HoshinoTouko
@License: (C) Copyright 2014 - 2019, HoshinoTouko
@Contact: i@insky.jp
@Website: https://touko.moe/
@Created at: 4/20/2019 22:47
@Desc: 
"""


def forward_alg(
    mat_a: list, mat_b: list, mat_pi: list,
    prescribed_seq: list
) -> float:

    n: int = len(mat_a)
    seq_length: int = len(prescribed_seq)

    def alpha(order: int, status: int) -> float:
        seq_item = prescribed_seq[order]
        if order == 1:
            return mat_pi[status] * mat_b[status][seq_item]

        alp_prob: float = 0.
        for i in range(n):
            alp_prob += alpha(order-1, i) * mat_a[i][status]
        return alp_prob * mat_b[status][seq_item]

    prob: int = 0.
    for i in range(n):
        prob += alpha(seq_length-1, i)

    return prob


def main():
    mat_a_label = ['Sunny', 'Cloudy', 'Windy']
    mat_b_label = ['Hot', 'Cold', 'Wet']
    mat_a = [
        [0.8, 0.1, 0.1],
        [0.3, 0.4, 0.3],
        [0.4, 0.2, 0.4],
    ]
    mat_b = [
        [0.8, 0.1, 0.1],
        [0.2, 0.6, 0.2],
        [0.1, 0.3, 0.6],
    ]
    mat_pi = [30/47, 9/47, 8/47]

    prescribed_seq = [0, 0, 1, 2, 1, 2, 1, 0]
    print('Prescribed Sequence: %s' % list(map(
        lambda x: mat_b_label[x], prescribed_seq
    )))

    forward_res = forward_alg(
        mat_a, mat_b, mat_pi, prescribed_seq)
    print('Forward Algorithm Result(Prob): %s' % forward_res)


if __name__ == '__main__':
    main()
