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
        if order == 0:
            return mat_pi[status] * mat_b[seq_item][status]

        alp_prob: float = 0.
        for i in range(n):
            alp_prob += alpha(order-1, i) * mat_a[i][status]
        return alp_prob * mat_b[seq_item][status]

    prob: int = 0.
    for ii in range(n):
        prob += alpha(seq_length-1, ii)

    return prob


def viterbi(
        mat_a: list, mat_b: list, mat_pi: list,
        prescribed_seq: list,
) -> list:

    def argmax(li: list) -> int:
        return max(range(len(li)), key=lambda x: li[x])

    n: int = len(mat_a)
    seq_length: int = len(prescribed_seq)

    prob_seq = []
    phi_seq = []
    # First item
    tmp_prob_seq = []
    tmp_phi_seq = []
    for status_id in range(n):
        tmp_prob_seq.append(
            mat_pi[status_id] * mat_b[prescribed_seq[0]][status_id])
        tmp_phi_seq.append(0)
    else:
        prob_seq.append(tmp_prob_seq)
        phi_seq.append(tmp_phi_seq)

    for seq_item in range(1, seq_length):
        tmp_prob_seq = []
        tmp_phi_seq = []
        for status_id in range(n):
            all_prob_seq = [
                prob_seq[seq_item - 1][_i] *
                mat_a[_i][status_id]
                for _i in range(n)
            ]
            tmp_prob_seq.append(
                max(all_prob_seq) *
                mat_b[prescribed_seq[seq_item]][status_id]
            )
            tmp_phi_seq.append(argmax(all_prob_seq))
        else:
            prob_seq.append(tmp_prob_seq)
            phi_seq.append(tmp_phi_seq)

    q = [argmax(prob_seq[seq_length-1])]
    for i in range(seq_length-1):
        q.append(phi_seq[seq_length-i-1][q[i]])
    else:
        q.reverse()

    return q


def main():
    mat_a_label = ['Sunny', 'Cloudy', 'Windy']
    mat_b_label = ['Hot', 'Cold', 'Wet']
    mat_a = [
        [0.8, 0.1, 0.1],
        [0.3, 0.4, 0.3],
        [0.4, 0.2, 0.4],
    ]
    mat_b = [
        [0.8, 0.2, 0.1],
        [0.1, 0.5, 0.2],
        [0.1, 0.3, 0.7],
    ]
    mat_pi = [30/47, 9/47, 8/47]

    prescribed_seq = [0, 0, 1, 2, 1, 2, 1, 0]
    print('Prescribed Sequence: %s' % list(map(
        lambda x: mat_b_label[x], prescribed_seq
    )))

    forward_res = forward_alg(
        mat_a, mat_b, mat_pi, prescribed_seq
    )
    print('Forward Algorithm Result(Prob): %s' % forward_res)

    viterbi_res = viterbi(
        mat_a, mat_b, mat_pi, prescribed_seq
    )
    print('Viterbi Result(Prob): %s' % list(map(
        lambda x: mat_a_label[x], viterbi_res
    )))


if __name__ == '__main__':
    main()
