import numpy as np

from util_tsne_exact import (sum_operations, sum_operations_by_function,
                             sum_operations_by_measure)

FUNCTION_NAMES = ("normalize", "compute_pairwise_affinity_perplexity",
                  "symmetrize_affinities", "early_exageration",
                  "compute_low_dimensional_affinities", "create_tree",
                  "gradient_computation", "gradient_update", "normalize_2")


def read_iters(file_name):
    it = []
    it_ec = []
    it_buildFromPoints = []
    it_ins = []
    it_sub = []

    with open(file_name, 'r') as f:
        for line in f:
            tokens = line.strip().split(' ')
            if tokens[0] == 'it':
                it.append(int(tokens[1]))
            elif tokens[0] == 'it_ec':
                it_ec.append(int(tokens[1]))
            elif tokens[0] == 'it_buildFromPoints':
                it_buildFromPoints.append(int(tokens[1]))
            elif tokens[0] == 'it_ins':
                it_ins.append(int(tokens[1]))
            elif tokens[0] == 'it_sub':
                it_sub.append(int(tokens[1]))

    it = np.array(it, dtype="float")
    it_ec = np.array(it_ec, dtype="float")
    it_buildFromPoints = np.array(it_buildFromPoints, dtype="float")
    it_ins = np.array(it_ins, dtype="float")
    it_sub = np.array(it_sub, dtype="float")

    return it, it_ec, it_buildFromPoints, it_ins, it_sub


def read_benchmark_bh(file_prefix, start=200, stop=4000, interval=200):
    input_size = np.arange(start, stop + 1, interval)
    it, it_ec, it_buildFromPoints, it_ins, it_sub = read_iters(
        file_prefix + "iters.txt")
    cycles = np.loadtxt(file_prefix + "cycles.txt")
    assert np.all(cycles[:, :-1].sum(axis=1) == cycles[:, -1])

    N = input_size.astype("float")
    D = 28 * 28
    d = 2
    T = 1000  # Number of iterations
    K = 3 * 50  #conserider neighbors

    count_measure = ("add", "mult", "div", "exp", "log")

    flops = {
        "normalize": {
            "add": 2 * N * D,
            "div": D + N * D
        },
        "compute_pairwise_affinity_perplexity": {
            "add": it * (2 * K + 2) + it_ec * 2 * D,
            "mult": it * (3 * K) + it_ec * D + it_buildFromPoints,
            "div": it * 2 + N * K + it_buildFromPoints,
            "exp": it * K,
            "log": it * 2,
            "sqrt": it_ec
        },
        "symmetrize_affinities": {
            "add": N * K * 3,
            "div": N * K * 2
        },
        "early_exageration": {
            "mult": 2 * N * K
        },
        "compute_low_dimensional_affinities": {
            "compute_squared_euclidean_distance": {
                "add": T * d * N * (N - 1) / 2 * 2,
                "mult": T * d * N * (N - 1) / 2
            },
            "compute": {
                "add": T * N * (N - 1) * 2,
                "div": T * N * (N - 1)
            }
        },
        "create_tree": {
            "add": np.zeros(1)
        },
        "gradient_computation": {
            "add":
            T * N * (d + 1) + it_ins * d + it_sub * d + T *
            (N * K * 3 * d + N) + T * N * d,
            "mult":
            it_ins * 2 * d + it_sub * 2 * d + T * (N * K * 2 * d),
            "div":
            T * d * (1 + 3) + T * N * K + T * N * d
        },
        "gradient_update": {
            "add": T * (N * d + N * d * 2),
            "mult": T * N * d * 3
        },
        "normalize_2": {
            "add": T * 2 * N * d,
            "div": T * (d + N * d)
        }
    }

    flops_by_function_measure = dict(flops)
    flops_by_function_measure["compute_low_dimensional_affinities"] = \
        sum_operations_by_measure(flops["compute_low_dimensional_affinities"])
    flops_by_function = sum_operations_by_function(
        flops_by_function_measure,
        size=N.shape[0],
        function_list=FUNCTION_NAMES)

    total_flops = sum_operations_by_measure(flops_by_function_measure)
    total_flops = sum_operations(total_flops)

    return N, total_flops, flops_by_function, cycles
