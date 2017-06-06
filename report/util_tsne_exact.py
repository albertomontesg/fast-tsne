import numpy as np

FUNCTION_NAMES = ("normalize", "compute_pairwise_affinity_perplexity",
                  "symmetrize_affinities", "early_exageration",
                  "compute_low_dimensional_affinities", "gradient_computation",
                  "gradient_update", "normalize_2")


def sum_operations_by_measure(flops):
    flops_count = dict()
    # for measure in count_measure:
    #    flops_count[measure] =

    for k, v in flops.items():
        for op, cycles in v.items():
            if op not in flops_count:
                flops_count[op] = cycles.copy()
            else:
                flops_count[op] += cycles.copy()

    return flops_count


def sum_operations(flops):
    flops_count = None
    for k, v in flops.items():
        if flops_count is None:
            flops_count = v.copy()
        else:
            flops_count += v.copy()
    return flops_count


def sum_operations_by_function(flops, size, function_list=FUNCTION_NAMES):
    flops_count = np.zeros((size, len(function_list)), dtype="float")

    for i, func in enumerate(function_list):
        flops_count[:, i] += sum_operations(flops[func])
    return flops_count


def read_benchmark_exact(file_prefix, start=200, stop=4000, interval=200):
    input_size = np.arange(start, stop + 1, interval)
    iters = np.loadtxt(file_prefix + "iters.txt")
    cycles = np.loadtxt(file_prefix + "cycles.txt")
    assert np.all(cycles[:, :-1].sum(axis=1) == cycles[:, -1])

    N = input_size.astype("float")
    it = iters.astype("float")
    D = 28 * 28
    d = 2
    T = 1000  # Number of iterations

    count_measure = ("add", "mult", "div", "exp", "log")

    flops = {
        "normalize": {
            "add": 2 * N * D,
            "div": D + N * D
        },
        "compute_pairwise_affinity_perplexity": {
            "compute_squared_euclidean_distance": {
                "add": D * N * (N - 1) / 2 * 2,
                "mult": D * N * (N - 1) / 2
            },
            "binary_search": {
                "add": it * (N + N + 1 + 1),
                "mult": it * (N + 2 * N),
                "div": it * (1 + 1) + N * N,
                "exp": it * N,
                "log": it * (1 + 1)
            }
        },
        "symmetrize_affinities": {
            "add": N * (N - 1) / 2 + N * N,
            "div": N * N
        },
        "early_exageration": {
            "mult": 2 * N * N
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
        "gradient_computation": {
            "add": T * N * (N - 1) * (1 + 2 * d),
            "mult": T * N * (N - 1) * (1 + d),
            "div": T * N * (N - 1)
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
    flops_by_function_measure["compute_pairwise_affinity_perplexity"] = \
        sum_operations_by_measure(flops["compute_pairwise_affinity_perplexity"])
    flops_by_function_measure["compute_low_dimensional_affinities"] = \
        sum_operations_by_measure(flops["compute_low_dimensional_affinities"])
    flops_by_function = sum_operations_by_function(
        flops_by_function_measure, size=N.shape[0])

    total_flops = sum_operations_by_measure(flops_by_function_measure)
    total_flops = sum_operations(total_flops)

    return N, total_flops, flops_by_function, cycles
