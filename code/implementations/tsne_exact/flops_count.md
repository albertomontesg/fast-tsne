# Flops Count

Lets use first the following count measure:
`Cost = {add, mult, div, exp, log}`

Lets define the variables:
N: Number of input vectors
D: Input dimension
d: Output dimension
T: Number of iterations


Now lets check function by function at each step:

* `normalize`:
`add = 2 * N * D`;
`div = D + N * D`;

* `compute_pairwise_affinity_perplexity`:
    * `compute_squared_euclidean_distance`:
    `add = D * N*(N-1)/2 * 2`;
    `mult = D * N*(N-1)/2`;
    * Binary search for perplexity (depends on number of iterations `it` required to achieve the binary search):
    `add = it * (N + N + 1 + 1)`;
    `mult = it * (N + 2*N)`;
    `div = it * (1 + 1) + N * N`;
    `exp = it * (N)`;
    `log = it * (1 + 1)`;
    All the operations regarding the update of the `beta` value at the binary search it has been considered to perform at the final ifs the sum and the division as observing the resulting betas, the values are between 0 and 1.

* `symmetrize_affinities`:
`add = N*(N-1) / 2 + N*N`;
`div = N*N`;

* `early_exageration`:
`mult = N*N`;
`div = N*N`;
or `mult = 2 * N*N`;

From now on all the computations are done at every iteration so should be counted `T` times.

* `compute_low_dimensional_affinities`:
    * `compute_squared_euclidean_distance`:
    `add = d * N*(N-1)/2 * 2`;
    `mult = d * N*(N-1)/2`;
    * `add = N * (N-1) * 2`; `div = N * (N-1)`;

* `gradient_computation`:
`add = N * (N-1) * (1 + 2*d)`;
`mult = N * (N-1) * (1 + d)`;
`div = N * (N-1)`;

* `gradient_update`:
`add = N * d * 2 + N*d`;
`mult = N * d * 3`;
From gains, all the operations are counted as multiplications.

* `normalize`:
`add = 2 * N * d`;
`div = D + N * d`;
