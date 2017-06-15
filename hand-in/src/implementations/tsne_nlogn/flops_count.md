# Flops Count

Lets use first the following count measure:
`Cost = {add, mult, div, exp, log}`

Lets define the variables:
N: Number of input vectors
K: Number of considered neighbors
D: Input dimension
d: Output dimension
T: Number of iterations


Now lets check function by function at each step:

* `normalize`:
`add = 2 * N * D`;
`div = D + N * D`; same as n^2 case

* `compute_pairwise_affinity_perplexity`:
    * the function `eucliedian_distance` has `2*D adds`, `D mult`, `1 sqrt`
    * create vptree
	    * calls `buildFromPoints` `it_buildFromPoints` many times, each has `1 div` and `1 mul`
	    * calls `eucliedian_distance` `it_create_eucledian_distance` times
    * The binary search is the same as in the n^2 case, but we always only consider K neighbors, thus:
    `add = it * (K + K + 1 + 1)`;
    `mult = it * (K + 2*K)`;
    `div = it * (1 + 1) + N * K`;
    `exp = it * (K)`;
    `log = it * (1 + 1)`;
   * But this additionally calls also the function search on the tree. Search does some heap operations and calls saerch `it_search_eucledian_distance` many times.
   * If we call `it_ec = it_create_eucledian_distance + it_search_eucledian_distance ` we get the following costs for `compute_pairwise_affinity_perplexity`:
    `add = it * (2K + 2) + it_ec  * 2 * D`;
    `mult = it * (3K) + it_ec  * D + it_buildFromPoints `;
    `div = it * 2 + N * K + it_buildFromPoints `;
    `exp = it * K`;
    `log = it * 2`;
    `sqrt = it_ec`;


* `symmetrize_affinities_nlogn`:
`add = N*K*3`;
`div = N*K*2`;

* `early_exageration`:
`mult = N*K`;
`div = N*K`;
or `mult = 2 * N*K`;

From now on all the computations are done at every iteration so should be counted `T` times.

* `compute_low_dimensional_affinities`: same as n^2 algorithm
    * `compute_squared_euclidean_distance`:
    `add = d * N*(N-1)/2 * 2`;
    `mult = d * N*(N-1)/2`;
    * `add = N * (N-1) * 2`; `div = N * (N-1)`;

* `gradient_computation`:
	* create SPTree
		* `adds = T*N*(d + 1)`, `divs = T*d * (1 + 3)`
		* calls to insert `add = it_insert * d`; `mul = it_insert * 2 * d`
		* calls to subdivide `add = it_sub * d`; `mul = it_sub * 2 * d`
	* computeEdgeForces `add=T*(N*K*3*d+N)`;`mult=T*(N*K*2*d)`;`div=T*N*K`;
	* finalize gradients `add = T*N*d`; `div = T*N*d`

* `gradient_update`: same as n^2 case
`add = N * d * 2 + N*d`;
`mult = N * d * 3`;
From gains, all the operations are counted as multiplications.

* `normalize`: same as n^2 case
`add = 2 * N * d`;
`div = D + N * d`;
