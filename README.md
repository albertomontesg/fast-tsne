

# Fast 2d N^2 t-distributed Stochastic Neighbor Embedding

by *Marc Fischer*, *Alberto Montes*, *Marko Pichler Trauber*, *Andreas Blöchliger*

## Abstract

We present a faster implementation of the N^2 t-SNE embedding. Our version is restricted to producing 2d output vectors and developed for CPU architectures supporting AVX2 instructions. We analyze performance blockers and elevate them by increasing locality and hand-optimizing the code. This yields the fastest implementation known to the authors.

# Code

All the optimizations and running code, as well as the presentation and the report are placed inside the `hand-in` folder.
