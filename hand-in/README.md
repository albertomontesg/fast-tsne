This is just a one-time hand in of the code and presentation. We used GitHub for our development (Alen is invited) and we refer to that for the commit history, and more details.

`src/implementations/optimizations` - Folders in which the different parts where optimized.
`src/implementations/original` - Implementation by van der Maaten, as cited in the paper.
`src/implementations/tsne_exact` - Instrumented version of the exact algorithm from the original.
`src/implementations/tsne_nlogn` - Instrumented version of the `n log n` algorithm from the original.
`src/implementations/tsne_exact_optimizations` - The instrumented version of the exact algorithm split up a bit more and with slight optimization.
`src/implementations/tsne_exact_final` - The algorithm with the baseline, the best optimizations and benchmarking scripts.
`src/implementations/utils` - Utility scripts we wrote.
`src/implementations/validation` - Verification scripts we wrote.
data/ - Contains a script to generate data and the MNIST dataset.
