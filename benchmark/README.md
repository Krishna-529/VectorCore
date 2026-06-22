# VectorCore Benchmark Harness

Measures **recall@k** (vs. exact NumPy ground truth) and **QPS** for VectorCore
indexes. Reused across stages to gate HNSW and Product Quantization on quality.

## Requirements
- `vectorcore` installed (`pip install .` from the repo root)
- `numpy`

## Quick start (synthetic — no download)
```bash
# From the repo root:
python -m benchmark.run --metric l2 --n 50000 --dim 128 --queries 1000 --k 10
python -m benchmark.run --metric cosine --n 20000 --dim 128 --queries 500
python -m benchmark.run --metric l2 --n 4000 --dim 64 --hnsw   # compare prototype HNSW
```

Brute force is exact, so its recall@k must be ~1.0 — the harness asserts this as
a gate (`GATE PASS/FAIL`). The prototype `HnswIndex` currently scores very low
recall; raising it is the goal of Stage 2.

> Note: the current `HnswIndex` graph build is O(N²); keep `--n` small (≤ ~5000)
> when using `--hnsw` until the real multi-layer index lands.

## Real dataset: SIFT1M
Download the ANN_SIFT1M dataset (e.g. from the
[corpus-texmex](http://corpus-texmex.irisa.fr/) site) and extract it so the
directory contains:
```
sift_base.fvecs
sift_query.fvecs
sift_groundtruth.ivecs
```
Then:
```bash
python -m benchmark.run --sift path/to/sift --k 10
```
SIFT1M ships its own ground truth, so the harness loads it directly instead of
recomputing.

## Reusing the harness in code
```python
from benchmark.harness import make_synthetic, evaluate_index, print_table
import vectorcore, numpy as np, time

ds = make_synthetic(n=20000, dim=128, n_queries=500, metric="l2", k=10)
idx = vectorcore.BruteForceIndex(dim=ds.dim, metric=ds.metric)
t0 = time.perf_counter(); idx.add(ds.train, np.arange(ds.n, dtype=np.uint64))
res = evaluate_index(idx, ds, k=10, index_name="BruteForce",
                     build_seconds=time.perf_counter() - t0)
print_table([res])
```

`harness.py` exposes: `make_synthetic`, `load_sift`, `compute_ground_truth`,
`recall_at_k`, `measure_qps`, `evaluate_index`, `print_table`.
