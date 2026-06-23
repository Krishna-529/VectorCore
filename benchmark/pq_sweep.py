"""Product Quantization memory/recall tradeoff sweep.

Sweeps the number of subspaces `m` and reports, for each, the PQ code size,
the memory-reduction factor vs. raw float32, recall@k (ADC vs. exact ground
truth), and query latency. This is the PQ "tradeoff curve".

Examples
--------
    # Synthetic (no download):
    python -m benchmark.pq_sweep --n 50000 --dim 128 --queries 1000 --k 10

    # SIFT1M (held-out queries + official ground truth):
    python -m benchmark.pq_sweep --sift benchmark/data/sift --k 10 --max-queries 1000 \
        --train 100000
"""

from __future__ import annotations

import argparse
import time

import numpy as np

import vectorcore

from .harness import Dataset, load_sift, make_synthetic, recall_at_k


def divisors_of(dim: int, candidates=(8, 16, 32, 64)) -> list[int]:
    return [m for m in candidates if dim % m == 0]


def main() -> None:
    p = argparse.ArgumentParser(description="PQ memory/recall tradeoff sweep")
    p.add_argument("--sift", type=str, default=None)
    p.add_argument("--metric", choices=("l2", "ip", "cosine"), default="l2")
    p.add_argument("--n", type=int, default=50_000)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--queries", type=int, default=1000)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--max-queries", type=int, default=None)
    p.add_argument("--train", type=int, default=100_000,
                   help="number of vectors to train codebooks on (sampled)")
    p.add_argument("--m-list", type=int, nargs="*", default=None,
                   help="explicit list of m values (defaults to divisors of dim)")
    p.add_argument("--kmeans-iters", type=int, default=25)
    args = p.parse_args()

    print(f"vectorcore {vectorcore.__version__}")

    if args.sift:
        ds: Dataset = load_sift(args.sift, k=args.k)
    else:
        ds = make_synthetic(args.n, args.dim, args.queries, args.metric,
                            k=args.k, seed=0)

    queries = ds.queries
    gt = ds.ground_truth
    if args.max_queries is not None and args.max_queries < queries.shape[0]:
        queries = queries[:args.max_queries]
        gt = gt[:args.max_queries]

    n, dim = ds.n, ds.dim
    orig_bytes = n * dim * 4
    ids = np.arange(n, dtype=np.uint64)
    train_n = min(args.train, n)
    train_x = np.ascontiguousarray(ds.train[:train_n])

    m_list = args.m_list or divisors_of(dim)
    print(f"dataset: {ds.name}  metric={ds.metric}  queries={queries.shape[0]}  "
          f"k={args.k}  train_on={train_n}\n")

    header = (f"{'m':>4}{'code':>8}{'reduction':>12}{'recall@k':>10}"
              f"{'lat(ms)':>10}{'train(s)':>10}{'build(s)':>10}")
    print(header)
    print("-" * len(header))

    for m in m_list:
        pq = vectorcore.PQIndex(dim=dim, m=m, metric=ds.metric, nbits=8)
        t0 = time.perf_counter()
        pq.train(train_x, args.kmeans_iters)
        train_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        pq.add(ds.train, ids)
        build_s = time.perf_counter() - t0

        found = np.empty((queries.shape[0], args.k), dtype=np.uint64)
        t0 = time.perf_counter()
        for i in range(queries.shape[0]):
            found[i], _ = pq.search(queries[i], args.k)
        lat_ms = (time.perf_counter() - t0) / queries.shape[0] * 1e3

        recall = recall_at_k(found, gt, args.k)
        code_bytes = n * m
        reduction = orig_bytes / code_bytes
        print(f"{m:>4}{str(m) + 'B':>8}{f'{reduction:.0f}x':>12}{recall:>10.4f}"
              f"{lat_ms:>10.3f}{train_s:>10.1f}{build_s:>10.1f}")

    print(f"\n(raw float32 size = {orig_bytes / 1e6:.0f} MB; reduction factor = "
          f"32*dim / (8*m) per vector)")


if __name__ == "__main__":
    main()
