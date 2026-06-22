"""CLI entry point for the VectorCore benchmark harness.

Examples
--------
    # Quick synthetic benchmark (no download needed):
    python -m benchmark.run --metric l2 --n 50000 --dim 128 --queries 1000 --k 10

    # Real SIFT1M (expects sift_base.fvecs / sift_query.fvecs / sift_groundtruth.ivecs):
    python -m benchmark.run --sift path/to/sift --k 10
"""

from __future__ import annotations

import argparse
import time

import numpy as np

import vectorcore

from .harness import (
    BenchmarkResult,
    Dataset,
    evaluate_index,
    load_sift,
    make_synthetic,
    print_table,
)


def build_bruteforce(dataset: Dataset) -> tuple[vectorcore.BruteForceIndex, float]:
    index = vectorcore.BruteForceIndex(dim=dataset.dim, metric=dataset.metric)
    ids = np.arange(dataset.n, dtype=np.uint64)
    t0 = time.perf_counter()
    index.add(dataset.train, ids)
    return index, time.perf_counter() - t0


def build_hnsw(dataset: Dataset, M: int) -> tuple[vectorcore.HnswIndex, float]:
    index = vectorcore.HnswIndex(dim=dataset.dim, M=M, metric=dataset.metric)
    ids = np.arange(dataset.n, dtype=np.uint64)
    t0 = time.perf_counter()
    index.add(dataset.train, ids)
    return index, time.perf_counter() - t0


def main() -> None:
    p = argparse.ArgumentParser(description="VectorCore benchmark harness")
    p.add_argument("--sift", type=str, default=None, help="path to a SIFT1M directory")
    p.add_argument("--metric", choices=("l2", "ip", "cosine"), default="l2")
    p.add_argument("--n", type=int, default=50_000, help="synthetic dataset size")
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--queries", type=int, default=1000)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hnsw", action="store_true", help="also benchmark HnswIndex (prototype)")
    p.add_argument("--hnsw-M", type=int, default=16)
    args = p.parse_args()

    print(f"vectorcore {vectorcore.__version__}")

    if args.sift:
        dataset = load_sift(args.sift, k=args.k)
    else:
        dataset = make_synthetic(args.n, args.dim, args.queries, args.metric,
                                 k=args.k, seed=args.seed)
    print(f"dataset: {dataset.name}  metric={dataset.metric}  "
          f"queries={dataset.n_queries}  k={args.k}\n")

    results: list[BenchmarkResult] = []

    bf, bf_build = build_bruteforce(dataset)
    results.append(evaluate_index(bf, dataset, args.k, index_name="BruteForce",
                                  build_seconds=bf_build, repeats=args.repeats))

    if args.hnsw:
        hn, hn_build = build_hnsw(dataset, args.hnsw_M)
        results.append(evaluate_index(hn, dataset, args.k, index_name="HNSW(proto)",
                                      build_seconds=hn_build, repeats=args.repeats))

    print()
    print_table(results)

    # Stage 1 gate: brute force is exact, so recall@k must be ~1.0.
    bf_recall = results[0].recall
    print()
    if bf_recall >= 0.999:
        print(f"GATE PASS: BruteForce recall@{args.k} = {bf_recall:.4f} (>= 0.999)")
    else:
        print(f"GATE FAIL: BruteForce recall@{args.k} = {bf_recall:.4f} (< 0.999)")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
