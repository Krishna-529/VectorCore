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


def build_hnsw(dataset: Dataset, M: int, ef_construction: int,
               ef_search: int) -> tuple[vectorcore.HnswIndex, float]:
    index = vectorcore.HnswIndex(dim=dataset.dim, M=M, metric=dataset.metric,
                                 ef_construction=ef_construction)
    index.ef_search = ef_search
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
    p.add_argument("--max-queries", type=int, default=None,
                   help="limit number of queries (subset) for faster runs")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hnsw", action="store_true", help="also benchmark HnswIndex")
    p.add_argument("--hnsw-M", type=int, default=16)
    p.add_argument("--hnsw-ef-construction", type=int, default=200)
    p.add_argument("--hnsw-ef-search", type=int, default=64)
    args = p.parse_args()

    print(f"vectorcore {vectorcore.__version__}")

    if args.sift:
        dataset = load_sift(args.sift, k=args.k)
    else:
        dataset = make_synthetic(args.n, args.dim, args.queries, args.metric,
                                 k=args.k, seed=args.seed)

    if args.max_queries is not None and args.max_queries < dataset.n_queries:
        dataset.queries = dataset.queries[:args.max_queries]
        dataset.ground_truth = dataset.ground_truth[:args.max_queries]

    print(f"dataset: {dataset.name}  metric={dataset.metric}  "
          f"queries={dataset.n_queries}  k={args.k}\n")

    results: list[BenchmarkResult] = []

    bf, bf_build = build_bruteforce(dataset)
    results.append(evaluate_index(bf, dataset, args.k, index_name="BruteForce",
                                  build_seconds=bf_build, repeats=args.repeats))

    if args.hnsw:
        hn, hn_build = build_hnsw(dataset, args.hnsw_M, args.hnsw_ef_construction,
                                  args.hnsw_ef_search)
        results.append(evaluate_index(hn, dataset, args.k,
                                      index_name=f"HNSW(M{args.hnsw_M},ef{args.hnsw_ef_search})",
                                      build_seconds=hn_build, repeats=args.repeats))

    print()
    print_table(results)
    print()

    failures = []

    # Brute force is exact modulo float32 rounding on near-ties, so recall is
    # ~1.0 (0.99 tolerance covers boundary rank-swaps vs an exact-L2 ground truth).
    bf = results[0]
    if bf.recall >= 0.99:
        print(f"GATE PASS: BruteForce recall@{args.k} = {bf.recall:.4f} (>= 0.99)")
    else:
        failures.append(f"BruteForce recall@{args.k} = {bf.recall:.4f} (< 0.99)")

    # Stage 2 gate: HNSW recall@k >= 0.95 and QPS >= 10x brute force.
    if args.hnsw:
        hn = results[1]
        speedup = hn.qps / bf.qps if bf.qps else float("inf")
        recall_ok = hn.recall >= 0.95
        speed_ok = speedup >= 10.0
        status = "PASS" if (recall_ok and speed_ok) else "FAIL"
        print(f"GATE {status}: HNSW recall@{args.k} = {hn.recall:.4f} (>= 0.95), "
              f"speedup = {speedup:.1f}x (>= 10x)")
        if not recall_ok:
            failures.append(f"HNSW recall@{args.k} = {hn.recall:.4f} (< 0.95)")
        if not speed_ok:
            failures.append(f"HNSW speedup = {speedup:.1f}x (< 10x)")

    if failures:
        print("GATE FAIL: " + "; ".join(failures))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
