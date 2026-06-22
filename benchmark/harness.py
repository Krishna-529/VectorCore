"""Reusable benchmark harness for VectorCore indexes.

Provides dataset generation/loading, exact ground-truth computation, and
recall@k / QPS measurement. Designed to be reused as later stages add HNSW and
Product Quantization: any object exposing

    index.add(x: float32[n, dim], ids: uint64[n] | None)
    index.search(q: float32[dim], k) -> (ids: uint64[k], scores: float32[k])

can be evaluated with :func:`evaluate_index`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

# Metrics understood by both this harness and the C++ engine.
METRICS = ("l2", "ip", "cosine")


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #
@dataclass
class Dataset:
    train: np.ndarray  # (n, dim) float32 — vectors to index
    queries: np.ndarray  # (q, dim) float32
    ground_truth: np.ndarray  # (q, k) int64 — exact nearest-neighbor row indices
    metric: str
    name: str

    @property
    def n(self) -> int:
        return self.train.shape[0]

    @property
    def dim(self) -> int:
        return self.train.shape[1]

    @property
    def n_queries(self) -> int:
        return self.queries.shape[0]


def make_synthetic(n: int, dim: int, n_queries: int, metric: str, k: int = 10,
                   seed: int = 0) -> Dataset:
    """Random Gaussian dataset with exact ground truth computed via NumPy."""
    rng = np.random.default_rng(seed)
    train = rng.standard_normal((n, dim), dtype=np.float32)
    queries = rng.standard_normal((n_queries, dim), dtype=np.float32)
    gt = compute_ground_truth(train, queries, k, metric)
    return Dataset(train, queries, gt, metric, name=f"synthetic[{n}x{dim}]")


def _read_fvecs(path: str) -> np.ndarray:
    """Read a .fvecs file (int32 dim header per row, then `dim` float32s)."""
    raw = np.fromfile(path, dtype=np.int32)
    if raw.size == 0:
        return np.zeros((0, 0), dtype=np.float32)
    dim = int(raw[0])
    rows = raw.reshape(-1, dim + 1)
    return np.ascontiguousarray(rows[:, 1:].view(np.float32))


def _read_ivecs(path: str) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.int32)
    dim = int(raw[0])
    return np.ascontiguousarray(raw.reshape(-1, dim + 1)[:, 1:])


def load_sift(base_dir: str, k: int = 10) -> Dataset:
    """Load the SIFT1M ANN benchmark (expects the standard `sift/` layout:
    sift_base.fvecs, sift_query.fvecs, sift_groundtruth.ivecs)."""
    import os

    base = _read_fvecs(os.path.join(base_dir, "sift_base.fvecs")).astype(np.float32)
    query = _read_fvecs(os.path.join(base_dir, "sift_query.fvecs")).astype(np.float32)
    gt = _read_ivecs(os.path.join(base_dir, "sift_groundtruth.ivecs"))[:, :k].astype(np.int64)
    return Dataset(base, query, gt, metric="l2", name=f"sift1m[{base.shape[0]}x{base.shape[1]}]")


# --------------------------------------------------------------------------- #
# Ground truth & metrics
# --------------------------------------------------------------------------- #
def compute_ground_truth(train: np.ndarray, queries: np.ndarray, k: int,
                         metric: str, batch: int = 256) -> np.ndarray:
    """Exact top-k nearest-neighbor row indices for each query (NumPy reference).

    Returns int64 array of shape (n_queries, k). Batched over queries to bound
    peak memory on large datasets.
    """
    train = np.ascontiguousarray(train, dtype=np.float32)
    queries = np.ascontiguousarray(queries, dtype=np.float32)
    n_q = queries.shape[0]
    out = np.empty((n_q, k), dtype=np.int64)

    if metric == "cosine":
        train = _l2_normalize(train)
        queries = _l2_normalize(queries)

    for start in range(0, n_q, batch):
        qb = queries[start:start + batch]
        if metric == "l2":
            # squared L2 via (a-b)^2 = |a|^2 - 2ab + |b|^2; smaller = closer
            d = (np.sum(qb * qb, axis=1, keepdims=True)
                 - 2.0 * qb @ train.T
                 + np.sum(train * train, axis=1)[None, :])
            idx = np.argpartition(d, kth=k - 1, axis=1)[:, :k]
            order = np.argsort(np.take_along_axis(d, idx, axis=1), axis=1)
        else:  # ip or cosine: larger score = closer
            s = qb @ train.T
            idx = np.argpartition(-s, kth=k - 1, axis=1)[:, :k]
            order = np.argsort(-np.take_along_axis(s, idx, axis=1), axis=1)
        out[start:start + qb.shape[0]] = np.take_along_axis(idx, order, axis=1)

    return out


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (x / norms).astype(np.float32)


def recall_at_k(found_ids: np.ndarray, gt_ids: np.ndarray, k: int) -> float:
    """Mean recall@k: average fraction of true top-k found, over all queries."""
    total = 0.0
    for found_row, gt_row in zip(found_ids, gt_ids):
        truth = set(int(x) for x in gt_row[:k])
        hits = sum(1 for x in found_row[:k] if int(x) in truth)
        total += hits / k
    return total / len(gt_ids)


# --------------------------------------------------------------------------- #
# Measurement
# --------------------------------------------------------------------------- #
@dataclass
class BenchmarkResult:
    index_name: str
    dataset: str
    metric: str
    k: int
    recall: float
    qps: float
    mean_latency_ms: float
    build_seconds: float


def measure_qps(search_fn, queries: np.ndarray, k: int,
                repeats: int = 3) -> tuple[np.ndarray, float, float]:
    """Run single-query searches; return (found_ids, qps, mean_latency_ms).

    `search_fn(q, k) -> (ids, scores)`. Reports the best (lowest) wall-time
    across `repeats` to reduce noise. Found ids are from the first pass.
    """
    n_q = queries.shape[0]
    found = np.empty((n_q, k), dtype=np.uint64)

    best = float("inf")
    for r in range(repeats):
        t0 = time.perf_counter()
        for i in range(n_q):
            ids, _ = search_fn(queries[i], k)
            if r == 0:
                found[i] = ids[:k]
        best = min(best, time.perf_counter() - t0)

    qps = n_q / best if best > 0 else float("inf")
    mean_latency_ms = (best / n_q) * 1e3 if n_q else 0.0
    return found, qps, mean_latency_ms


def evaluate_index(index, dataset: Dataset, k: int, *, index_name: str,
                   build_seconds: float, repeats: int = 3) -> BenchmarkResult:
    """Measure recall@k and QPS of an already-built index against a dataset."""
    found, qps, lat = measure_qps(lambda q, kk: index.search(q, kk),
                                  dataset.queries, k, repeats=repeats)
    # found ids are row indices into the train set (default ids 0..n-1).
    recall = recall_at_k(found, dataset.ground_truth, k)
    return BenchmarkResult(index_name, dataset.name, dataset.metric, k,
                           recall, qps, lat, build_seconds)


def print_table(results: list[BenchmarkResult]) -> None:
    header = f"{'index':<16}{'dataset':<22}{'metric':<8}{'k':>4}{'recall@k':>10}{'QPS':>12}{'lat(ms)':>10}{'build(s)':>10}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r.index_name:<16}{r.dataset:<22}{r.metric:<8}{r.k:>4}"
              f"{r.recall:>10.4f}{r.qps:>12.1f}{r.mean_latency_ms:>10.3f}{r.build_seconds:>10.3f}")
