"""Generate the HNSW search-path visualization data.

Builds a small HnswIndex, lays the vectors out in 2D (PCA), then traces a single
query through the real graph — greedy descent across the upper layers followed
by the efSearch beam search at layer 0 — recording the order nodes are visited.
The result is written as JSON for the React + D3 front-end.

Usage:
    python viz/export_graph.py            # writes viz/public/graph_data.json
"""

from __future__ import annotations

import json
import os
import time

import numpy as np

import vectorcore

OUT = os.path.join(os.path.dirname(__file__), "public", "graph_data.json")

N = 600          # nodes (kept small so the graph is legible)
DIM = 24
CLUSTERS = 6
M = 8
EF_SEARCH = 32
K = 10
SEED = 3


def make_clustered(n, dim, clusters, seed):
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-5, 5, size=(clusters, dim)).astype(np.float32)
    labels = rng.integers(0, clusters, size=n)
    pts = centers[labels] + rng.normal(0, 1.0, size=(n, dim)).astype(np.float32)
    return np.ascontiguousarray(pts, dtype=np.float32), labels


def pca_2d(x):
    """Project to 2D via PCA (NumPy SVD — no extra dependency)."""
    mu = x.mean(axis=0)
    xc = x - mu
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    coords = xc @ vt[:2].T
    return coords, mu, vt[:2]


def badness(metric, q, v):
    if metric == "l2":
        d = q - v
        return float(d @ d)
    return -float(q @ v)


def search_layer_trace(index, vectors, q, entry_points, ef, layer, metric,
                       visited_order, visited_edges):
    """Faithful replica of the C++ search_layer, recording visit order and the
    parent->child edges discovered (the actual traversal tree)."""
    import heapq
    visited = set()
    cand_heap = []          # min-heap by badness
    result_heap = []        # max-heap by badness (store negatives)
    for ep in entry_points:
        if ep in visited:
            continue
        visited.add(ep)
        b = badness(metric, q, vectors[ep])
        heapq.heappush(cand_heap, (b, ep))
        heapq.heappush(result_heap, (-b, ep))
        visited_order.append(int(ep))
    while len(result_heap) > ef:
        heapq.heappop(result_heap)

    while cand_heap:
        cb, c = heapq.heappop(cand_heap)
        worst = -result_heap[0][0] if result_heap else float("inf")
        if cb > worst and len(result_heap) >= ef:
            break
        for e in index.neighbors(int(c), layer):
            if e in visited:
                continue
            visited.add(e)
            b = badness(metric, q, vectors[e])
            worst = -result_heap[0][0] if result_heap else float("inf")
            if b < worst or len(result_heap) < ef:
                heapq.heappush(cand_heap, (b, e))
                heapq.heappush(result_heap, (-b, e))
                visited_order.append(int(e))
                visited_edges.append([int(c), int(e)])
                if len(result_heap) > ef:
                    heapq.heappop(result_heap)

    return [(-nb, n) for nb, n in sorted(result_heap)]


def main():
    data, labels = make_clustered(N, DIM, CLUSTERS, SEED)
    ids = np.arange(N, dtype=np.uint64)
    metric = "l2"

    index = vectorcore.HnswIndex(dim=DIM, M=M, metric=metric, ef_construction=100, seed=42)
    index.ef_search = EF_SEARCH
    index.add(data, ids)

    # 2D layout for all nodes (and project the query into the same space).
    coords, mu, comps = pca_2d(data)

    # Pick a query near a cluster but not exactly a point, for a nice path.
    rng = np.random.default_rng(SEED + 1)
    qi = int(rng.integers(0, N))
    query = (data[qi] + rng.normal(0, 0.5, size=DIM)).astype(np.float32)
    q2d = (query - mu) @ comps.T

    # --- Trace the search the way the C++ engine does it ---
    descent = []  # list of {layer, node}
    visited_order = []
    visited_edges = []
    ep = int(index.entry_point)
    descent.append({"layer": int(index.max_level), "node": ep})
    for lc in range(int(index.max_level), 0, -1):
        w = search_layer_trace(index, data, query, [ep], 1, lc, metric, [], [])
        if w:
            ep = int(min(w, key=lambda t: t[0])[1])
        descent.append({"layer": lc - 1, "node": ep})

    layer0 = search_layer_trace(index, data, query, [ep], max(EF_SEARCH, K), 0, metric,
                                visited_order, visited_edges)
    results = [int(n) for _, n in sorted(layer0)[:K]]

    # Real engine timing + recall vs brute force.
    t0 = time.perf_counter()
    for _ in range(50):
        r_ids, _ = index.search(query, K)
    latency_ms = (time.perf_counter() - t0) / 50 * 1e3

    bf = vectorcore.BruteForceIndex(dim=DIM, metric=metric)
    bf.add(data, ids)
    true_ids, _ = bf.search(query, K)
    recall = len(set(int(x) for x in r_ids) & set(int(x) for x in true_ids)) / K

    # Layer-0 edges among nodes (deduped, undirected) — for the backdrop graph.
    edges = set()
    for n in range(N):
        for nb in index.neighbors(n, 0):
            a, b = (n, int(nb)) if n < nb else (int(nb), n)
            edges.add((a, b))

    nodes = [{
        "id": i,
        "x": float(coords[i, 0]),
        "y": float(coords[i, 1]),
        "level": int(index.node_level(i)),
        "cluster": int(labels[i]),
    } for i in range(N)]

    payload = {
        "meta": {
            "n": N, "dim": DIM, "M": M, "ef_search": EF_SEARCH, "k": K,
            "metric": metric, "max_level": int(index.max_level),
            "latency_ms": latency_ms, "recall": recall,
            "visited_count": len(set(visited_order)),
        },
        "nodes": nodes,
        "edges": [[a, b] for a, b in edges],
        "query": {"x": float(q2d[0]), "y": float(q2d[1])},
        "trace": {
            "entry_point": int(index.entry_point),
            "descent": descent,             # node chosen at each layer (zoom-in)
            "visited": visited_order,       # order of layer-0 expansion
            "visited_edges": visited_edges, # parent->child traversal tree at layer 0
            "results": results,             # final top-k
        },
    }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(payload, f)
    print(f"wrote {OUT}")
    print(f"  nodes={N} edges={len(edges)} max_level={index.max_level} "
          f"visited={len(set(visited_order))} latency={latency_ms:.3f}ms recall@{K}={recall:.2f}")


if __name__ == "__main__":
    main()
