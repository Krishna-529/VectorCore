#pragma once

#include <cstddef>
#include <cstdint>
#include <random>
#include <utility>
#include <vector>

#include "vectorcore/aligned_allocator.h"
#include "vectorcore/distance.h"

namespace vectorcore {

// HnswIndex
// ---------
// Hierarchical Navigable Small World graph for approximate nearest-neighbor
// search (Malkov & Yashunin, 2016).
//
// Structure:
// - A multi-layer proximity graph. Layer 0 holds every node; each higher layer
//   is exponentially sparser and acts as an "express lane" (skip-list analogy).
// - A node's maximum layer is drawn from a geometric distribution.
// - Embeddings live in one flat, cache-friendly buffer (the flat-memory
//   constraint applies to vectors; adjacency is graph metadata).
//
// Search: greedy descent through the sparse upper layers to get close fast,
// then an efSearch-width beam search at layer 0 for the final top-k.
//
// Insertion uses the RNG (relative neighborhood graph) heuristic to choose
// diverse neighbors, which keeps the graph navigable instead of clustered.
class HnswIndex {
public:
  HnswIndex(std::size_t dim, std::size_t M = 16, Metric metric = Metric::L2_SQUARED,
            std::size_t ef_construction = 200, std::uint64_t seed = 100);

  std::size_t dim() const noexcept { return dim_; }
  std::size_t size() const noexcept { return size_; }

  // Beam width used at layer 0 during search. Higher = better recall, slower.
  // Tunable per query without rebuilding the index.
  std::size_t ef_search() const noexcept { return ef_search_; }
  void set_ef_search(std::size_t ef) noexcept { ef_search_ = ef; }

  void add(const float* vectors, std::size_t n, const std::uint64_t* ids = nullptr);
  void search(const float* query, std::size_t k, std::uint64_t* out_ids, float* out_scores) const;

private:
  using Candidate = std::pair<float, std::uint32_t>;  // (badness, node index)

  // Metric-aware "badness": smaller == closer for every metric.
  //   L2:           squared distance
  //   IP / cosine:  negated similarity (vectors are pre-normalized for cosine)
  float badness(const float* a, const float* b) const noexcept;

  const float* vec(std::size_t idx) const noexcept {
    return embeddings_.data() + (idx * dim_);
  }

  // Beam search within a single layer. Returns up to `ef` closest nodes
  // (badness, node), unordered. `node_count` bounds the visited bitmap.
  std::vector<Candidate> search_layer(const float* q,
                                      const std::vector<std::uint32_t>& entry_points,
                                      std::size_t ef, int layer,
                                      std::size_t node_count) const;

  // RNG heuristic neighbor selection (HNSW Algorithm 4): pick up to M diverse
  // neighbors. `candidates` carry badness to the base element being connected.
  void select_neighbors_heuristic(std::vector<Candidate>& candidates, std::size_t M,
                                  std::vector<std::uint32_t>& out) const;

  // Insert the node at internal index `idx` (assumes nodes [0, idx) already in
  // the graph). Mutates the graph and possibly the entry point.
  void insert_one(std::uint32_t idx);

  int random_level();

  std::size_t dim_ = 0;
  std::size_t size_ = 0;
  std::size_t M_ = 16;            // target degree for layers > 0
  std::size_t M_max0_ = 32;       // max degree at layer 0 (2*M)
  std::size_t ef_construction_ = 200;
  std::size_t ef_search_ = 50;
  Metric metric_ = Metric::L2_SQUARED;
  double mL_ = 1.0;               // level-generation normalization factor

  std::mt19937_64 rng_;
  int max_level_ = -1;
  std::uint32_t entry_point_ = 0;

  std::vector<float, AlignedAllocator<float, 32>> embeddings_;
  std::vector<std::uint64_t> ids_;
  std::vector<int> node_level_;                              // per node: top layer
  std::vector<std::vector<std::vector<std::uint32_t>>> links_;  // [node][layer] -> neighbors

  // Version-stamped visited set for search_layer: O(1) reset per call instead
  // of zeroing an N-byte bitmap every time. Mutable so const search() can use
  // it (consequently search is not safe to call concurrently on one index).
  mutable std::vector<std::uint32_t> visited_;
  mutable std::uint32_t visit_stamp_ = 0;
};

}  // namespace vectorcore
