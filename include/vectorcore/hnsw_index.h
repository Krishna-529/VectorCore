#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "vectorcore/aligned_allocator.h"
#include "vectorcore/distance.h"

namespace vectorcore {

// HnswIndex (partial)
// -------------------
// This is intentionally a *partial* implementation:
// - Single-layer graph (level 0 only)
// - Simple neighbor selection (connect-to-nearest so far)
//
// Why it still matters for interviews:
// - You can explain the data structures: flat embeddings + adjacency lists
// - You can discuss how real HNSW extends this with multi-level routing,
//   efConstruction/efSearch, heuristic neighbor pruning, and bidirectional edges.

class HnswIndex {
public:
  HnswIndex(std::size_t dim, std::size_t M = 16, Metric metric = Metric::L2_SQUARED);

  std::size_t dim() const noexcept { return dim_; }
  std::size_t size() const noexcept { return size_; }

  void add(const float* vectors, std::size_t n, const std::uint64_t* ids = nullptr);
  void search(const float* query, std::size_t k, std::uint64_t* out_ids, float* out_scores) const;

private:
  std::size_t dim_ = 0;
  std::size_t size_ = 0;
  std::size_t M_ = 16;
  Metric metric_ = Metric::L2_SQUARED;

  std::vector<float, AlignedAllocator<float, 32>> embeddings_;
  std::vector<std::uint64_t> ids_;

  // Graph adjacency for level 0.
  // This is not a vector-of-vectors for embeddings (which is forbidden);
  // adjacency is metadata, and keeping it separate is standard for HNSW.
  std::vector<std::vector<std::uint32_t>> neighbors_;

  float score(const float* a, const float* b) const noexcept;
};

} // namespace vectorcore
