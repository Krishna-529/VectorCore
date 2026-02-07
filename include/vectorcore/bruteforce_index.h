#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <queue>
#include <stdexcept>
#include <utility>
#include <vector>

#include "vectorcore/aligned_allocator.h"
#include "vectorcore/distance.h"

namespace vectorcore {

// BruteForceIndex
// --------------
// Baseline index with predictable behavior and excellent correctness.
// It is also a perfect performance baseline to compare HNSW against.
//
// Non-negotiable design choices:
// - Flat memory model: embeddings are stored as a single contiguous array.
//   This improves spatial locality, cache line utilization, and SIMD throughput.
// - Dimension is fixed per index to avoid per-vector metadata and branches.

class BruteForceIndex {
public:
  explicit BruteForceIndex(std::size_t dim, Metric metric = Metric::L2_SQUARED);

  std::size_t dim() const noexcept { return dim_; }
  std::size_t size() const noexcept { return size_; }
  Metric metric() const noexcept { return metric_; }

  // Adds n vectors from a row-major [n, dim] matrix.
  //
  // Note: we must persist the vectors in the index, so we copy into our flat
  // storage exactly once. The "zero-copy" constraint applies to the Python
  // bridge for reading NumPy arrays without intermediate memcpy.
  void add(const float* vectors, std::size_t n, const std::uint64_t* ids = nullptr);

  // kNN search for a single query vector.
  // Output arrays must have capacity >= k.
  void search(const float* query, std::size_t k, std::uint64_t* out_ids, float* out_scores) const;

private:
  std::size_t dim_ = 0;
  std::size_t size_ = 0;
  Metric metric_ = Metric::L2_SQUARED;

  // Flat contiguous memory: [size_ * dim_]
  std::vector<float, AlignedAllocator<float, 32>> embeddings_;
  std::vector<std::uint64_t> ids_;

  float score(const float* a, const float* b) const noexcept;
};

} // namespace vectorcore
