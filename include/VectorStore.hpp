#pragma once

#include <cstddef>
#include <utility>
#include <stdexcept>
#include <vector>

namespace vectrax {

// VectorStore
// -----------
// High-performance storage for fixed-dimension vectors.
//
// **Cache Locality (educational note)**
// Modern CPUs are fast, but memory is slow. The CPU fetches memory in *cache
// lines* (commonly 64 bytes). If your data is stored contiguously, a single
// cache line fetch brings in multiple adjacent floats that you will use next.
//
// A flat layout like:
//   data = [v0_0, v0_1, ..., v0_(d-1), v1_0, v1_1, ..., vN_(d-1)]
// means:
// - Iteration over vectors touches sequential memory (great for HW prefetch).
// - SIMD kernels can load contiguous chunks efficiently.
// - Fewer pointer indirections and fewer cache misses.
//
// In contrast, std::vector<std::vector<float>>:
// - Allocates each inner vector separately (heap fragmentation).
// - Requires chasing pointers (more cache misses and branchy code paths).
// - Makes tight loops slower even if the total number of floats is the same.
//
// This class therefore stores *all* vectors in one std::vector<float>.

class VectorStore {
public:
  explicit VectorStore(std::size_t dim);

  std::size_t dim() const noexcept { return dim_; }
  std::size_t size() const noexcept { return ids_.size(); }

  // Adds a single vector.
  // - `id` is an external identifier (metadata).
  // - `vec_data` points to `dim` floats.
  // - `dim` must match the store's fixed dimension.
  void add_vector(int id, const float* vec_data, std::size_t dim);

  // Returns a raw pointer to the start of the vector at internal index.
  // The returned pointer is valid as long as `data_` isn't reallocated.
  // (i.e., until the next push/insert that grows the vector).
  const float* get_vector(std::size_t internal_idx) const;

  int get_id(std::size_t internal_idx) const;

  // Brute-force kNN search over all stored vectors.
  // Returns (distance, external_id) pairs.
  std::vector<std::pair<float, int>> search(const float* query, int k) const;

private:
  // Phase 3 requirement: a static L2 distance function with scalar + AVX2.
  //
  // **SIMD (educational note)**
  // SIMD = Single Instruction, Multiple Data. With AVX2, a single instruction
  // can operate on 8 float32 values at once (256-bit registers).
  // For L2 distance we repeatedly do:
  //   diff = a[i] - b[i]
  //   acc += diff * diff
  // Using AVX2 we load 8 floats, subtract, and fused-multiply-add into an
  // accumulator vector.
  //
  // **Loop unrolling (educational note)**
  // Loop unrolling reduces loop overhead (branch + index increment) and can
  // help the compiler schedule instructions better. In SIMD code, the loop is
  // effectively unrolled by processing 8 elements per iteration.
  static float calculate_l2_dist(const float* a, const float* b, std::size_t dim) noexcept;

  std::size_t dim_ = 0;

  // Non-negotiable: a single flat storage vector for *all* embeddings.
  std::vector<float> data_;

  // Maps internal index -> external ID.
  std::vector<int> ids_;
};

} // namespace vectrax
