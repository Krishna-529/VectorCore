#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "vectorcore/aligned_allocator.h"
#include "vectorcore/distance.h"

namespace vectorcore {

// PQIndex — Product Quantization
// ------------------------------
// Compresses high-dimensional float vectors into compact codes and searches
// them with Asymmetric Distance Computation (ADC).
//
// Idea (Jegou et al., 2011):
// - Split each D-dim vector into `m` subvectors of dim D/m.
// - Per subspace, run K-Means to learn `ksub = 2^nbits` centroids (a codebook).
// - Encode each subvector as the index of its nearest centroid: one byte per
//   subspace (for nbits <= 8). A D-float vector (4*D bytes) becomes `m` bytes.
//   e.g. D=128, m=16  ->  512 bytes -> 16 bytes (32x smaller).
//
// Search (ADC): the query stays full precision. For each subspace we precompute
// a lookup table of query-subvector-to-centroid scores, then each database
// vector's score is just `m` table lookups summed — no decompression.
//
// Only the codes + codebooks are stored; the original float vectors are not
// kept, which is where the memory saving comes from.
class PQIndex {
public:
  PQIndex(std::size_t dim, std::size_t m, Metric metric = Metric::L2_SQUARED,
          std::size_t nbits = 8, std::uint64_t seed = 100);

  std::size_t dim() const noexcept { return dim_; }
  std::size_t size() const noexcept { return size_; }
  std::size_t m() const noexcept { return m_; }
  std::size_t ksub() const noexcept { return ksub_; }
  std::size_t code_size() const noexcept { return m_; }  // bytes per vector
  bool is_trained() const noexcept { return trained_; }

  // Learn the per-subspace codebooks from a training set (n x dim, row-major).
  void train(const float* vectors, std::size_t n, std::size_t kmeans_iters = 25);

  // Encode and store n vectors. Requires the index to be trained first.
  void add(const float* vectors, std::size_t n, const std::uint64_t* ids = nullptr);

  // Approximate kNN via ADC. Output arrays must hold >= k entries.
  void search(const float* query, std::size_t k, std::uint64_t* out_ids,
              float* out_scores) const;

  // Binary persistence (same-architecture portable). Stores codebooks + codes.
  void save(const std::string& path) const;
  static PQIndex load(const std::string& path);

private:
  // Encode a single (already metric-prepared) vector into `m_` codes.
  void encode_one(const float* vec, std::uint8_t* out_codes) const;

  // Centroid c of subspace s: codebooks_ is [m_][ksub_][dsub_], row-major.
  const float* centroid(std::size_t s, std::size_t c) const noexcept {
    return codebooks_.data() + (((s * ksub_) + c) * dsub_);
  }

  std::size_t dim_ = 0;
  std::size_t m_ = 0;       // number of subspaces
  std::size_t dsub_ = 0;    // sub-vector dimension = dim_ / m_
  std::size_t ksub_ = 256;  // centroids per subspace = 2^nbits
  std::size_t size_ = 0;
  Metric metric_ = Metric::L2_SQUARED;
  bool trained_ = false;
  std::uint64_t seed_ = 100;

  std::vector<float, AlignedAllocator<float, 32>> codebooks_;  // [m_*ksub_*dsub_]
  std::vector<std::uint8_t> codes_;                            // [size_*m_]
  std::vector<std::uint64_t> ids_;
};

}  // namespace vectorcore
