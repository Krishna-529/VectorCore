#include "VectorStore.hpp"

#include <algorithm>
#include <immintrin.h>
#include <limits>
#include <queue>
#include <utility>

namespace vectorcore {

float VectorStore::calculate_l2_dist(const float* a, const float* b, std::size_t dim) noexcept {
  // Scalar fallback (always correct).
  //
  // Note: This function is hot in brute-force search. Keeping it `noexcept`
  // allows the compiler to optimize more aggressively.

#if defined(__AVX2__)
  // AVX2 path: process 8 floats per iteration.
  // We use unaligned loads (_mm256_loadu_ps) because callers only guarantee
  // pointer validity, not 32-byte alignment.
  __m256 acc = _mm256_setzero_ps();
  std::size_t i = 0;

  for (; i + 8 <= dim; i += 8) {
    const __m256 va = _mm256_loadu_ps(a + i);
    const __m256 vb = _mm256_loadu_ps(b + i);
    const __m256 diff = _mm256_sub_ps(va, vb);

    // FMA: acc += diff * diff
    // On CPUs with FMA, this reduces instruction count and improves throughput.
    acc = _mm256_fmadd_ps(diff, diff, acc);
  }

  alignas(32) float tmp[8];
  _mm256_store_ps(tmp, acc);

  float sum = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];

  // Tail loop for remaining elements.
  for (; i < dim; ++i) {
    const float d = a[i] - b[i];
    sum += d * d;
  }

  return sum;
#else
  // Scalar loop.
  float sum = 0.0f;

  // Mild unrolling helps reduce loop overhead and may improve instruction
  // scheduling even without SIMD.
  std::size_t i = 0;
  for (; i + 4 <= dim; i += 4) {
    const float d0 = a[i + 0] - b[i + 0];
    const float d1 = a[i + 1] - b[i + 1];
    const float d2 = a[i + 2] - b[i + 2];
    const float d3 = a[i + 3] - b[i + 3];
    sum += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
  }
  for (; i < dim; ++i) {
    const float d = a[i] - b[i];
    sum += d * d;
  }
  return sum;
#endif
}

VectorStore::VectorStore(std::size_t dim) : dim_(dim) {
  if (dim_ == 0) {
    throw std::invalid_argument("VectorStore dim must be > 0");
  }
}

void VectorStore::add_vector(int id, const float* vec_data, std::size_t dim) {
  if (!vec_data) {
    throw std::invalid_argument("vec_data pointer is null");
  }
  if (dim != dim_) {
    throw std::invalid_argument("dim mismatch: vector dim must match store dim");
  }

  // Reserve to avoid repeated reallocations.
  // Each reallocation copies the entire backing array, which is expensive.
  const std::size_t next_size = size() + 1;
  data_.reserve(next_size * dim_);
  ids_.reserve(next_size);

  ids_.push_back(id);
  data_.insert(data_.end(), vec_data, vec_data + dim_);
}

const float* VectorStore::get_vector(std::size_t internal_idx) const {
  if (internal_idx >= size()) {
    throw std::out_of_range("internal_idx out of range");
  }

  // Pointer arithmetic is safe here because:
  // - data_ is contiguous
  // - we store exactly dim_ floats per vector
  // - internal_idx is bounds-checked above
  return data_.data() + (internal_idx * dim_);
}

int VectorStore::get_id(std::size_t internal_idx) const {
  if (internal_idx >= size()) {
    throw std::out_of_range("internal_idx out of range");
  }
  return ids_[internal_idx];
}

std::vector<std::pair<float, int>> VectorStore::search(const float* query, int k) const {
  if (!query) {
    throw std::invalid_argument("query pointer is null");
  }
  if (k <= 0) {
    return {};
  }

  const std::size_t n = size();
  if (n == 0) {
    return {};
  }

  const std::size_t kk = std::min<std::size_t>(static_cast<std::size_t>(k), n);

  // Requirement: use a std::priority_queue (min-heap) to keep top-k.
  //
  // A min-heap naturally gives you the smallest element at the top.
  // For top-k *nearest* neighbors (smallest distances) we'd like to quickly
  // evict the *worst* (largest distance) among the kept items.
  //
  // Trick: store `key = -distance`. Then:
  // - Worst (largest distance) => most negative key => becomes the *smallest*
  //   key => rises to the top of a min-heap.
  // - When heap grows beyond k, popping removes the worst candidate.
  using Keyed = std::pair<float, int>; // (key=-dist, external_id)
  struct GreaterKey {
    bool operator()(const Keyed& a, const Keyed& b) const noexcept { return a.first > b.first; }
  };

  std::priority_queue<Keyed, std::vector<Keyed>, GreaterKey> heap;

  for (std::size_t i = 0; i < n; ++i) {
    const float* v = data_.data() + (i * dim_);
    const float dist = calculate_l2_dist(query, v, dim_);
    heap.emplace(-dist, ids_[i]);

    if (heap.size() > kk) {
      heap.pop();
    }
  }

  std::vector<std::pair<float, int>> result;
  result.reserve(kk);
  while (!heap.empty()) {
    const auto [key, id] = heap.top();
    heap.pop();
    result.emplace_back(-key, id); // convert back to positive distance
  }

  std::sort(result.begin(), result.end(), [](const auto& a, const auto& b) {
    return a.first < b.first; // smaller distance first
  });

  return result;
}

} // namespace vectorcore
