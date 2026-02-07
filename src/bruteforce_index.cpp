#include "vectrax/bruteforce_index.h"

#include <algorithm>
#include <cstring>

namespace vectrax {

namespace {

// For L2 we want *smaller* scores; for inner product we want *larger* scores.
// To reuse a single heap structure we store a "badness" value:
// - L2: badness = distance (larger is worse)
// - IP: badness = -similarity (larger is worse)
inline float badness_from_score(Metric metric, float score) noexcept {
  return (metric == Metric::L2_SQUARED) ? score : -score;
}

} // namespace

BruteForceIndex::BruteForceIndex(std::size_t dim, Metric metric) : dim_(dim), metric_(metric) {
  if (dim_ == 0) {
    throw std::invalid_argument("dim must be > 0");
  }
}

void BruteForceIndex::add(const float* vectors, std::size_t n, const std::uint64_t* ids) {
  if (!vectors) {
    throw std::invalid_argument("vectors pointer is null");
  }
  if (n == 0) {
    return;
  }

  // Reserve once to avoid repeated reallocations (each reallocation is a full memcpy).
  const std::size_t old_size = size_;
  const std::size_t new_size = size_ + n;

  embeddings_.reserve(new_size * dim_);
  ids_.reserve(new_size);

  // Append the new vectors in a single flat block.
  embeddings_.insert(embeddings_.end(), vectors, vectors + (n * dim_));

  if (ids) {
    ids_.insert(ids_.end(), ids, ids + n);
  } else {
    // Deterministic IDs (0..N-1) are interview-friendly.
    // In production you might accept external IDs (uint64) from the caller.
    for (std::size_t i = 0; i < n; ++i) {
      ids_.push_back(static_cast<std::uint64_t>(old_size + i));
    }
  }

  size_ = new_size;
}

float BruteForceIndex::score(const float* a, const float* b) const noexcept {
  // Small but important C++ detail:
  // - We keep metric_ as an enum class (scoped enum) for type safety.
  // - We switch to avoid branches inside the inner loop; each metric has its own kernel.
  switch (metric_) {
    case Metric::L2_SQUARED:
      return l2_squared(a, b, dim_);
    case Metric::INNER_PRODUCT:
      return inner_product(a, b, dim_);
    default:
      return l2_squared(a, b, dim_);
  }
}

void BruteForceIndex::search(const float* query, std::size_t k, std::uint64_t* out_ids, float* out_scores) const {
  if (!query) {
    throw std::invalid_argument("query pointer is null");
  }
  if (!out_ids || !out_scores) {
    throw std::invalid_argument("output pointers are null");
  }
  if (k == 0) {
    return;
  }

  const std::size_t kk = std::min(k, size_);

  // Max-heap of current best results by "badness".
  // top() is the worst among the kept candidates, which makes replacement O(log k).
  using Item = std::pair<float, std::uint64_t>; // (badness, id)
  auto worse = [](const Item& a, const Item& b) { return a.first < b.first; };
  std::priority_queue<Item, std::vector<Item>, decltype(worse)> heap(worse);

  for (std::size_t i = 0; i < size_; ++i) {
    const float* vec = embeddings_.data() + (i * dim_);
    const float s = score(query, vec);
    const float b = badness_from_score(metric_, s);

    if (heap.size() < kk) {
      heap.emplace(b, ids_[i]);
    } else if (b < heap.top().first) {
      heap.pop();
      heap.emplace(b, ids_[i]);
    }
  }

  // Extract heap into output arrays. Heap gives worst-first, so we reverse.
  // For L2: best has smallest distance; for IP: best has largest similarity.
  std::vector<Item> tmp;
  tmp.reserve(kk);
  while (!heap.empty()) {
    tmp.push_back(heap.top());
    heap.pop();
  }
  std::reverse(tmp.begin(), tmp.end());

  for (std::size_t i = 0; i < kk; ++i) {
    out_ids[i] = tmp[i].second;

    // Convert back from badness to the user-facing score.
    out_scores[i] = (metric_ == Metric::L2_SQUARED) ? tmp[i].first : -tmp[i].first;
  }

  // If caller asked for more than size_, pad deterministically.
  for (std::size_t i = kk; i < k; ++i) {
    out_ids[i] = std::numeric_limits<std::uint64_t>::max();
    out_scores[i] = std::numeric_limits<float>::infinity();
  }
}

} // namespace vectrax
