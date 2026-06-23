#include "vectorcore/bruteforce_index.h"

#include <algorithm>
#include <cstring>
#include <fstream>

#include "vectorcore/io.h"

namespace vectorcore {

namespace {

// For L2 we want *smaller* scores; for inner product / cosine we want *larger*
// scores. To reuse a single heap structure we store a "badness" value:
// - L2: badness = distance (larger is worse)
// - IP / cosine: badness = -similarity (larger is worse)
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

  // For cosine, store unit-length vectors so the inner product == cosine.
  if (metric_ == Metric::COSINE) {
    for (std::size_t i = 0; i < n; ++i) {
      l2_normalize_inplace(embeddings_.data() + ((old_size + i) * dim_), dim_);
    }
  }

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
    case Metric::COSINE:
      // Cosine vectors are pre-normalized (on add) and the query is normalized
      // in search(), so the raw inner product equals the cosine similarity.
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

  // For cosine, normalize the query once into a local buffer and point at it.
  std::vector<float> query_norm;
  if (metric_ == Metric::COSINE) {
    query_norm.assign(query, query + dim_);
    l2_normalize_inplace(query_norm.data(), dim_);
    query = query_norm.data();
  }

  const std::size_t kk = std::min(k, size_);

  // Max-heap of current best results by "badness".
  // top() is the worst among the kept candidates, which makes replacement O(log k).
  using Item = std::pair<float, std::uint64_t>; // (badness, id)
  auto worse = [](const Item& a, const Item& b) { return a.first < b.first; };
  using MaxHeap = std::priority_queue<Item, std::vector<Item>, decltype(worse)>;

  MaxHeap global_heap(worse);

  // MSVC ships OpenMP 2.0, which requires a *signed* loop index. Use a signed
  // type for the parallel-for counter and cast inside the body.
  const std::ptrdiff_t n_signed = static_cast<std::ptrdiff_t>(size_);

  #pragma omp parallel
  {
    MaxHeap local_heap(worse);

    #pragma omp for nowait
    for (std::ptrdiff_t i = 0; i < n_signed; ++i) {
      const std::size_t idx = static_cast<std::size_t>(i);
      const float* vec = embeddings_.data() + (idx * dim_);
      const float s = score(query, vec);
      const float b = badness_from_score(metric_, s);

      if (local_heap.size() < kk) {
        local_heap.emplace(b, ids_[idx]);
      } else if (b < local_heap.top().first) {
        local_heap.pop();
        local_heap.emplace(b, ids_[idx]);
      }
    }

    #pragma omp critical
    {
      while (!local_heap.empty()) {
        const auto& item = local_heap.top();
        if (global_heap.size() < kk) {
          global_heap.push(item);
        } else if (item.first < global_heap.top().first) {
          global_heap.pop();
          global_heap.push(item);
        }
        local_heap.pop();
      }
    }
  }

  // Extract heap into output arrays. Heap gives worst-first, so we reverse.
  // For L2: best has smallest distance; for IP: best has largest similarity.
  std::vector<Item> tmp;
  tmp.reserve(kk);
  while (!global_heap.empty()) {
    tmp.push_back(global_heap.top());
    global_heap.pop();
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

namespace {
constexpr char kBruteForceMagic[5] = "VCBF";
constexpr std::uint32_t kBruteForceVersion = 1;
}  // namespace

void BruteForceIndex::save(const std::string& path) const {
  std::ofstream os(path, std::ios::binary);
  if (!os) {
    throw std::runtime_error("BruteForceIndex::save: cannot open " + path);
  }
  io::write_magic(os, kBruteForceMagic);
  io::write_pod(os, kBruteForceVersion);
  io::write_pod(os, static_cast<std::uint64_t>(dim_));
  io::write_pod(os, static_cast<std::uint8_t>(metric_));
  io::write_pod_vector(os, embeddings_);
  io::write_pod_vector(os, ids_);
}

BruteForceIndex BruteForceIndex::load(const std::string& path) {
  std::ifstream is(path, std::ios::binary);
  if (!is) {
    throw std::runtime_error("BruteForceIndex::load: cannot open " + path);
  }
  io::expect_magic(is, kBruteForceMagic);
  if (io::read_pod<std::uint32_t>(is) != kBruteForceVersion) {
    throw std::runtime_error("BruteForceIndex::load: unsupported format version");
  }
  const auto dim = static_cast<std::size_t>(io::read_pod<std::uint64_t>(is));
  const auto metric = static_cast<Metric>(io::read_pod<std::uint8_t>(is));

  BruteForceIndex idx(dim, metric);
  io::read_pod_vector(is, idx.embeddings_);
  io::read_pod_vector(is, idx.ids_);
  idx.size_ = idx.ids_.size();
  if (idx.embeddings_.size() != idx.size_ * dim) {
    throw std::runtime_error("BruteForceIndex::load: corrupt file (size mismatch)");
  }
  return idx;
}

} // namespace vectorcore
