#include "vectorcore/hnsw_index.h"

#include <algorithm>
#include <limits>
#include <queue>
#include <stdexcept>

namespace vectorcore {

namespace {
inline float badness_from_score(Metric metric, float score) noexcept {
  return (metric == Metric::L2_SQUARED) ? score : -score;
}
}

HnswIndex::HnswIndex(std::size_t dim, std::size_t M, Metric metric)
    : dim_(dim), M_(M), metric_(metric) {
  if (dim_ == 0) {
    throw std::invalid_argument("dim must be > 0");
  }
  if (M_ == 0) {
    throw std::invalid_argument("M must be > 0");
  }
}

float HnswIndex::score(const float* a, const float* b) const noexcept {
  switch (metric_) {
    case Metric::L2_SQUARED:
      return l2_squared(a, b, dim_);
    case Metric::INNER_PRODUCT:
      return inner_product(a, b, dim_);
    default:
      return l2_squared(a, b, dim_);
  }
}

void HnswIndex::add(const float* vectors, std::size_t n, const std::uint64_t* ids) {
  if (!vectors) {
    throw std::invalid_argument("vectors pointer is null");
  }
  if (n == 0) {
    return;
  }

  const std::size_t old_size = size_;
  const std::size_t new_size = size_ + n;

  embeddings_.reserve(new_size * dim_);
  ids_.reserve(new_size);
  neighbors_.reserve(new_size);

  // Insert vectors first.
  embeddings_.insert(embeddings_.end(), vectors, vectors + (n * dim_));

  if (ids) {
    ids_.insert(ids_.end(), ids, ids + n);
  } else {
    for (std::size_t i = 0; i < n; ++i) {
      ids_.push_back(static_cast<std::uint64_t>(old_size + i));
    }
  }

  // Build a very small "HNSW-like" graph: connect each new node to its M nearest
  // existing nodes (by brute force). This is not the full algorithm, but it gives
  // you a concrete graph structure and a real approximate search routine.
  for (std::size_t local = 0; local < n; ++local) {
    const std::size_t idx = old_size + local;
    const float* v = embeddings_.data() + (idx * dim_);

    neighbors_.push_back({});

    if (idx == 0) {
      continue; // first node has no neighbors
    }

    using Candidate = std::pair<float, std::uint32_t>; // (badness, neighbor index)
    auto worse = [](const Candidate& a, const Candidate& b) { return a.first < b.first; };
    std::priority_queue<Candidate, std::vector<Candidate>, decltype(worse)> heap(worse);

    const std::size_t max_neighbors = std::min<std::size_t>(M_, idx);

    for (std::size_t j = 0; j < idx; ++j) {
      const float* u = embeddings_.data() + (j * dim_);
      const float s = score(v, u);
      const float b = badness_from_score(metric_, s);

      if (heap.size() < max_neighbors) {
        heap.emplace(b, static_cast<std::uint32_t>(j));
      } else if (b < heap.top().first) {
        heap.pop();
        heap.emplace(b, static_cast<std::uint32_t>(j));
      }
    }

    auto& adj = neighbors_.back();
    adj.reserve(max_neighbors);
    while (!heap.empty()) {
      adj.push_back(heap.top().second);
      heap.pop();
    }

    // Make edges bidirectional (again, simplified vs true HNSW pruning).
    for (const std::uint32_t nb : adj) {
      auto& back = neighbors_[nb];
      if (back.size() < M_) {
        back.push_back(static_cast<std::uint32_t>(idx));
      }
    }
  }

  size_ = new_size;
}

void HnswIndex::search(const float* query, std::size_t k, std::uint64_t* out_ids, float* out_scores) const {
  if (!query) {
    throw std::invalid_argument("query pointer is null");
  }
  if (!out_ids || !out_scores) {
    throw std::invalid_argument("output pointers are null");
  }
  if (k == 0) {
    return;
  }

  if (size_ == 0) {
    for (std::size_t i = 0; i < k; ++i) {
      out_ids[i] = std::numeric_limits<std::uint64_t>::max();
      out_scores[i] = std::numeric_limits<float>::infinity();
    }
    return;
  }

  // Greedy best-first exploration from entrypoint 0.
  // Real HNSW uses efSearch, candidate queues, and multi-level routing.
  const std::size_t kk = std::min(k, size_);

  using Node = std::pair<float, std::uint32_t>; // (badness, index)

  auto better = [](const Node& a, const Node& b) { return a.first > b.first; };
  std::priority_queue<Node, std::vector<Node>, decltype(better)> candidates(better);

  std::vector<std::uint8_t> visited(size_, 0);

  auto push = [&](std::uint32_t idx) {
    if (visited[idx]) {
      return;
    }
    visited[idx] = 1;
    const float* v = embeddings_.data() + (static_cast<std::size_t>(idx) * dim_);
    const float s = score(query, v);
    const float b = badness_from_score(metric_, s);
    candidates.emplace(b, idx);
  };

  push(0);

  // Keep a max-heap of the best results.
  auto worse = [](const Node& a, const Node& b) { return a.first < b.first; };
  std::priority_queue<Node, std::vector<Node>, decltype(worse)> best(worse);

  // Fixed exploration budget for the prototype.
  const std::size_t ef = std::min<std::size_t>(64, size_);

  while (!candidates.empty() && visited.size() > 0 && (best.size() < ef)) {
    const auto [b, idx] = candidates.top();
    candidates.pop();

    if (best.size() < ef) {
      best.emplace(b, idx);
    } else if (b < best.top().first) {
      best.pop();
      best.emplace(b, idx);
    }

    for (const std::uint32_t nb : neighbors_[idx]) {
      push(nb);
    }
  }

  // Convert best set to top-k.
  std::vector<Node> tmp;
  tmp.reserve(best.size());
  while (!best.empty()) {
    tmp.push_back(best.top());
    best.pop();
  }

  std::sort(tmp.begin(), tmp.end(), [](const Node& a, const Node& b) { return a.first < b.first; });

  for (std::size_t i = 0; i < kk; ++i) {
    const auto idx = tmp[i].second;
    out_ids[i] = ids_[idx];
    out_scores[i] = (metric_ == Metric::L2_SQUARED) ? tmp[i].first : -tmp[i].first;
  }

  for (std::size_t i = kk; i < k; ++i) {
    out_ids[i] = std::numeric_limits<std::uint64_t>::max();
    out_scores[i] = std::numeric_limits<float>::infinity();
  }
}

} // namespace vectorcore
