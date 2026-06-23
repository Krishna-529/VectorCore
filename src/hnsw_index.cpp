#include "vectorcore/hnsw_index.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <stdexcept>

namespace vectorcore {

namespace {
constexpr std::uint64_t kNoId = std::numeric_limits<std::uint64_t>::max();
constexpr float kInf = std::numeric_limits<float>::infinity();
}  // namespace

HnswIndex::HnswIndex(std::size_t dim, std::size_t M, Metric metric,
                     std::size_t ef_construction, std::uint64_t seed)
    : dim_(dim), M_(M), M_max0_(2 * M), ef_construction_(ef_construction),
      metric_(metric), rng_(seed) {
  if (dim_ == 0) {
    throw std::invalid_argument("dim must be > 0");
  }
  if (M_ < 2) {
    throw std::invalid_argument("M must be >= 2");
  }
  if (ef_construction_ < 1) {
    throw std::invalid_argument("ef_construction must be >= 1");
  }
  mL_ = 1.0 / std::log(static_cast<double>(M_));
  ef_search_ = std::max<std::size_t>(ef_construction_ / 4, 16);
}

float HnswIndex::badness(const float* a, const float* b) const noexcept {
  switch (metric_) {
    case Metric::L2_SQUARED:
      return l2_squared(a, b, dim_);
    case Metric::INNER_PRODUCT:
    case Metric::COSINE:
      return -inner_product(a, b, dim_);
    default:
      return l2_squared(a, b, dim_);
  }
}

int HnswIndex::random_level() {
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  double r = dist(rng_);
  if (r < 1e-12) {
    r = 1e-12;
  }
  return static_cast<int>(-std::log(r) * mL_);
}

// Beam search within `layer`. Maintains:
// - `candidates`: min-heap by badness (closest first) of nodes left to expand.
// - `result`:     max-heap by badness (farthest first), the running top-ef set.
std::vector<HnswIndex::Candidate> HnswIndex::search_layer(
    const float* q, const std::vector<std::uint32_t>& entry_points, std::size_t ef,
    int layer, std::size_t node_count) const {
  // O(1) visited reset via a monotonically increasing stamp.
  if (visited_.size() < node_count) {
    visited_.assign(node_count, 0);
    visit_stamp_ = 0;
  }
  if (++visit_stamp_ == 0) {  // wrapped around: clear and restart
    std::fill(visited_.begin(), visited_.end(), 0);
    visit_stamp_ = 1;
  }
  const std::uint32_t stamp = visit_stamp_;
  auto is_visited = [&](std::uint32_t e) { return visited_[e] == stamp; };
  auto mark = [&](std::uint32_t e) { visited_[e] = stamp; };

  auto nearer = [](const Candidate& a, const Candidate& b) { return a.first > b.first; };
  auto farther = [](const Candidate& a, const Candidate& b) { return a.first < b.first; };
  std::priority_queue<Candidate, std::vector<Candidate>, decltype(nearer)> candidates(nearer);
  std::priority_queue<Candidate, std::vector<Candidate>, decltype(farther)> result(farther);

  for (std::uint32_t ep : entry_points) {
    if (ep >= node_count || is_visited(ep)) {
      continue;
    }
    mark(ep);
    const float d = badness(q, vec(ep));
    candidates.emplace(d, ep);
    result.emplace(d, ep);
  }
  while (result.size() > ef) {
    result.pop();
  }

  while (!candidates.empty()) {
    const Candidate cur = candidates.top();
    candidates.pop();

    const float farthest = result.empty() ? kInf : result.top().first;
    if (cur.first > farthest && result.size() >= ef) {
      break;  // nearest remaining candidate is worse than our worst kept result
    }

    const std::uint32_t c = cur.second;
    if (layer >= static_cast<int>(links_[c].size())) {
      continue;  // node not present at this layer (defensive)
    }
    for (std::uint32_t e : links_[c][layer]) {
      if (e >= node_count || is_visited(e)) {
        continue;
      }
      mark(e);
      const float d = badness(q, vec(e));
      const float worst = result.empty() ? kInf : result.top().first;
      if (d < worst || result.size() < ef) {
        candidates.emplace(d, e);
        result.emplace(d, e);
        if (result.size() > ef) {
          result.pop();
        }
      }
    }
  }

  std::vector<Candidate> out;
  out.reserve(result.size());
  while (!result.empty()) {
    out.push_back(result.top());
    result.pop();
  }
  return out;
}

// HNSW Algorithm 4: keep candidate `e` only if it is closer to the base element
// than to every neighbor already selected. This builds an approximate relative
// neighborhood graph (diverse edges), which keeps the graph navigable.
void HnswIndex::select_neighbors_heuristic(std::vector<Candidate>& candidates,
                                           std::size_t M,
                                           std::vector<std::uint32_t>& out) const {
  std::sort(candidates.begin(), candidates.end(),
            [](const Candidate& a, const Candidate& b) { return a.first < b.first; });

  out.clear();
  out.reserve(M);
  for (const auto& cand : candidates) {
    if (out.size() >= M) {
      break;
    }
    const float dist_to_base = cand.first;
    const float* ve = vec(cand.second);
    bool keep = true;
    for (std::uint32_t r : out) {
      if (badness(ve, vec(r)) < dist_to_base) {
        keep = false;  // e is closer to an already-selected neighbor than to base
        break;
      }
    }
    if (keep) {
      out.push_back(cand.second);
    }
  }

  // Backfill with the nearest pruned candidates if we ended up under M, so the
  // graph stays well-connected (keepPrunedConnections).
  if (out.size() < M) {
    for (const auto& cand : candidates) {
      if (out.size() >= M) {
        break;
      }
      if (std::find(out.begin(), out.end(), cand.second) == out.end()) {
        out.push_back(cand.second);
      }
    }
  }
}

void HnswIndex::insert_one(std::uint32_t idx) {
  const int level = random_level();
  node_level_[idx] = level;
  links_[idx].assign(static_cast<std::size_t>(level) + 1, {});

  // First node ever: becomes the entry point, no edges to add.
  if (size_ == 0) {
    entry_point_ = idx;
    max_level_ = level;
    return;
  }

  const float* q = vec(idx);
  std::uint32_t ep = entry_point_;
  const int top = max_level_;

  // Phase 1: greedy descent through layers above this node's level (ef = 1).
  for (int lc = top; lc > level; --lc) {
    auto w = search_layer(q, {ep}, 1, lc, /*node_count=*/idx);
    if (!w.empty()) {
      ep = std::min_element(w.begin(), w.end(),
                            [](const Candidate& a, const Candidate& b) {
                              return a.first < b.first;
                            })->second;
    }
  }

  // Phase 2: from min(top, level) down to 0, find neighbors and link.
  std::vector<std::uint32_t> entry_points = {ep};
  for (int lc = std::min(top, level); lc >= 0; --lc) {
    auto w = search_layer(q, entry_points, ef_construction_, lc, /*node_count=*/idx);

    std::vector<std::uint32_t> neighbors;
    select_neighbors_heuristic(w, M_, neighbors);
    links_[idx][lc] = neighbors;

    const std::size_t m_max = (lc == 0) ? M_max0_ : M_;
    for (std::uint32_t e : neighbors) {
      auto& e_conn = links_[e][lc];
      e_conn.push_back(idx);
      if (e_conn.size() > m_max) {
        // Re-run the heuristic on e's (now over-full) neighbor list.
        std::vector<Candidate> cand;
        cand.reserve(e_conn.size());
        const float* ve = vec(e);
        for (std::uint32_t nb : e_conn) {
          cand.emplace_back(badness(ve, vec(nb)), nb);
        }
        std::vector<std::uint32_t> pruned;
        select_neighbors_heuristic(cand, m_max, pruned);
        e_conn = std::move(pruned);
      }
    }

    // Entry points for the next layer down are this layer's found candidates.
    entry_points.clear();
    entry_points.reserve(w.size());
    for (const auto& c : w) {
      entry_points.push_back(c.second);
    }
    if (entry_points.empty()) {
      entry_points.push_back(ep);
    }
  }

  if (level > max_level_) {
    max_level_ = level;
    entry_point_ = idx;
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
  node_level_.resize(new_size);
  links_.resize(new_size);
  // Size the visited set to full capacity once (avoids per-call reallocation).
  visited_.assign(new_size, 0);
  visit_stamp_ = 0;

  embeddings_.insert(embeddings_.end(), vectors, vectors + (n * dim_));

  // Cosine: store unit-length vectors so the graph is built in cosine geometry.
  if (metric_ == Metric::COSINE) {
    for (std::size_t i = 0; i < n; ++i) {
      l2_normalize_inplace(embeddings_.data() + ((old_size + i) * dim_), dim_);
    }
  }

  if (ids) {
    ids_.insert(ids_.end(), ids, ids + n);
  } else {
    for (std::size_t i = 0; i < n; ++i) {
      ids_.push_back(static_cast<std::uint64_t>(old_size + i));
    }
  }

  // Insert one node at a time; size_ tracks how many nodes are live so that
  // search_layer's visited bitmap is correctly bounded during construction.
  for (std::size_t i = 0; i < n; ++i) {
    insert_one(static_cast<std::uint32_t>(old_size + i));
    size_ = old_size + i + 1;
  }
}

void HnswIndex::search(const float* query, std::size_t k, std::uint64_t* out_ids,
                       float* out_scores) const {
  if (!query) {
    throw std::invalid_argument("query pointer is null");
  }
  if (!out_ids || !out_scores) {
    throw std::invalid_argument("output pointers are null");
  }
  if (k == 0) {
    return;
  }

  auto pad_from = [&](std::size_t start) {
    for (std::size_t i = start; i < k; ++i) {
      out_ids[i] = kNoId;
      out_scores[i] = kInf;
    }
  };

  if (size_ == 0) {
    pad_from(0);
    return;
  }

  // Cosine: normalize the query into a local buffer.
  std::vector<float> query_norm;
  if (metric_ == Metric::COSINE) {
    query_norm.assign(query, query + dim_);
    l2_normalize_inplace(query_norm.data(), dim_);
    query = query_norm.data();
  }

  std::uint32_t ep = entry_point_;
  for (int lc = max_level_; lc > 0; --lc) {
    auto w = search_layer(query, {ep}, 1, lc, size_);
    if (!w.empty()) {
      ep = std::min_element(w.begin(), w.end(),
                            [](const Candidate& a, const Candidate& b) {
                              return a.first < b.first;
                            })->second;
    }
  }

  const std::size_t ef = std::max(ef_search_, k);
  auto w = search_layer(query, {ep}, ef, 0, size_);

  std::sort(w.begin(), w.end(),
            [](const Candidate& a, const Candidate& b) { return a.first < b.first; });

  const std::size_t kk = std::min(k, w.size());
  for (std::size_t i = 0; i < kk; ++i) {
    out_ids[i] = ids_[w[i].second];
    out_scores[i] = (metric_ == Metric::L2_SQUARED) ? w[i].first : -w[i].first;
  }
  pad_from(kk);
}

}  // namespace vectorcore
