#include "vectorcore/pq_index.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <limits>
#include <queue>
#include <random>
#include <stdexcept>

#include "vectorcore/io.h"

namespace vectorcore {

namespace {

constexpr std::uint64_t kNoId = std::numeric_limits<std::uint64_t>::max();
constexpr float kInf = std::numeric_limits<float>::infinity();

// Index of the nearest centroid (by squared L2) to `v` among `k` centroids.
std::size_t nearest_centroid(const float* v, const float* centroids, std::size_t k,
                             std::size_t d) {
  std::size_t best = 0;
  float best_d = kInf;
  for (std::size_t c = 0; c < k; ++c) {
    const float dist = l2_squared(v, centroids + c * d, d);
    if (dist < best_d) {
      best_d = dist;
      best = c;
    }
  }
  return best;
}

// Lloyd's K-Means with k-means++ initialization on `data` (n x d).
// Writes `k` centroids (k x d) into `centroids`. Always uses squared-L2, which
// is the correct quantization objective even for inner-product/cosine indexes
// (cosine vectors are normalized before training).
void kmeans(const float* data, std::size_t n, std::size_t d, std::size_t k,
            std::size_t iters, std::uint64_t seed, std::vector<float>& centroids) {
  centroids.assign(k * d, 0.0f);
  std::mt19937_64 rng(seed);

  if (n == 0) {
    return;
  }
  // Fewer points than centroids: seed centroids from points (with wrap).
  if (n <= k) {
    for (std::size_t c = 0; c < k; ++c) {
      std::memcpy(&centroids[c * d], data + (c % n) * d, d * sizeof(float));
    }
    return;
  }

  // --- k-means++ seeding ---
  std::uniform_int_distribution<std::size_t> uni(0, n - 1);
  std::size_t first = uni(rng);
  std::memcpy(&centroids[0], data + first * d, d * sizeof(float));

  std::vector<float> nearest_sq(n, kInf);
  for (std::size_t c = 1; c < k; ++c) {
    double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      const float dist = l2_squared(data + i * d, &centroids[(c - 1) * d], d);
      if (dist < nearest_sq[i]) {
        nearest_sq[i] = dist;
      }
      sum += nearest_sq[i];
    }
    std::uniform_real_distribution<double> pick(0.0, sum);
    double target = pick(rng);
    std::size_t chosen = n - 1;
    for (std::size_t i = 0; i < n; ++i) {
      target -= nearest_sq[i];
      if (target <= 0.0) {
        chosen = i;
        break;
      }
    }
    std::memcpy(&centroids[c * d], data + chosen * d, d * sizeof(float));
  }

  // --- Lloyd iterations ---
  std::vector<std::size_t> assign(n, 0);
  std::vector<double> sums(k * d, 0.0);
  std::vector<std::size_t> counts(k, 0);

  for (std::size_t it = 0; it < iters; ++it) {
    std::fill(sums.begin(), sums.end(), 0.0);
    std::fill(counts.begin(), counts.end(), 0);

    for (std::size_t i = 0; i < n; ++i) {
      const std::size_t a = nearest_centroid(data + i * d, centroids.data(), k, d);
      assign[i] = a;
      ++counts[a];
      const float* v = data + i * d;
      double* acc = &sums[a * d];
      for (std::size_t j = 0; j < d; ++j) {
        acc[j] += v[j];
      }
    }

    for (std::size_t c = 0; c < k; ++c) {
      if (counts[c] == 0) {
        // Empty cluster: reseed from a random data point to avoid dead centroids.
        std::memcpy(&centroids[c * d], data + uni(rng) * d, d * sizeof(float));
        continue;
      }
      const double inv = 1.0 / static_cast<double>(counts[c]);
      float* cen = &centroids[c * d];
      const double* acc = &sums[c * d];
      for (std::size_t j = 0; j < d; ++j) {
        cen[j] = static_cast<float>(acc[j] * inv);
      }
    }
  }
}

}  // namespace

PQIndex::PQIndex(std::size_t dim, std::size_t m, Metric metric, std::size_t nbits,
                 std::uint64_t seed)
    : dim_(dim), m_(m), metric_(metric), seed_(seed) {
  if (dim_ == 0) {
    throw std::invalid_argument("dim must be > 0");
  }
  if (m_ == 0 || dim_ % m_ != 0) {
    throw std::invalid_argument("m must be > 0 and divide dim evenly");
  }
  if (nbits == 0 || nbits > 8) {
    throw std::invalid_argument("nbits must be in 1..8 (codes are stored as uint8)");
  }
  dsub_ = dim_ / m_;
  ksub_ = static_cast<std::size_t>(1) << nbits;
}

void PQIndex::train(const float* vectors, std::size_t n, std::size_t kmeans_iters) {
  if (!vectors) {
    throw std::invalid_argument("vectors pointer is null");
  }
  if (n == 0) {
    throw std::invalid_argument("cannot train on 0 vectors");
  }

  // Prepare (optionally normalized) training data once.
  std::vector<float> prepared(vectors, vectors + n * dim_);
  if (metric_ == Metric::COSINE) {
    for (std::size_t i = 0; i < n; ++i) {
      l2_normalize_inplace(&prepared[i * dim_], dim_);
    }
  }

  codebooks_.assign(m_ * ksub_ * dsub_, 0.0f);

  // Train one codebook per subspace on the corresponding slice of each vector.
  std::vector<float> sub(n * dsub_);
  std::vector<float> centroids;
  for (std::size_t s = 0; s < m_; ++s) {
    const std::size_t off = s * dsub_;
    for (std::size_t i = 0; i < n; ++i) {
      std::memcpy(&sub[i * dsub_], &prepared[i * dim_ + off], dsub_ * sizeof(float));
    }
    kmeans(sub.data(), n, dsub_, ksub_, kmeans_iters, seed_ + s, centroids);
    std::memcpy(&codebooks_[s * ksub_ * dsub_], centroids.data(),
                ksub_ * dsub_ * sizeof(float));
  }

  trained_ = true;
}

void PQIndex::encode_one(const float* v, std::uint8_t* out_codes) const {
  for (std::size_t s = 0; s < m_; ++s) {
    const std::size_t c = nearest_centroid(v + s * dsub_, centroid(s, 0), ksub_, dsub_);
    out_codes[s] = static_cast<std::uint8_t>(c);
  }
}

void PQIndex::add(const float* vectors, std::size_t n, const std::uint64_t* ids) {
  if (!trained_) {
    throw std::invalid_argument("PQIndex must be trained before add()");
  }
  if (!vectors) {
    throw std::invalid_argument("vectors pointer is null");
  }
  if (n == 0) {
    return;
  }

  const std::size_t old_size = size_;
  const std::size_t new_size = size_ + n;
  codes_.resize(new_size * m_);
  ids_.reserve(new_size);

  std::vector<float> buf(dim_);
  for (std::size_t i = 0; i < n; ++i) {
    const float* src = vectors + i * dim_;
    if (metric_ == Metric::COSINE) {
      std::memcpy(buf.data(), src, dim_ * sizeof(float));
      l2_normalize_inplace(buf.data(), dim_);
      src = buf.data();
    }
    encode_one(src, &codes_[(old_size + i) * m_]);
  }

  if (ids) {
    ids_.insert(ids_.end(), ids, ids + n);
  } else {
    for (std::size_t i = 0; i < n; ++i) {
      ids_.push_back(static_cast<std::uint64_t>(old_size + i));
    }
  }

  size_ = new_size;
}

void PQIndex::search(const float* query, std::size_t k, std::uint64_t* out_ids,
                     float* out_scores) const {
  if (!trained_) {
    throw std::invalid_argument("PQIndex must be trained before search()");
  }
  if (!query) {
    throw std::invalid_argument("query pointer is null");
  }
  if (!out_ids || !out_scores) {
    throw std::invalid_argument("output pointers are null");
  }
  if (k == 0) {
    return;
  }

  const bool is_l2 = (metric_ == Metric::L2_SQUARED);

  // Normalize the query for cosine.
  std::vector<float> qbuf;
  if (metric_ == Metric::COSINE) {
    qbuf.assign(query, query + dim_);
    l2_normalize_inplace(qbuf.data(), dim_);
    query = qbuf.data();
  }

  // Build the ADC lookup table: lut[s * ksub_ + c] = score(query_sub_s, centroid).
  // L2 uses squared distance (smaller better); IP/cosine use dot (larger better).
  std::vector<float> lut(m_ * ksub_);
  for (std::size_t s = 0; s < m_; ++s) {
    const float* qsub = query + s * dsub_;
    float* row = &lut[s * ksub_];
    for (std::size_t c = 0; c < ksub_; ++c) {
      const float* cen = centroid(s, c);
      row[c] = is_l2 ? l2_squared(qsub, cen, dsub_) : inner_product(qsub, cen, dsub_);
    }
  }

  const std::size_t kk = std::min(k, size_);

  // Max-heap by "badness" (smaller distance / larger similarity = better).
  using Item = std::pair<float, std::uint64_t>;
  auto worse = [](const Item& a, const Item& b) { return a.first < b.first; };
  using MaxHeap = std::priority_queue<Item, std::vector<Item>, decltype(worse)>;
  MaxHeap global_heap(worse);

  const std::ptrdiff_t n_signed = static_cast<std::ptrdiff_t>(size_);

  #pragma omp parallel
  {
    MaxHeap local_heap(worse);

    #pragma omp for nowait
    for (std::ptrdiff_t i = 0; i < n_signed; ++i) {
      const std::uint8_t* code = &codes_[static_cast<std::size_t>(i) * m_];
      float acc = 0.0f;
      for (std::size_t s = 0; s < m_; ++s) {
        acc += lut[s * ksub_ + code[s]];
      }
      const float badness = is_l2 ? acc : -acc;

      if (local_heap.size() < kk) {
        local_heap.emplace(badness, ids_[static_cast<std::size_t>(i)]);
      } else if (badness < local_heap.top().first) {
        local_heap.pop();
        local_heap.emplace(badness, ids_[static_cast<std::size_t>(i)]);
      }
    }

    #pragma omp critical
    {
      while (!local_heap.empty()) {
        const Item& item = local_heap.top();
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

  std::vector<Item> tmp;
  tmp.reserve(global_heap.size());
  while (!global_heap.empty()) {
    tmp.push_back(global_heap.top());
    global_heap.pop();
  }
  std::reverse(tmp.begin(), tmp.end());

  for (std::size_t i = 0; i < kk; ++i) {
    out_ids[i] = tmp[i].second;
    out_scores[i] = is_l2 ? tmp[i].first : -tmp[i].first;
  }
  for (std::size_t i = kk; i < k; ++i) {
    out_ids[i] = kNoId;
    out_scores[i] = kInf;
  }
}

namespace {
constexpr char kPQMagic[5] = "VCPQ";
constexpr std::uint32_t kPQVersion = 1;

std::uint8_t nbits_from_ksub(std::size_t ksub) {
  std::uint8_t nbits = 0;
  while ((static_cast<std::size_t>(1) << nbits) < ksub) {
    ++nbits;
  }
  return nbits;
}
}  // namespace

void PQIndex::save(const std::string& path) const {
  std::ofstream os(path, std::ios::binary);
  if (!os) {
    throw std::runtime_error("PQIndex::save: cannot open " + path);
  }
  io::write_magic(os, kPQMagic);
  io::write_pod(os, kPQVersion);

  io::write_pod(os, static_cast<std::uint64_t>(dim_));
  io::write_pod(os, static_cast<std::uint64_t>(m_));
  io::write_pod(os, nbits_from_ksub(ksub_));
  io::write_pod(os, static_cast<std::uint8_t>(metric_));
  io::write_pod(os, static_cast<std::uint8_t>(trained_ ? 1 : 0));
  io::write_pod(os, seed_);
  io::write_pod(os, static_cast<std::uint64_t>(size_));

  io::write_pod_vector(os, codebooks_);
  io::write_pod_vector(os, codes_);
  io::write_pod_vector(os, ids_);
}

PQIndex PQIndex::load(const std::string& path) {
  std::ifstream is(path, std::ios::binary);
  if (!is) {
    throw std::runtime_error("PQIndex::load: cannot open " + path);
  }
  io::expect_magic(is, kPQMagic);
  if (io::read_pod<std::uint32_t>(is) != kPQVersion) {
    throw std::runtime_error("PQIndex::load: unsupported format version");
  }

  const auto dim = static_cast<std::size_t>(io::read_pod<std::uint64_t>(is));
  const auto m = static_cast<std::size_t>(io::read_pod<std::uint64_t>(is));
  const auto nbits = io::read_pod<std::uint8_t>(is);
  const auto metric = static_cast<Metric>(io::read_pod<std::uint8_t>(is));
  const bool trained = io::read_pod<std::uint8_t>(is) != 0;
  const auto seed = io::read_pod<std::uint64_t>(is);
  const auto size = static_cast<std::size_t>(io::read_pod<std::uint64_t>(is));

  PQIndex idx(dim, m, metric, nbits, seed);
  idx.trained_ = trained;
  idx.size_ = size;
  io::read_pod_vector(is, idx.codebooks_);
  io::read_pod_vector(is, idx.codes_);
  io::read_pod_vector(is, idx.ids_);
  return idx;
}

}  // namespace vectorcore
