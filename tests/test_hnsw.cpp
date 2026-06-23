// Correctness tests for the multi-layer HnswIndex.
//
// HNSW is approximate, so we don't assert exact equality with brute force.
// Instead we assert high recall against an exact reference, plus structural
// guarantees (determinism, padding, single-node behavior).

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <random>
#include <set>
#include <vector>

#include "vectorcore/bruteforce_index.h"
#include "vectorcore/hnsw_index.h"

namespace {

using vectorcore::BruteForceIndex;
using vectorcore::HnswIndex;
using vectorcore::Metric;

constexpr std::uint64_t kNoId = std::numeric_limits<std::uint64_t>::max();

std::vector<float> random_data(std::size_t n, std::size_t dim, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> v(n * dim);
  for (float& x : v) x = dist(rng);
  return v;
}

// Mean recall@k of an HNSW index vs. an exact brute-force index over the same
// data, averaged across `n_queries` random queries.
double hnsw_recall(Metric metric, std::size_t n, std::size_t dim, std::size_t k,
                   std::size_t M, std::size_t ef_search, std::size_t n_queries,
                   unsigned seed) {
  const auto data = random_data(n, dim, seed);

  BruteForceIndex bf(dim, metric);
  bf.add(data.data(), n);

  HnswIndex hnsw(dim, M, metric, /*ef_construction=*/200, /*seed=*/42);
  hnsw.set_ef_search(ef_search);
  hnsw.add(data.data(), n);

  const auto queries = random_data(n_queries, dim, seed + 999u);

  std::vector<std::uint64_t> bf_ids(k), hnsw_ids(k);
  std::vector<float> bf_sc(k), hnsw_sc(k);

  double total = 0.0;
  for (std::size_t qi = 0; qi < n_queries; ++qi) {
    const float* q = queries.data() + qi * dim;
    bf.search(q, k, bf_ids.data(), bf_sc.data());
    hnsw.search(q, k, hnsw_ids.data(), hnsw_sc.data());

    std::set<std::uint64_t> truth(bf_ids.begin(), bf_ids.end());
    std::size_t hits = 0;
    for (std::uint64_t id : hnsw_ids) {
      if (truth.count(id)) ++hits;
    }
    total += static_cast<double>(hits) / static_cast<double>(k);
  }
  return total / static_cast<double>(n_queries);
}

TEST(Hnsw, CtorRejectsBadParams) {
  EXPECT_THROW(HnswIndex(0, 16, Metric::L2_SQUARED), std::invalid_argument);
  EXPECT_THROW(HnswIndex(8, 1, Metric::L2_SQUARED), std::invalid_argument);  // M < 2
}

TEST(Hnsw, HighRecallL2) {
  const double recall = hnsw_recall(Metric::L2_SQUARED, /*n=*/2000, /*dim=*/32,
                                    /*k=*/10, /*M=*/16, /*ef_search=*/100,
                                    /*n_queries=*/100, /*seed=*/1);
  EXPECT_GE(recall, 0.95) << "L2 recall@10 = " << recall;
}

TEST(Hnsw, HighRecallCosine) {
  const double recall = hnsw_recall(Metric::COSINE, 2000, 32, 10, 16, 100, 100, 7);
  EXPECT_GE(recall, 0.95) << "cosine recall@10 = " << recall;
}

TEST(Hnsw, HigherEfImprovesOrHoldsRecall) {
  const double low = hnsw_recall(Metric::L2_SQUARED, 2000, 32, 10, 16, 16, 100, 3);
  const double high = hnsw_recall(Metric::L2_SQUARED, 2000, 32, 10, 16, 200, 100, 3);
  EXPECT_GE(high, low - 1e-9) << "low=" << low << " high=" << high;
  EXPECT_GE(high, 0.95);
}

TEST(Hnsw, DeterministicWithFixedSeed) {
  const auto data = random_data(500, 16, 5);
  const float q[16] = {0.1f, -0.2f, 0.3f, 0.0f, 0.5f, -0.5f, 0.2f, 0.1f,
                       -0.1f, 0.4f, -0.3f, 0.2f, 0.0f, 0.1f, -0.4f, 0.3f};

  auto run = []() {
    HnswIndex h(16, 16, Metric::L2_SQUARED, 200, /*seed=*/123);
    h.set_ef_search(50);
    return h;
  };

  HnswIndex a = run();
  HnswIndex b = run();
  a.add(data.data(), 500);
  b.add(data.data(), 500);

  std::uint64_t ia[10], ib[10];
  float sa[10], sb[10];
  a.search(q, 10, ia, sa);
  b.search(q, 10, ib, sb);
  for (int i = 0; i < 10; ++i) {
    EXPECT_EQ(ia[i], ib[i]) << "rank " << i;
  }
}

TEST(Hnsw, SingleNodeReturnsItself) {
  const float v[4] = {1.f, 2.f, 3.f, 4.f};
  HnswIndex h(4, 16, Metric::L2_SQUARED);
  h.add(v, 1);
  ASSERT_EQ(h.size(), 1u);

  std::uint64_t ids[3];
  float scores[3];
  h.search(v, 3, ids, scores);
  EXPECT_EQ(ids[0], 0u);
  EXPECT_NEAR(scores[0], 0.0f, 1e-5f);
  // k > size -> padded
  EXPECT_EQ(ids[1], kNoId);
  EXPECT_EQ(ids[2], kNoId);
}

TEST(Hnsw, EmptyIndexPads) {
  HnswIndex h(4, 16, Metric::L2_SQUARED);
  const float q[4] = {1.f, 1.f, 1.f, 1.f};
  std::uint64_t ids[2];
  float scores[2];
  h.search(q, 2, ids, scores);
  EXPECT_EQ(ids[0], kNoId);
  EXPECT_EQ(ids[1], kNoId);
}

TEST(Hnsw, PreservesExternalIds) {
  const auto data = random_data(300, 8, 11);
  std::vector<std::uint64_t> ids(300);
  for (std::size_t i = 0; i < 300; ++i) ids[i] = 5000 + i;

  HnswIndex h(8, 16, Metric::L2_SQUARED);
  h.set_ef_search(64);
  h.add(data.data(), 300, ids.data());

  std::uint64_t out[5];
  float sc[5];
  h.search(data.data(), 5, out, sc);  // query == vector 0 -> id 5000
  EXPECT_EQ(out[0], 5000u);
}

}  // namespace
