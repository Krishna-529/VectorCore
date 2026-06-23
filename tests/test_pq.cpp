// Tests for PQIndex (Product Quantization + ADC search).
//
// PQ is lossy, so we assert structural guarantees plus a recall floor against
// an exact brute-force reference on clustered data (where quantization is well
// behaved).

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <random>
#include <set>
#include <vector>

#include "vectorcore/bruteforce_index.h"
#include "vectorcore/pq_index.h"

namespace {

using vectorcore::BruteForceIndex;
using vectorcore::Metric;
using vectorcore::PQIndex;

constexpr std::uint64_t kNoId = std::numeric_limits<std::uint64_t>::max();

// Clustered data: PQ shines when sub-vectors fall into compact groups.
std::vector<float> clustered_data(std::size_t n, std::size_t dim, std::size_t clusters,
                                  unsigned seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> jitter(0.0f, 0.05f);
  std::uniform_real_distribution<float> center(-1.0f, 1.0f);

  std::vector<float> centers(clusters * dim);
  for (float& c : centers) c = center(rng);

  std::uniform_int_distribution<std::size_t> pick(0, clusters - 1);
  std::vector<float> data(n * dim);
  for (std::size_t i = 0; i < n; ++i) {
    const std::size_t cl = pick(rng);
    for (std::size_t j = 0; j < dim; ++j) {
      data[i * dim + j] = centers[cl * dim + j] + jitter(rng);
    }
  }
  return data;
}

TEST(PQ, CtorValidatesParams) {
  EXPECT_THROW(PQIndex(0, 4), std::invalid_argument);       // dim 0
  EXPECT_THROW(PQIndex(10, 3), std::invalid_argument);      // 10 % 3 != 0
  EXPECT_THROW(PQIndex(16, 0), std::invalid_argument);      // m 0
  EXPECT_THROW(PQIndex(16, 4, Metric::L2_SQUARED, 9), std::invalid_argument);  // nbits>8
  EXPECT_NO_THROW(PQIndex(16, 4));
}

TEST(PQ, CodeSizeAndShape) {
  PQIndex pq(32, 8, Metric::L2_SQUARED, /*nbits=*/8);
  EXPECT_EQ(pq.code_size(), 8u);  // m bytes per vector
  EXPECT_EQ(pq.ksub(), 256u);
  EXPECT_EQ(pq.m(), 8u);
}

TEST(PQ, RequiresTrainingBeforeAddAndSearch) {
  PQIndex pq(16, 4);
  std::vector<float> v(16, 0.5f);
  EXPECT_THROW(pq.add(v.data(), 1), std::invalid_argument);

  std::uint64_t ids[1];
  float sc[1];
  EXPECT_THROW(pq.search(v.data(), 1, ids, sc), std::invalid_argument);
  EXPECT_FALSE(pq.is_trained());
}

TEST(PQ, TrainEncodesAndPadsBeyondSize) {
  const std::size_t dim = 16, n = 500;
  const auto data = clustered_data(n, dim, 8, 1);

  PQIndex pq(dim, 4, Metric::L2_SQUARED);
  pq.train(data.data(), n);
  EXPECT_TRUE(pq.is_trained());
  pq.add(data.data(), n);
  EXPECT_EQ(pq.size(), n);

  std::uint64_t ids[3];
  float scores[3];
  pq.search(data.data(), 3, ids, scores);
  // Self-query: the top hit should be the vector itself (id 0) in clustered data.
  EXPECT_EQ(ids[0], 0u);

  // k beyond size pads.
  std::vector<std::uint64_t> ids2(n + 2);
  std::vector<float> sc2(n + 2);
  pq.search(data.data(), n + 2, ids2.data(), sc2.data());
  EXPECT_EQ(ids2[n], kNoId);
  EXPECT_EQ(ids2[n + 1], kNoId);
}

// Recall@10 of PQ+ADC vs exact brute force. Uses Gaussian data with 2-dim
// subspaces (fine quantization) so true neighbors are distinguishable — tight
// clusters produce near-duplicate points whose exact order PQ cannot (and need
// not) resolve, which would make exact-id recall meaningless.
TEST(PQ, HighRecallVsBruteForce) {
  const std::size_t dim = 32, n = 3000, k = 10, nq = 100;
  std::mt19937 gen(7);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::vector<float> data(n * dim);
  for (float& x : data) x = nd(gen);

  BruteForceIndex bf(dim, Metric::L2_SQUARED);
  bf.add(data.data(), n);

  PQIndex pq(dim, /*m=*/16, Metric::L2_SQUARED);  // 32/16 = 2-dim subspaces
  pq.train(data.data(), n);
  pq.add(data.data(), n);

  std::mt19937 rng(123);
  std::uniform_int_distribution<std::size_t> qpick(0, n - 1);

  std::vector<std::uint64_t> bf_ids(k), pq_ids(k);
  std::vector<float> bf_sc(k), pq_sc(k);

  double total = 0.0;
  for (std::size_t t = 0; t < nq; ++t) {
    const float* q = data.data() + qpick(rng) * dim;
    bf.search(q, k, bf_ids.data(), bf_sc.data());
    pq.search(q, k, pq_ids.data(), pq_sc.data());

    std::set<std::uint64_t> truth(bf_ids.begin(), bf_ids.end());
    std::size_t hits = 0;
    for (std::uint64_t id : pq_ids) {
      if (truth.count(id)) ++hits;
    }
    total += static_cast<double>(hits) / static_cast<double>(k);
  }
  const double recall = total / nq;
  EXPECT_GE(recall, 0.80) << "PQ recall@10 = " << recall;
}

TEST(PQ, PreservesExternalIds) {
  const std::size_t dim = 16, n = 400;
  const auto data = clustered_data(n, dim, 8, 3);
  std::vector<std::uint64_t> ids(n);
  for (std::size_t i = 0; i < n; ++i) ids[i] = 7000 + i;

  PQIndex pq(dim, 4);
  pq.train(data.data(), n);
  pq.add(data.data(), n, ids.data());

  std::uint64_t out[5];
  float sc[5];
  pq.search(data.data(), 5, out, sc);
  EXPECT_EQ(out[0], 7000u);
}

}  // namespace
