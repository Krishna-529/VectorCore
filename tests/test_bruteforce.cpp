// Correctness and edge-case tests for BruteForceIndex.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#include "vectorcore/bruteforce_index.h"

namespace {

using vectorcore::BruteForceIndex;
using vectorcore::Metric;

constexpr std::uint64_t kNoId = std::numeric_limits<std::uint64_t>::max();

TEST(BruteForce, CtorRejectsZeroDim) {
  EXPECT_THROW(BruteForceIndex(0, Metric::L2_SQUARED), std::invalid_argument);
}

TEST(BruteForce, L2FindsNearestInOrder) {
  constexpr std::size_t dim = 4;
  BruteForceIndex index(dim, Metric::L2_SQUARED);

  const float data[] = {
      0.f, 0.f, 0.f, 0.f,  // id 0
      1.f, 0.f, 0.f, 0.f,  // id 1
      5.f, 0.f, 0.f, 0.f,  // id 2
  };
  index.add(data, 3);
  ASSERT_EQ(index.size(), 3u);

  const float q[] = {0.9f, 0.f, 0.f, 0.f};
  std::uint64_t ids[3];
  float scores[3];
  index.search(q, 3, ids, scores);

  EXPECT_EQ(ids[0], 1u);  // closest to 0.9
  EXPECT_EQ(ids[1], 0u);
  EXPECT_EQ(ids[2], 2u);
  // L2 scores must be non-decreasing.
  EXPECT_LE(scores[0], scores[1]);
  EXPECT_LE(scores[1], scores[2]);
}

TEST(BruteForce, DefaultIdsAreSequential) {
  constexpr std::size_t dim = 2;
  BruteForceIndex index(dim, Metric::L2_SQUARED);
  const float data[] = {0.f, 0.f, 1.f, 1.f, 2.f, 2.f};
  index.add(data, 3);  // no ids -> 0,1,2

  const float q[] = {2.f, 2.f};
  std::uint64_t ids[1];
  float scores[1];
  index.search(q, 1, ids, scores);
  EXPECT_EQ(ids[0], 2u);
}

TEST(BruteForce, ExternalIdsArePreserved) {
  constexpr std::size_t dim = 2;
  BruteForceIndex index(dim, Metric::L2_SQUARED);
  const float data[] = {0.f, 0.f, 1.f, 1.f};
  const std::uint64_t ids_in[] = {42u, 99u};
  index.add(data, 2, ids_in);

  const float q[] = {1.f, 1.f};
  std::uint64_t ids[1];
  float scores[1];
  index.search(q, 1, ids, scores);
  EXPECT_EQ(ids[0], 99u);
}

TEST(BruteForce, InnerProductPrefersLargestDot) {
  constexpr std::size_t dim = 2;
  BruteForceIndex index(dim, Metric::INNER_PRODUCT);
  const float data[] = {1.f, 0.f, 10.f, 0.f, 0.f, 1.f};
  index.add(data, 3);

  const float q[] = {1.f, 0.f};
  std::uint64_t ids[1];
  float scores[1];
  index.search(q, 1, ids, scores);
  EXPECT_EQ(ids[0], 1u);             // dot=10 is largest
  EXPECT_NEAR(scores[0], 10.f, 1e-4f);
}

TEST(BruteForce, CosineIsScaleInvariant) {
  constexpr std::size_t dim = 3;
  BruteForceIndex index(dim, Metric::COSINE);
  const float data[] = {
      1.f, 0.f, 0.f,   // id 0, same direction as query
      0.f, 1.f, 0.f,   // id 1, orthogonal
      -1.f, 0.f, 0.f,  // id 2, opposite
  };
  index.add(data, 3);

  // Query is a long vector in the +x direction; cosine ignores magnitude.
  const float q[] = {100.f, 0.f, 0.f};
  std::uint64_t ids[3];
  float scores[3];
  index.search(q, 3, ids, scores);

  EXPECT_EQ(ids[0], 0u);
  EXPECT_NEAR(scores[0], 1.0f, 1e-4f);   // identical direction
  EXPECT_EQ(ids[1], 1u);
  EXPECT_NEAR(scores[1], 0.0f, 1e-4f);   // orthogonal
  EXPECT_EQ(ids[2], 2u);
  EXPECT_NEAR(scores[2], -1.0f, 1e-4f);  // opposite
}

TEST(BruteForce, PadsWhenKExceedsSize) {
  constexpr std::size_t dim = 2;
  BruteForceIndex index(dim, Metric::L2_SQUARED);
  const float data[] = {0.f, 0.f, 1.f, 1.f};
  index.add(data, 2);

  const float q[] = {0.f, 0.f};
  std::uint64_t ids[5];
  float scores[5];
  index.search(q, 5, ids, scores);

  EXPECT_EQ(ids[0], 0u);
  EXPECT_EQ(ids[1], 1u);
  for (int i = 2; i < 5; ++i) {
    EXPECT_EQ(ids[i], kNoId) << "i=" << i;
    EXPECT_TRUE(std::isinf(scores[i])) << "i=" << i;
  }
}

TEST(BruteForce, EmptyIndexReturnsAllPadding) {
  BruteForceIndex index(4, Metric::L2_SQUARED);
  const float q[] = {1.f, 2.f, 3.f, 4.f};
  std::uint64_t ids[3];
  float scores[3];
  index.search(q, 3, ids, scores);
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(ids[i], kNoId);
    EXPECT_TRUE(std::isinf(scores[i]));
  }
}

TEST(BruteForce, NullPointersThrow) {
  BruteForceIndex index(2, Metric::L2_SQUARED);
  const float data[] = {0.f, 0.f};
  index.add(data, 1);

  const float q[] = {0.f, 0.f};
  std::uint64_t ids[1];
  float scores[1];
  EXPECT_THROW(index.search(nullptr, 1, ids, scores), std::invalid_argument);
  EXPECT_THROW(index.search(q, 1, nullptr, scores), std::invalid_argument);
  EXPECT_THROW(index.search(q, 1, ids, nullptr), std::invalid_argument);
  EXPECT_THROW(index.add(nullptr, 1), std::invalid_argument);
}

// Brute force is exact, so its top-k must equal a double-precision reference.
TEST(BruteForce, MatchesReferenceTopK) {
  constexpr std::size_t dim = 32;
  constexpr std::size_t n = 2000;
  constexpr std::size_t k = 10;

  std::mt19937 rng(12345);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  std::vector<float> data(n * dim);
  for (float& v : data) v = dist(rng);

  BruteForceIndex index(dim, Metric::L2_SQUARED);
  index.add(data.data(), n);

  std::vector<float> q(dim);
  for (float& v : q) v = dist(rng);

  // Reference: exact distances in double, argsort ascending.
  std::vector<std::pair<double, std::uint64_t>> ref(n);
  for (std::size_t i = 0; i < n; ++i) {
    double acc = 0.0;
    for (std::size_t d = 0; d < dim; ++d) {
      const double diff = static_cast<double>(q[d]) - static_cast<double>(data[i * dim + d]);
      acc += diff * diff;
    }
    ref[i] = {acc, static_cast<std::uint64_t>(i)};
  }
  std::sort(ref.begin(), ref.end());

  std::uint64_t ids[k];
  float scores[k];
  index.search(q.data(), k, ids, scores);

  for (std::size_t i = 0; i < k; ++i) {
    EXPECT_EQ(ids[i], ref[i].second) << "rank " << i;
    EXPECT_NEAR(scores[i], static_cast<float>(ref[i].first), 1e-2f) << "rank " << i;
  }
}

}  // namespace
