// Persistence round-trip tests: save -> load must reproduce identical search
// results for every index type.

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "vectorcore/bruteforce_index.h"
#include "vectorcore/hnsw_index.h"
#include "vectorcore/pq_index.h"

namespace {

using vectorcore::BruteForceIndex;
using vectorcore::HnswIndex;
using vectorcore::Metric;
using vectorcore::PQIndex;

std::vector<float> random_data(std::size_t n, std::size_t dim, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> d(-1.0f, 1.0f);
  std::vector<float> v(n * dim);
  for (float& x : v) x = d(rng);
  return v;
}

// A unique temp path per test (cleaned up at the end of each case).
std::string tmp_path(const char* tag) {
  return std::string("vectorcore_io_test_") + tag + ".bin";
}

TEST(IO, BruteForceRoundTrip) {
  const std::size_t dim = 16, n = 500, k = 10;
  const auto data = random_data(n, dim, 1);

  BruteForceIndex a(dim, Metric::COSINE);
  a.add(data.data(), n);
  const std::string path = tmp_path("bf");
  a.save(path);

  BruteForceIndex b = BruteForceIndex::load(path);
  ASSERT_EQ(b.size(), a.size());
  EXPECT_EQ(b.dim(), a.dim());

  const auto q = random_data(20, dim, 2);
  std::uint64_t ia[k], ib[k];
  float sa[k], sb[k];
  for (std::size_t i = 0; i < 20; ++i) {
    a.search(q.data() + i * dim, k, ia, sa);
    b.search(q.data() + i * dim, k, ib, sb);
    for (std::size_t j = 0; j < k; ++j) {
      EXPECT_EQ(ia[j], ib[j]);
      EXPECT_FLOAT_EQ(sa[j], sb[j]);
    }
  }
  std::remove(path.c_str());
}

TEST(IO, HnswRoundTrip) {
  const std::size_t dim = 24, n = 1000, k = 10;
  const auto data = random_data(n, dim, 5);

  HnswIndex a(dim, 16, Metric::L2_SQUARED, 200, /*seed=*/77);
  a.set_ef_search(80);
  a.add(data.data(), n);
  const std::string path = tmp_path("hnsw");
  a.save(path);

  HnswIndex b = HnswIndex::load(path);
  ASSERT_EQ(b.size(), a.size());
  EXPECT_EQ(b.dim(), a.dim());
  EXPECT_EQ(b.ef_search(), a.ef_search());

  const auto q = random_data(30, dim, 6);
  std::uint64_t ia[k], ib[k];
  float sa[k], sb[k];
  for (std::size_t i = 0; i < 30; ++i) {
    a.search(q.data() + i * dim, k, ia, sa);
    b.search(q.data() + i * dim, k, ib, sb);
    for (std::size_t j = 0; j < k; ++j) {
      EXPECT_EQ(ia[j], ib[j]) << "query " << i << " rank " << j;
      EXPECT_FLOAT_EQ(sa[j], sb[j]);
    }
  }
  std::remove(path.c_str());
}

TEST(IO, PQRoundTrip) {
  const std::size_t dim = 32, n = 800, k = 10;
  const auto data = random_data(n, dim, 9);

  PQIndex a(dim, 8, Metric::L2_SQUARED);
  a.train(data.data(), n);
  a.add(data.data(), n);
  const std::string path = tmp_path("pq");
  a.save(path);

  PQIndex b = PQIndex::load(path);
  ASSERT_EQ(b.size(), a.size());
  EXPECT_EQ(b.m(), a.m());
  EXPECT_EQ(b.ksub(), a.ksub());
  EXPECT_TRUE(b.is_trained());

  const auto q = random_data(25, dim, 10);
  std::uint64_t ia[k], ib[k];
  float sa[k], sb[k];
  for (std::size_t i = 0; i < 25; ++i) {
    a.search(q.data() + i * dim, k, ia, sa);
    b.search(q.data() + i * dim, k, ib, sb);
    for (std::size_t j = 0; j < k; ++j) {
      EXPECT_EQ(ia[j], ib[j]);
      EXPECT_FLOAT_EQ(sa[j], sb[j]);
    }
  }
  std::remove(path.c_str());
}

TEST(IO, WrongMagicThrows) {
  const std::size_t dim = 8;
  const auto data = random_data(10, dim, 3);
  BruteForceIndex bf(dim, Metric::L2_SQUARED);
  bf.add(data.data(), 10);
  const std::string path = tmp_path("magic");
  bf.save(path);
  // A BruteForce file is not a valid HNSW file.
  EXPECT_THROW(HnswIndex::load(path), std::runtime_error);
  std::remove(path.c_str());
}

}  // namespace
