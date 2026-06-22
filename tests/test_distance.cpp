// Unit tests for the distance kernels.
//
// Strategy: compute an independent reference in double precision, then assert
// that BOTH the scalar kernel and the compile-time dispatcher (which selects
// the AVX2 path when available) match it within a float tolerance. This
// validates scalar<->AVX2 parity without ever calling the disabled AVX2 stub
// directly (when AVX2 is off the dispatcher resolves to the scalar kernel).

#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

#include "vectorcore/distance.h"

namespace {

using vectorcore::inner_product;
using vectorcore::inner_product_scalar;
using vectorcore::l2_normalize_inplace;
using vectorcore::l2_squared;
using vectorcore::l2_squared_scalar;

double ref_l2(const std::vector<float>& a, const std::vector<float>& b) {
  double acc = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
    acc += d * d;
  }
  return acc;
}

double ref_ip(const std::vector<float>& a, const std::vector<float>& b) {
  double acc = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    acc += static_cast<double>(a[i]) * static_cast<double>(b[i]);
  }
  return acc;
}

std::pair<std::vector<float>, std::vector<float>> random_pair(std::size_t dim, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> a(dim), b(dim);
  for (std::size_t i = 0; i < dim; ++i) {
    a[i] = dist(rng);
    b[i] = dist(rng);
  }
  return {a, b};
}

// Cover dims that exercise the SIMD body (multiples of 8) and the scalar tail.
const std::vector<std::size_t> kDims = {1, 3, 7, 8, 9, 16, 31, 64, 100, 128, 129, 384};

TEST(Distance, L2ScalarMatchesReference) {
  for (std::size_t dim : kDims) {
    auto [a, b] = random_pair(dim, 1u + static_cast<unsigned>(dim));
    const double ref = ref_l2(a, b);
    EXPECT_NEAR(l2_squared_scalar(a.data(), b.data(), dim), ref, 1e-3 + 1e-4 * ref)
        << "dim=" << dim;
  }
}

TEST(Distance, L2DispatcherMatchesReference) {
  // When AVX2 is enabled, this exercises the AVX2 path and asserts parity.
  for (std::size_t dim : kDims) {
    auto [a, b] = random_pair(dim, 100u + static_cast<unsigned>(dim));
    const double ref = ref_l2(a, b);
    EXPECT_NEAR(l2_squared(a.data(), b.data(), dim), ref, 1e-3 + 1e-4 * ref)
        << "dim=" << dim;
  }
}

TEST(Distance, ScalarAndDispatcherAgree) {
  for (std::size_t dim : kDims) {
    auto [a, b] = random_pair(dim, 7u + static_cast<unsigned>(dim));
    EXPECT_NEAR(l2_squared(a.data(), b.data(), dim),
                l2_squared_scalar(a.data(), b.data(), dim), 1e-3)
        << "l2 dim=" << dim;
    EXPECT_NEAR(inner_product(a.data(), b.data(), dim),
                inner_product_scalar(a.data(), b.data(), dim), 1e-3)
        << "ip dim=" << dim;
  }
}

TEST(Distance, InnerProductMatchesReference) {
  for (std::size_t dim : kDims) {
    auto [a, b] = random_pair(dim, 50u + static_cast<unsigned>(dim));
    const double ref = ref_ip(a, b);
    EXPECT_NEAR(inner_product(a.data(), b.data(), dim), ref, 1e-3 + 1e-4 * std::abs(ref))
        << "dim=" << dim;
  }
}

TEST(Distance, L2OfIdenticalVectorsIsZero) {
  auto [a, b] = random_pair(128, 3u);
  EXPECT_NEAR(l2_squared(a.data(), a.data(), 128), 0.0f, 1e-5f);
}

TEST(Normalize, ProducesUnitLength) {
  auto [a, b] = random_pair(96, 11u);
  l2_normalize_inplace(a.data(), 96);
  const float self_ip = inner_product(a.data(), a.data(), 96);
  EXPECT_NEAR(self_ip, 1.0f, 1e-4f);
}

TEST(Normalize, ZeroVectorLeftUntouched) {
  std::vector<float> z(32, 0.0f);
  l2_normalize_inplace(z.data(), 32);
  for (float v : z) {
    EXPECT_EQ(v, 0.0f);
  }
}

}  // namespace
