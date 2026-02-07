#pragma once

#include <cstddef>
#include <cstdint>

namespace vectorcore {

// We compute distances in float for throughput.
// For ranking (top-k), float precision is typically sufficient for embeddings.
// If you need better numerical stability, we can accumulate into double.

enum class Metric : std::uint8_t {
  L2_SQUARED = 0,
  INNER_PRODUCT = 1,
};

float l2_squared_scalar(const float* a, const float* b, std::size_t dim) noexcept;
float inner_product_scalar(const float* a, const float* b, std::size_t dim) noexcept;

// AVX2 implementations (compiled in conditionally). The dispatcher lives in .cpp.
float l2_squared_avx2(const float* a, const float* b, std::size_t dim) noexcept;
float inner_product_avx2(const float* a, const float* b, std::size_t dim) noexcept;

// Chooses the best available kernel at compile-time.
float l2_squared(const float* a, const float* b, std::size_t dim) noexcept;
float inner_product(const float* a, const float* b, std::size_t dim) noexcept;

} // namespace vectorcore
