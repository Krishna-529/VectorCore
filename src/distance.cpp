#include "vectorcore/distance.h"

#include <immintrin.h>

namespace vectorcore {

float l2_squared_scalar(const float* a, const float* b, std::size_t dim) noexcept {
  float acc = 0.0f;
  for (std::size_t i = 0; i < dim; ++i) {
    const float d = a[i] - b[i];
    acc += d * d;
  }
  return acc;
}

float inner_product_scalar(const float* a, const float* b, std::size_t dim) noexcept {
  float acc = 0.0f;
  for (std::size_t i = 0; i < dim; ++i) {
    acc += a[i] * b[i];
  }
  return acc;
}

// AVX2 kernels
// Note: We keep these functions available even when AVX2 isn't enabled;
// the dispatcher will call scalar fallbacks when __AVX2__ is not defined.

float l2_squared_avx2(const float* a, const float* b, std::size_t dim) noexcept {
#if defined(__AVX2__)
  __m256 sum = _mm256_setzero_ps();
  std::size_t i = 0;

  // Process 8 floats per iteration.
  for (; i + 8 <= dim; i += 8) {
    const __m256 va = _mm256_loadu_ps(a + i);
    const __m256 vb = _mm256_loadu_ps(b + i);
    const __m256 diff = _mm256_sub_ps(va, vb);
    sum = _mm256_fmadd_ps(diff, diff, sum); // sum += diff * diff
  }

  // Horizontal sum of sum's lanes.
  alignas(32) float tmp[8];
  _mm256_store_ps(tmp, sum);
  float acc = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];

  // Tail.
  for (; i < dim; ++i) {
    const float d = a[i] - b[i];
    acc += d * d;
  }

  return acc;
#else
  (void)a;
  (void)b;
  (void)dim;
  return 0.0f;
#endif
}

float inner_product_avx2(const float* a, const float* b, std::size_t dim) noexcept {
#if defined(__AVX2__)
  __m256 sum = _mm256_setzero_ps();
  std::size_t i = 0;

  for (; i + 8 <= dim; i += 8) {
    const __m256 va = _mm256_loadu_ps(a + i);
    const __m256 vb = _mm256_loadu_ps(b + i);
    sum = _mm256_fmadd_ps(va, vb, sum);
  }

  alignas(32) float tmp[8];
  _mm256_store_ps(tmp, sum);
  float acc = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];

  for (; i < dim; ++i) {
    acc += a[i] * b[i];
  }

  return acc;
#else
  (void)a;
  (void)b;
  (void)dim;
  return 0.0f;
#endif
}

float l2_squared(const float* a, const float* b, std::size_t dim) noexcept {
#if defined(__AVX2__)
  return l2_squared_avx2(a, b, dim);
#else
  return l2_squared_scalar(a, b, dim);
#endif
}

float inner_product(const float* a, const float* b, std::size_t dim) noexcept {
#if defined(__AVX2__)
  return inner_product_avx2(a, b, dim);
#else
  return inner_product_scalar(a, b, dim);
#endif
}

} // namespace vectorcore
