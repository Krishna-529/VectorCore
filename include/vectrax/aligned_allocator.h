#pragma once

#include <cstddef>
#include <cstdlib>
#include <new>
#include <type_traits>

#if defined(_MSC_VER)
  #include <malloc.h> // _aligned_malloc, _aligned_free
#endif

namespace vectrax {

// A small aligned allocator for std::vector.
//
// Why this exists (interview-friendly explanation):
// - SIMD loads/stores (AVX2) benefit from aligned memory.
// - Even when we use unaligned loads (safe for any pointer), keeping the
//   underlying storage aligned reduces the chance that a vector spans cache
//   lines and improves prefetch behavior.
// - We still keep a *flat* memory model: a single contiguous vector of floats.
//
// Alignment is a compile-time constant so the optimizer can reason about it.

template <typename T, std::size_t Alignment>
class AlignedAllocator {
  static_assert(Alignment >= alignof(T), "Alignment must satisfy element alignment");
  static_assert((Alignment & (Alignment - 1)) == 0, "Alignment must be power of two");

public:
  using value_type = T;

  AlignedAllocator() noexcept = default;

  template <typename U>
  AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

  [[nodiscard]] T* allocate(std::size_t n) {
    if (n == 0) {
      return nullptr;
    }

    // Overflow check: n * sizeof(T)
    if (n > (static_cast<std::size_t>(-1) / sizeof(T))) {
      throw std::bad_array_new_length();
    }

    const std::size_t bytes = n * sizeof(T);

#if defined(_MSC_VER)
    void* p = _aligned_malloc(bytes, Alignment);
    if (!p) {
      throw std::bad_alloc();
    }
    return static_cast<T*>(p);
#else
    void* p = nullptr;
    // posix_memalign requires alignment to be power-of-two multiple of sizeof(void*).
    const int rc = posix_memalign(&p, Alignment, bytes);
    if (rc != 0 || !p) {
      throw std::bad_alloc();
    }
    return static_cast<T*>(p);
#endif
  }

  void deallocate(T* p, std::size_t /*n*/) noexcept {
#if defined(_MSC_VER)
    _aligned_free(p);
#else
    std::free(p);
#endif
  }

  template <typename U>
  struct rebind {
    using other = AlignedAllocator<U, Alignment>;
  };

  // Allocators are stateless; all instances are equivalent.
  using is_always_equal = std::true_type;
};

template <typename T1, std::size_t A1, typename T2, std::size_t A2>
inline bool operator==(const AlignedAllocator<T1, A1>&, const AlignedAllocator<T2, A2>&) noexcept {
  return A1 == A2;
}

template <typename T1, std::size_t A1, typename T2, std::size_t A2>
inline bool operator!=(const AlignedAllocator<T1, A1>& a, const AlignedAllocator<T2, A2>& b) noexcept {
  return !(a == b);
}

} // namespace vectrax
