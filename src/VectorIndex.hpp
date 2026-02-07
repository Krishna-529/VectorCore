#pragma once

#include <cstddef>

namespace vectrax {

// This header is a placeholder for the 1-day hack setup.
// In the full CMake-based prototype, the real implementation lives under
// include/vectrax/ (BruteForceIndex, HnswIndex, AVX2 distance kernels).
//
// Keeping it here matches the requested file structure:
//   src/main.cpp
//   src/VectorIndex.hpp
//
// For the interview build, you can move/merge the real index API into this
// header if you want a single-translation-unit demo.

struct VectorIndex {
  std::size_t dim = 0;
};

} // namespace vectrax
