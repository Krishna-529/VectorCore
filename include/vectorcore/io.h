#pragma once

#include <cstddef>
#include <cstdint>
#include <istream>
#include <ostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

// Minimal binary (de)serialization helpers for index persistence.
//
// Format is little-endian, host-native — files are portable across runs on the
// same architecture (x86-64). Each index writes a 4-byte magic + uint32 version
// so a wrong/corrupt file fails loudly instead of silently mis-parsing.
namespace vectorcore::io {

inline void write_bytes(std::ostream& os, const void* p, std::size_t n) {
  os.write(reinterpret_cast<const char*>(p), static_cast<std::streamsize>(n));
  if (!os) {
    throw std::runtime_error("vectorcore: write failed");
  }
}

inline void read_bytes(std::istream& is, void* p, std::size_t n) {
  if (!is.read(reinterpret_cast<char*>(p), static_cast<std::streamsize>(n))) {
    throw std::runtime_error("vectorcore: unexpected end of file while reading");
  }
}

template <class T>
void write_pod(std::ostream& os, const T& v) {
  static_assert(std::is_trivially_copyable<T>::value, "write_pod requires a POD type");
  write_bytes(os, &v, sizeof(T));
}

template <class T>
T read_pod(std::istream& is) {
  static_assert(std::is_trivially_copyable<T>::value, "read_pod requires a POD type");
  T v;
  read_bytes(is, &v, sizeof(T));
  return v;
}

// Flat vector of POD elements: [uint64 count][count * element bytes].
// Works for std::vector with any allocator (e.g. AlignedAllocator).
template <class Vec>
void write_pod_vector(std::ostream& os, const Vec& v) {
  using T = typename Vec::value_type;
  static_assert(std::is_trivially_copyable<T>::value, "element must be POD");
  const std::uint64_t count = v.size();
  write_pod(os, count);
  if (count) {
    write_bytes(os, v.data(), static_cast<std::size_t>(count) * sizeof(T));
  }
}

template <class Vec>
void read_pod_vector(std::istream& is, Vec& v) {
  using T = typename Vec::value_type;
  static_assert(std::is_trivially_copyable<T>::value, "element must be POD");
  const std::uint64_t count = read_pod<std::uint64_t>(is);
  v.resize(static_cast<std::size_t>(count));
  if (count) {
    read_bytes(is, v.data(), static_cast<std::size_t>(count) * sizeof(T));
  }
}

// 4-byte magic tag helpers.
inline void write_magic(std::ostream& os, const char (&magic)[5]) {
  write_bytes(os, magic, 4);
}

inline void expect_magic(std::istream& is, const char (&magic)[5]) {
  char buf[4];
  read_bytes(is, buf, 4);
  for (int i = 0; i < 4; ++i) {
    if (buf[i] != magic[i]) {
      throw std::runtime_error("vectorcore: bad file magic (wrong index type or corrupt file)");
    }
  }
}

}  // namespace vectorcore::io
