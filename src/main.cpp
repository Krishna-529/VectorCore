#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstddef>
#include <stdexcept>
#include <string>

#include "VectorStore.hpp"

namespace py = pybind11;

namespace {

// Phase 4: Zero-copy NumPy -> C++
// ----------------------------
// We use the Python Buffer Protocol via `py::buffer_info`.
//
// Why this matters:
// - When you pass a NumPy array into C++, you *can* avoid memcpy by directly
//   reading the array's backing memory.
// - This is critical for HPC-style throughput: copying 768 floats per query is
//   small, but at high QPS it becomes measurable overhead.
//
// We validate the buffer to keep the C++ core simple and fast:
// - 1D only (buf.ndim == 1)
// - float32 only (format check)
// - contiguous (stride == sizeof(float))

inline const float* require_1d_float32_contiguous(const py::array& arr, std::size_t expected_dim) {
  py::buffer_info buf = arr.request();

  if (buf.ndim != 1) {
    throw std::invalid_argument("Expected a 1D NumPy array of shape (dim,)");
  }

  if (static_cast<std::size_t>(buf.shape[0]) != expected_dim) {
    throw std::invalid_argument("dim mismatch: NumPy array length must equal store dim");
  }

  // Crucial: itemsize alone isn't enough (int32 is also 4 bytes).
  if (buf.itemsize != sizeof(float) || buf.format != py::format_descriptor<float>::format()) {
    throw std::invalid_argument("Expected dtype float32");
  }

  if (buf.strides[0] != static_cast<py::ssize_t>(sizeof(float))) {
    throw std::invalid_argument("Expected contiguous float32 array (no slicing)");
  }

  // Requirement reference:
  //   float* ptr = static_cast<float*>(buf.ptr);
  // We immediately treat it as const because we only *read* from NumPy.
  float* ptr = static_cast<float*>(buf.ptr);
  return ptr;
}

} // namespace

PYBIND11_MODULE(vectorcore, m) {
  py::print("VectorCore Online");
  m.doc() = "VectorCore: zero-copy VectorStore bindings (pybind11)";

  m.def("ping", []() { return "VectorCore Online"; });

  py::class_<vectorcore::VectorStore>(m, "VectorStore")
      .def(py::init<std::size_t>(), py::arg("dim"))
      .def_property_readonly("dim", &vectorcore::VectorStore::dim)
      .def_property_readonly("size", &vectorcore::VectorStore::size)

      // store.add_vector(id, np.ndarray[float32, (dim,)])
      .def(
          "add_vector",
          [](vectorcore::VectorStore& self, int id, const py::array& vec) {
            const float* ptr = require_1d_float32_contiguous(vec, self.dim());
            self.add_vector(id, ptr, self.dim());
          },
          py::arg("id"),
          py::arg("vec"),
          "Add a single vector (zero-copy read from NumPy buffer).")

      // store.search(np.ndarray[float32, (dim,)], k) -> List[Tuple[distance, id]]
      .def(
          "search",
          [](const vectorcore::VectorStore& self, const py::array& query, int k) {
            const float* ptr = require_1d_float32_contiguous(query, self.dim());
            return self.search(ptr, k);
          },
          py::arg("query"),
          py::arg("k"),
          "Brute-force kNN search (returns list of (distance, id)).");
}
