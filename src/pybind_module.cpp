#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "vectorcore/bruteforce_index.h"
#include "vectorcore/hnsw_index.h"

namespace py = pybind11;

namespace {

// Validates and unwraps a NumPy array via the buffer protocol without copying.
//
// Interview talking points:
// - `py::buffer_info` gives you a raw pointer + shape/strides.
// - We explicitly reject non-contiguous arrays to keep the C++ inner loops fast
//   and to honor the "no memcpy bridge" constraint.
// - We validate dtype==float32 to avoid implicit conversion copies.

struct Float32MatrixView {
  const float* data = nullptr;
  std::size_t rows = 0;
  std::size_t cols = 0;
};

Float32MatrixView as_float32_matrix_view(const py::array& arr, std::size_t expected_cols) {
  py::buffer_info info = arr.request();

  if (info.ndim != 2) {
    throw std::invalid_argument("Expected a 2D NumPy array of shape (n, dim)");
  }
  if (static_cast<std::size_t>(info.shape[1]) != expected_cols) {
    throw std::invalid_argument("Second dimension (dim) mismatch");
  }
  // itemsize alone is not sufficient; e.g. int32 is also 4 bytes.
  if (info.itemsize != sizeof(float) || info.format != py::format_descriptor<float>::format()) {
    throw std::invalid_argument("Expected dtype float32");
  }

  // Ensure C-contiguous: row-major contiguous memory.
  // Strides in bytes.
  // Use pybind11's ssize_t (alias to Py_ssize_t). `ssize_t` is POSIX and is not
  // guaranteed to exist on Windows/MSVC.
  const py::ssize_t expected_stride1 = static_cast<py::ssize_t>(sizeof(float));
  const py::ssize_t expected_stride0 = static_cast<py::ssize_t>(expected_cols * sizeof(float));

  if (info.strides[1] != expected_stride1 || info.strides[0] != expected_stride0) {
    throw std::invalid_argument("Expected C-contiguous float32 array (no slicing/Fortran order)");
  }

  return Float32MatrixView{
      static_cast<const float*>(info.ptr),
      static_cast<std::size_t>(info.shape[0]),
      static_cast<std::size_t>(info.shape[1]),
  };
}

struct Float32VectorView {
  const float* data = nullptr;
  std::size_t dim = 0;
};

Float32VectorView as_float32_vector_view(const py::array& arr, std::size_t expected_dim) {
  py::buffer_info info = arr.request();

  if (info.ndim != 1) {
    throw std::invalid_argument("Expected a 1D NumPy array of shape (dim,)");
  }
  if (static_cast<std::size_t>(info.shape[0]) != expected_dim) {
    throw std::invalid_argument("dim mismatch");
  }
  if (info.itemsize != sizeof(float) || info.format != py::format_descriptor<float>::format()) {
    throw std::invalid_argument("Expected dtype float32");
  }
  if (info.strides[0] != static_cast<py::ssize_t>(sizeof(float))) {
    throw std::invalid_argument("Expected contiguous float32 vector");
  }

  return Float32VectorView{static_cast<const float*>(info.ptr), expected_dim};
}

vectorcore::Metric parse_metric(const std::string& m) {
  if (m == "l2" || m == "l2_squared") {
    return vectorcore::Metric::L2_SQUARED;
  }
  if (m == "ip" || m == "inner_product") {
    return vectorcore::Metric::INNER_PRODUCT;
  }
  throw std::invalid_argument("Unknown metric: " + m);
}

} // namespace

PYBIND11_MODULE(vectorcore, m) {
  m.doc() = "VectorCore: high-performance vector search engine (C++17 + pybind11)";
  m.attr("__version__") = vectorcore_VERSION;

  py::enum_<vectorcore::Metric>(m, "Metric")
      .value("L2_SQUARED", vectorcore::Metric::L2_SQUARED)
      .value("INNER_PRODUCT", vectorcore::Metric::INNER_PRODUCT);

  py::class_<vectorcore::BruteForceIndex>(m, "BruteForceIndex")
      .def(py::init([](std::size_t dim, const std::string& metric) {
             return vectorcore::BruteForceIndex(dim, parse_metric(metric));
           }),
           py::arg("dim"), py::arg("metric") = "l2")
      .def_property_readonly("dim", &vectorcore::BruteForceIndex::dim)
      .def_property_readonly("size", &vectorcore::BruteForceIndex::size)
      .def("add", [](vectorcore::BruteForceIndex& self, const py::array& x, py::object ids_obj) {
        auto view = as_float32_matrix_view(x, self.dim());

        const std::uint64_t* ids_ptr = nullptr;
        std::vector<std::uint64_t> ids_tmp;

        if (!ids_obj.is_none()) {
          py::array ids_arr = py::cast<py::array>(ids_obj);
          py::buffer_info ids_info = ids_arr.request();

          if (ids_info.ndim != 1) {
            throw std::invalid_argument("ids must be a 1D array");
          }
          if (static_cast<std::size_t>(ids_info.shape[0]) != view.rows) {
            throw std::invalid_argument("ids length must match x.shape[0]");
          }
          if (ids_info.itemsize != sizeof(std::uint64_t) ||
              ids_info.format != py::format_descriptor<std::uint64_t>::format()) {
            throw std::invalid_argument("ids must be uint64");
          }
          if (ids_info.strides[0] != static_cast<py::ssize_t>(sizeof(std::uint64_t))) {
            throw std::invalid_argument("ids must be contiguous");
          }

          ids_ptr = static_cast<const std::uint64_t*>(ids_info.ptr);
        }

        self.add(view.data, view.rows, ids_ptr);
      }, py::arg("x"), py::arg("ids") = py::none())
      .def("search", [](const vectorcore::BruteForceIndex& self, const py::array& q, std::size_t k) {
        // Support q shape (dim,) or (m, dim)
        py::buffer_info info = q.request();

        if (info.itemsize != sizeof(float) || info.format != py::format_descriptor<float>::format()) {
          throw std::invalid_argument("Expected float32 queries");
        }

        if (info.ndim == 1) {
          auto v = as_float32_vector_view(q, self.dim());
          py::array_t<std::uint64_t> out_ids(k);
          py::array_t<float> out_scores(k);

          self.search(v.data, k,
                      static_cast<std::uint64_t*>(out_ids.request().ptr),
                      static_cast<float*>(out_scores.request().ptr));
          return py::make_tuple(out_ids, out_scores);
        }

        if (info.ndim == 2) {
          auto mat = as_float32_matrix_view(q, self.dim());
          const std::size_t m_queries = mat.rows;

          py::array_t<std::uint64_t> out_ids({m_queries, k});
          py::array_t<float> out_scores({m_queries, k});

          auto ids_buf = out_ids.request();
          auto sc_buf = out_scores.request();

          auto* ids_ptr = static_cast<std::uint64_t*>(ids_buf.ptr);
          auto* sc_ptr = static_cast<float*>(sc_buf.ptr);

          for (std::size_t i = 0; i < m_queries; ++i) {
            const float* qi = mat.data + (i * self.dim());
            self.search(qi, k, ids_ptr + (i * k), sc_ptr + (i * k));
          }

          return py::make_tuple(out_ids, out_scores);
        }

        throw std::invalid_argument("q must be 1D (dim,) or 2D (m, dim)");
      }, py::arg("q"), py::arg("k"))
      ;

  py::class_<vectorcore::HnswIndex>(m, "HnswIndex")
      .def(py::init([](std::size_t dim, std::size_t M, const std::string& metric) {
             return vectorcore::HnswIndex(dim, M, parse_metric(metric));
           }),
           py::arg("dim"), py::arg("M") = 16, py::arg("metric") = "l2")
      .def_property_readonly("dim", &vectorcore::HnswIndex::dim)
      .def_property_readonly("size", &vectorcore::HnswIndex::size)
      .def("add", [](vectorcore::HnswIndex& self, const py::array& x, py::object ids_obj) {
        auto view = as_float32_matrix_view(x, self.dim());

        const std::uint64_t* ids_ptr = nullptr;
        if (!ids_obj.is_none()) {
          py::array ids_arr = py::cast<py::array>(ids_obj);
          py::buffer_info ids_info = ids_arr.request();

          if (ids_info.ndim != 1) {
            throw std::invalid_argument("ids must be a 1D array");
          }
          if (static_cast<std::size_t>(ids_info.shape[0]) != view.rows) {
            throw std::invalid_argument("ids length must match x.shape[0]");
          }
          if (ids_info.itemsize != sizeof(std::uint64_t) ||
              ids_info.format != py::format_descriptor<std::uint64_t>::format()) {
            throw std::invalid_argument("ids must be uint64");
          }
          if (ids_info.strides[0] != static_cast<py::ssize_t>(sizeof(std::uint64_t))) {
            throw std::invalid_argument("ids must be contiguous");
          }
          ids_ptr = static_cast<const std::uint64_t*>(ids_info.ptr);
        }

        self.add(view.data, view.rows, ids_ptr);
      }, py::arg("x"), py::arg("ids") = py::none())
      .def("search", [](const vectorcore::HnswIndex& self, const py::array& q, std::size_t k) {
        auto v = as_float32_vector_view(q, self.dim());

        py::array_t<std::uint64_t> out_ids(k);
        py::array_t<float> out_scores(k);

        self.search(v.data, k,
                    static_cast<std::uint64_t*>(out_ids.request().ptr),
                    static_cast<float*>(out_scores.request().ptr));
        return py::make_tuple(out_ids, out_scores);
      }, py::arg("q"), py::arg("k"))
      ;
}
