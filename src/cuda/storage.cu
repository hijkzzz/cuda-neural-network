#include <pybind11/pybind11.h>

#include <storage.cuh>
#include <utils.h>

// bind NumPy and Storage
py::class_<Matrix>(m, "Storage", py::buffer_protocol())
    .def_buffer([](Storage &m) -> py::buffer_info {
      return py::buffer_info(
          m.data(),                               /* Pointer to buffer */
          sizeof(float),                          /* Size of one scalar */
          py::format_descriptor<float>::format(), /* Python struct-style format
                                                     descriptor */
          2,                                      /* Number of dimensions */
          m.shape,                                /* Buffer dimensions */
          {sizeof(float) * m.cols(), /* Strides (in bytes) for each index */
           sizeof(float)});
    });

py::class_<Matrix>(m, "Storage", py::buffer_protocol())
    .def("__init__", [](Matrix &m, py::buffer b) {
      /* Request a buffer descriptor from Python */
      py::buffer_info info = b.request();

      /* Some sanity checks ... */
      if (info.format != py::format_descriptor<Scalar>::format())
        throw std::runtime_error(
            "Incompatible format: expected a double array!");

      if (info.ndim != 2)
        throw std::runtime_error("Incompatible buffer dimension!");

      auto strides =
          Strides(info.strides[rowMajor ? 0 : 1] / (py::ssize_t)sizeof(Scalar),
                  info.strides[rowMajor ? 1 : 0] / (py::ssize_t)sizeof(Scalar));

      auto map =
          Eigen::Map<Matrix, 0, Strides>(static_cast<Scalar *>(info.ptr),
                                         info.shape[0], info.shape[1], strides);

      new (&m) Matrix(map);
    });