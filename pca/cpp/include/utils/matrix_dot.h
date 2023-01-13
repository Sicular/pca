#ifndef COMPRESS_UTILS_MATRIX_DOT_H_
#define COMPRESS_UTILS_MATRIX_DOT_H_

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <stdlib.h>

namespace py = pybind11;

namespace compress {
auto dot(py::array_t<int32_t, py::array::c_style> dense,
         std::tuple<py::array_t<int64_t>, py::array_t<uint16_t>,
                    py::array_t<int32_t>>
             sparse,
         py::array_t<int64_t> output, std::tuple<int32_t, int32_t> n_blocks,
         std::tuple<int32_t, int32_t> z_block,
         std::tuple<int32_t, int32_t> size) -> py::array_t<int64_t>;
}  // namespace compress
#endif  // !COMPRESS_UTILS_MATRIX_DOT_H_