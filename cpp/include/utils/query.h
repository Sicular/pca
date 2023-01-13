#ifndef COMPRESS_UTILS_QUERY_H_
#define COMPRESS_UTILS_QUERY_H_

#include <core/block_matrix.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <stdlib.h>

namespace py = pybind11;

namespace compress {
auto get_row(py::array_t<int32_t> output, int32_t row,
             block_size<int32_t> n_blocks, block_size<int32_t> z_block,
             block_size<int32_t> size) -> py::array_t<int32_t>;

auto get_col(py::array_t<int32_t> output, int32_t col,
             block_size<int32_t> n_blocks, block_size<int32_t> z_block,
             block_size<int32_t> size) -> py::array_t<int32_t>;

auto get_value(block_matrix sparse, py::array_t<int64_t> output, int32_t row,
               int32_t col, block_size<int32_t> n_blocks,
               block_size<int32_t> z_block, block_size<int32_t> size)
    -> py::array_t<int32_t>;
}  // namespace compress
#endif  // !COMPRESS_UTILS_QUERY_H_