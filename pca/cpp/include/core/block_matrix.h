#ifndef PCA_CORE_BLOCK_MATRIX_H_
#define PCA_CORE_BLOCK_MATRIX_H_

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace pca {
template <typename T>
class csb_matrix {
 private:
  /* data */
 public:
  csb_matrix(){};
  csb_matrix(py::array_t<int32_t> pyarr){};
  csb_matrix(std::tuple<int32_t, int32_t, int32_t> block_data){};
  ~csb_matrix(){};
  auto row(size_t row) -> py::array_t<int32_t> &;
  auto col(size_t col) -> py::array_t<int32_t> &;
  auto dot(csb_matrix &other) -> csb_matrix;
  auto transpose() -> csb_matrix;
  auto sqrt(csb_matrix &x) -> csb_matrix &;
  auto q_decomposition(csb_matrix &&x) -> csb_matrix &;
  auto operator+(csb_matrix &x) -> csb_matrix;

  template <typename scalar_T>
  auto operator+(scalar_T &other) -> decltype(T + scalar_T);
  auto operator*(csb_matrix &other) -> csb_matrix;
  template <typename scalar_T>
  auto operator*(scalar_T &other) -> decltype(T * scalar_T);
  auto to_pyarray() -> py::array_t<int32_t>;
};

}  // namespace pca

template <typename T>
using block_size = std::tuple<T, T>;

#endif  // !PCA_CORE_BLOCK_MATRIX_H_