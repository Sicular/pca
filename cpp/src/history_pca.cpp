#include <core/file_interact.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <stdlib.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>
#include <random>
#include <tuple>

namespace py = pybind11;
typedef Eigen::SparseMatrix<int32_t, Eigen::RowMajor> SpMt;
typedef Eigen::SparseMatrix<double_t, Eigen::RowMajor> SpMt_d;
typedef Eigen::Matrix<double_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    Mt_d;
typedef Eigen::Matrix<int32_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    Mt_i;
typedef Eigen::Array<double_t, Eigen::Dynamic, Eigen::Dynamic> Arr_d;

using namespace std;
using namespace chrono;

namespace pca {
auto history_pca(SpMt matrix, Mt_d mean, int32_t k, int32_t batch_sz = 2048)
    -> Mt_d {
  int32_t n_rows = matrix.rows(), n_cols = matrix.cols();
  batch_sz = std::min(batch_sz, n_rows / 10);
  Mt_d s;

  default_random_engine engine;
  engine.seed(42);
  std::normal_distribution<double_t> distribution(0, 1);
  auto gauss = [&](double_t) { return distribution(engine); };

  Mt_d prev_w = Mt_d::NullaryExpr(n_cols, k, gauss);
  //   Mt_d prev_w = Mt_d::Identity(n_cols, k);
  Eigen::HouseholderQR<Mt_d> qr(prev_w);
  prev_w = qr.householderQ() * Mt_d::Identity(prev_w.rows(), prev_w.cols());

  for (int32_t i = 0; i < 3; ++i) {
    auto x = matrix.middleRows(0, batch_sz).cast<double_t>();
    Mt_d col_sums = Mt_d::Ones(1, x.rows()) * x;
    s = (x.transpose() * (x * prev_w) - mean * (col_sums * prev_w) +
         (mean * x.rows() - col_sums.transpose()) *
             (mean.transpose() * prev_w)) /
        batch_sz;
    Eigen::HouseholderQR<Mt_d> qr(s);
    prev_w = qr.householderQ() * Mt_d::Identity(s.rows(), s.cols());
  }

  Mt_d eigen_vals = (s.colwise().norm()).asDiagonal();

  for (int32_t start = 0; start < n_rows; start += batch_sz) {
    int32_t length = std::min(batch_sz, n_rows - start);
    int32_t end = start + length;

    auto x = matrix.middleRows(start, length).cast<double_t>();

    double_t alpha = std::pow((double_t)start / end, 2);
    Mt_d y = prev_w * eigen_vals;
    Mt_d col_sums = Mt_d::Ones(1, x.rows()) * x;

    s = alpha * (y * (prev_w.transpose() * prev_w)) +
        ((1 - alpha) / (end - start)) *
            (x.transpose() * (x * prev_w) - mean * (col_sums * prev_w) +
             (mean * x.rows() - col_sums.transpose()) *
                 (mean.transpose() * prev_w));

    Eigen::HouseholderQR<Mt_d> qr(s);
    prev_w = qr.householderQ() * Mt_d::Identity(s.rows(), s.cols());

    eigen_vals = (s.colwise().norm()).asDiagonal();
  }

  Mt_d output = (matrix.cast<double_t>() * prev_w).array().rowwise() -
                (mean.transpose() * prev_w).row(0).array();
  return output;
}

}  // namespace pca
PYBIND11_MODULE(_pca, m) {
  m.def("_history_pca", &pca::history_pca, py::return_value_policy::move);
  // m.def("check", &pca::check_matrix);
}