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

// auto oja_batch(SpMt matrix, Mt_d mean, int32_t k, int32_t batch_sz = 2048,
//                double_t step_sz = 1e-3, double_t epsilon = 1e-8,
//                double_t beta1 = 0.9, double_t beta2 = 0.999) -> Mt_d {
//   int32_t n_rows = matrix.rows(), n_cols = matrix.cols();

//   default_random_engine engine;
//   std::normal_distribution<double> distribution(0, 1);
//   auto gauss = [&](double) { return distribution(engine); };
//   Mt_d w = Mt_d::NullaryExpr(n_cols, k, gauss);

//   Eigen::HouseholderQR<Mt_d> qr(w);
//   w = qr.householderQ() * Mt_d::Identity(w.rows(), w.cols());

//   batch_sz = std::min(batch_sz, n_rows / 10);
//   Arr_d m(n_cols, k), v(1, k);
//   int32_t t = 0;
//   for (int32_t i = 0; i < 3; i++) {
//     t++;
//     Mt_d sX = matrix.middleRows(0, batch_sz).cast<double>();
//     Mt_d col_sums = Mt_d::Ones(1, sX.rows()) * sX;
//     Arr_d g =
//         (sX.transpose() * (sX * w) - mean * (col_sums * w) +
//          (mean * sX.rows() - col_sums.transpose()) * (mean.transpose() * w))
//             .array();
//     m = beta1 * m + (1 - beta1) * g;
//     v = beta2 * v + (1 - beta2) * g.colwise().squaredNorm();
//     Arr_d mh = m / (1 - std::pow(beta1, t));
//     Arr_d vh = v / (1 - std::pow(beta2, t));
//     Arr_d vh_sqrt = (vh.cwiseSqrt().array() + epsilon);
//     Mt_d mh_vh = (mh.array().rowwise() / vh_sqrt.row(0).array()).matrix();
//     step_sz = std::max(step_sz, 0.5 / (mh_vh.cwiseAbs()).mean());
//     Mt_d s = w - step_sz * mh_vh;
//     Eigen::HouseholderQR<Mt_d> qr(s);
//     w = qr.householderQ() * Mt_d::Identity(s.rows(), s.cols());
//   }

//   for (int32_t start = batch_sz; start < n_rows; start += batch_sz) {
//     t++;
//     int32_t n = std::min(batch_sz, n_rows - start);
//     Mt_d sX = matrix.middleRows(start, n).cast<double>();
//     Mt_d col_sums = Mt_d::Ones(1, sX.rows()) * sX;
//     Arr_d g = (sX.transpose() * (sX * w) - mean * (col_sums * w) +
//               (mean * sX.rows() - col_sums.transpose()) *
//                   (mean.transpose() * w))
//         .matrix();
//     m = beta1 * m + (1 - beta1) * g;
//     v = beta2 * v + (1 - beta2) * g.colwise().squaredNorm();
//     Arr_d mh = m / (1 - std::pow(beta1, t));
//     Arr_d vh = v / (1 - std::pow(beta2, t));
//     Arr_d vh_sqrt = ((vh.cwiseSqrt().array() + epsilon).matrix());
//     Mt_d mh_vh = (mh.array().rowwise() / vh_sqrt.row(0).array()).matrix();
//     Mt_d s = w - step_sz * mh_vh;
//     Eigen::HouseholderQR<Mt_d> qr(s);
//     w = qr.householderQ() * Mt_d::Identity(s.rows(), s.cols());
//     step_sz *= (double_t)t / (t + 1);
//   }

//   Mt_d output = matrix.cast<double>() * w;
//   return output;
// }

auto oja_batch_v1(SpMt matrix, Mt_d mean, int32_t k, int32_t batch_sz = 2048,
                  double_t step_sz = 1e-3, double_t epsilon = 1e-8,
                  double_t beta1 = 0.9, double_t beta2 = 0.999) -> Mt_d {
  Eigen::initParallel();
  Eigen::setNbThreads(16);

  int32_t n_rows = matrix.rows(), n_cols = matrix.cols();

  default_random_engine engine;
  std::normal_distribution<double> distribution(0, 1);
  auto gauss = [&](double) { return distribution(engine); };
  Mt_d w = Mt_d::NullaryExpr(n_cols, k, gauss);

  Eigen::HouseholderQR<Mt_d> qr(w);
  w = qr.householderQ() * Mt_d::Identity(w.rows(), w.cols());

  batch_sz = std::min(batch_sz, n_rows / 10);
  Mt_d m(n_cols, k), v(1, k);
  int32_t t = 0;
  for (int32_t i = 0; i < 3; i++) {
    t++;
    auto sX = matrix.middleRows(0, batch_sz).cast<double>();
    Mt_d col_sums = Mt_d::Ones(1, sX.rows()) * sX;
    Mt_d g =
        -(sX.transpose() * (sX * w) - mean * (col_sums * w) +
          (mean * sX.rows() - col_sums.transpose()) * (mean.transpose() * w));
    m = beta1 * m + (1 - beta1) * g;
    v = beta2 * v + (1 - beta2) * g.colwise().squaredNorm();
    Mt_d mh = m / (1 - std::pow(beta1, t));
    Mt_d vh = v / (1 - std::pow(beta2, t));
    Mt_d vh_sqrt = ((vh.cwiseSqrt().array() + epsilon).matrix());
    Mt_d mh_vh = (mh.array().rowwise() / vh_sqrt.row(0).array()).matrix();
    step_sz = std::max(step_sz, 0.5 / (mh_vh.cwiseAbs()).mean());
    Mt_d s = w - step_sz * mh_vh;
    Eigen::HouseholderQR<Mt_d> qr(s);
    w = qr.householderQ() * Mt_d::Identity(s.rows(), s.cols());
  }
  std::cout << "step size: " << step_sz << std::endl;
  for (int32_t start = batch_sz; start < n_rows; start += batch_sz) {
    t++;
    int32_t n = std::min(batch_sz, n_rows - start);
    auto sX = matrix.middleRows(start, n).cast<double>();
    Mt_d col_sums = Mt_d::Ones(1, sX.rows()) * sX;
    Mt_d g =
        -(sX.transpose() * (sX * w) - mean * (col_sums * w) +
          (mean * sX.rows() - col_sums.transpose()) * (mean.transpose() * w));
    m = beta1 * m + (1 - beta1) * g;
    v = beta2 * v + (1 - beta2) * g.colwise().squaredNorm();
    Mt_d mh = m / (1 - std::pow(beta1, t));
    Mt_d vh = v / (1 - std::pow(beta2, t));
    Mt_d vh_sqrt = ((vh.cwiseSqrt().array() + epsilon).matrix());
    Mt_d mh_vh = (mh.array().rowwise() / vh_sqrt.row(0).array()).matrix();
    Mt_d s = w - step_sz * mh_vh;
    Eigen::HouseholderQR<Mt_d> qr(s);
    w = qr.householderQ() * Mt_d::Identity(s.rows(), s.cols());
    step_sz *= (double_t)t / (t + 1);
  }

  Mt_d output = matrix.cast<double>() * w;
  return output;
}

}  // namespace pca
PYBIND11_MODULE(_oja, m) {
  m.def("_oja_batch", &pca::oja_batch_v1, py::return_value_policy::move);
  // m.def("check", &pca::check_matrix);
}