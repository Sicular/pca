#include <pybind11/pybind11.h>
#include <core/test.h>

PYBIND11_MODULE(_pca, m) {
  m.def("_history_pca", &print_any_thing);
  // m.def("check", &pca::check_matrix);
}