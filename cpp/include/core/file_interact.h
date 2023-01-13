#ifndef PCA_CORE_FILE_INTERACT_H_
#define PCA_CORE_FILE_INTERACT_H_

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <stdlib.h>

#include <fstream>

using namespace std;
namespace py = pybind11;

namespace pca {

template <typename T>
auto read_file(const char* file_name, int64_t start, int64_t end)
    -> py::array_t<T> {
  ifstream file(file_name, ios::in | ios::binary);
  if (file.good()) {
    int data_size = sizeof(T);
    py::array_t<T> output{{end - start}};
    py::buffer_info buf = output.request();
    T* ptr = static_cast<T*>(buf.ptr);
    file.seekg(start * data_size);
    file.read((char*)ptr, data_size * (end - start));
    file.close();

    return output;
  } else {
    throw std::runtime_error("File not found!");
  }
}

// const char* indptr_path =
//     "/data1/intern/nghiatnh/matrix_compression/data/output/large_size2/"
//     "indptr.bin";
// const char* indices_path =
//     "/data1/intern/nghiatnh/matrix_compression/data/output/large_size2/"
//     "index.bin";
// const char* data_path =
//     "/data1/intern/nghiatnh/matrix_compression/data/output/large_size2/"
//     "value.bin";
const char* indptr_path = "/data1/intern/csr_matrix/indptr.bin";
const char* indices_path = "/data1/intern/csr_matrix/indices.bin";
const char* data_path = "/data1/intern/csr_matrix/data.bin";
}  // namespace pca
#endif  // !COMPRESS_CORE_FILE_INTERACT_H_
