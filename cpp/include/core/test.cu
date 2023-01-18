/*
 * How to compile (assume cuda is installed at /usr/local/cuda/)
 *   nvcc -c -I/usr/local/cuda/include csrms2.cpp
 *   g++ -o csrm2 csrsm2.o -L/usr/local/cuda/lib64 -lcusparse -lcudart
 */
#include <assert.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <stdio.h>
#include <stdlib.h>

#include <iostream>

using namespace std;

__global__ void update(int *d_str) { d_str[0] += 1; }

cusparseStatus_t create_sparse(cusparseSpMatDescr_t *spMatDescr,
                               unsigned int *indptr, unsigned int *indices,
                               float *data) {
  int a = 0;
  cudaMemcpy(&a, indptr + 2, sizeof(int), cudaMemcpyDeviceToHost);
  std::cout << a << std::endl;
  return cusparseCreateCsr(spMatDescr, 4, 4, 9, indptr, indices, data,
                           CUSPARSE_INDEX_16U, CUSPARSE_INDEX_16U,
                           CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
}

void print_any_thing(int str) {
  cusparseSpMatDescr_t descrA;
  unsigned int csrRowPtrA[] = {1, 4, 5, 8, 10};
  unsigned int csrColIndA[] = {1, 3, 4, 2, 1, 3, 4, 2, 4};
  float csrValA[] = {1, 2, -3, 4, 5, 6, 7, 8, 9};
  unsigned int *d_csrRowPtrA = nullptr;
  unsigned int *d_csrColIndA = nullptr;
  float *d_csrValA = nullptr;

  cudaMalloc(&d_csrRowPtrA, sizeof(int) * 5000);
  cudaMalloc(&d_csrColIndA, sizeof(int) * 9000);
  cudaMalloc(&d_csrValA, sizeof(float) * 9000);
  cudaMemcpy(d_csrRowPtrA, csrRowPtrA, sizeof(int) * 5, cudaMemcpyHostToDevice);
  cudaMemcpy(d_csrColIndA, csrColIndA, sizeof(int) * 9, cudaMemcpyHostToDevice);
  cudaMemcpy(d_csrValA, csrValA, sizeof(float) * 9, cudaMemcpyHostToDevice);

  auto stt = create_sparse(&descrA, d_csrRowPtrA, d_csrColIndA, d_csrValA);

  std::cout << (stt == CUSPARSE_STATUS_SUCCESS) << std::endl;

  int *d_str;
  // int32_t
  // Allocate device memory for a
  // cudaMalloc((void **)&d_str, sizeof(int) * 1);

  // Transfer data from host to device memory
  // cudaMemcpy(d_str, &str, sizeof(int) * 1, cudaMemcpyHostToDevice);
  // update<<<1, 1>>>(d_str);
  // cudaMemcpy(&str, d_str, sizeof(int) * 1, cudaMemcpyDeviceToHost);
  // cudaFree(d_str);
  printf("%d\n", str);
};
