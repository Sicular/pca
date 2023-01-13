#include <stdlib.h>

#include <Eigen/Sparse>
#include <iostream>

using SpMt = Eigen::SparseMatrix<int32_t>;

using namespace std;

int main(int argc, char const *argv[]) {
  auto a = SpMt(2, 2);
  a.ind
  auto b = SpMt(2, 2);
  a.insert(1, 1) = 1;
  b.insert(1, 1) = 2;
  b.makeCompressed();
  a.makeCompressed();
  SpMt c = a * b;
  std::cout << a << std::endl;
  std::cout << a.isCompressed() << std::endl;
  std::cout << c << std::endl;
  std::cout << c.isCompressed() << std::endl;

  return 0;
}
