#include <Eigen/Dense>
#include <iostream>

using namespace std;
using namespace Eigen;

int main(int argc, char const *argv[]) {
  MatrixXf A(MatrixXf::Random(10000, 1)), thinQ(MatrixXf::Identity(5, 3)), Q;
  A.setRandom();
  HouseholderQR<MatrixXf> qr(A);
  Q = qr.householderQ();
//   thinQ = qr.householderQ() * thinQ;
  std::cout << "The complete unitary matrix Q is:\n" << Q << "\n\n";
  std::cout << "The thin matrix Q is:\n" << thinQ << "\n\n";

  return 0;
}
