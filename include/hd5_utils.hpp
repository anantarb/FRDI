#include "H5Cpp.h"
#include "hdf5.h"

#include <Eigen/Dense>

using namespace H5;
using namespace std;
using Eigen::MatrixBase;

namespace hd5_utils {

template <typename _Tp, typename _Ep>
void convert_to_eigen(MatrixBase<_Tp> &m, unique_ptr<_Ep[]> &raw) {
  for (unsigned int i = 0; i < m.rows(); i++)
    for (unsigned int j = 0; j < m.cols(); j++)
      m(i, j) = raw[i * (unsigned int)m.cols() + j];
}

template <typename _Tp, typename _Ep>
void LoadH5Model(hid_t file, const std::string &strPath,
                 unique_ptr<_Tp[]> &raw_data, MatrixBase<_Ep> &eigen_data,
                 hid_t predType) {
  hid_t dataSet = H5Dopen(file, strPath.c_str(), H5P_DEFAULT);
  herr_t status =
      H5Dread(dataSet, predType, H5S_ALL, H5S_ALL, H5P_DEFAULT, raw_data.get());
  convert_to_eigen(eigen_data, raw_data);
}

} // namespace hd5_utils
