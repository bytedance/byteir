#include "add.h"
// #include ""

// #pragma once

namespace brt {
namespace pim {
namespace hbm {

namespace kernel {

// declaration
template <typename T>
void add_kernel(shared_ptr<PIMKernel> *kernel, T *A, T *B,uint32_t output_dim) {
  int input_row0 = 0;
  int input_row1 = 128;
  int result_row = 256;


  // kernel->preloadNoReplacement(a, input_row0, 0);
  // kernel->preloadNoReplacement(b, input_row1, 0);
  // kernel->executeEltwise(a->dimTobShape(output_dim),
  //                        pimBankType::ALL_BANK, KernelType::ADD, input_row0,
  //                        result_row, input_row1);
  // result = new BurstType[output_dim];
  // kernel->readData(result, dim_data->dimTobShape(output_dim),
  //                  result_row, 0);
};
//initiate template
template void add_kernel<int>(shared_ptr<PIMKernel> *kernel, int *A, int *B,uint32_t output_dim);
// template float add_kernel<float>(shared_ptr<PIMKernel> *kernel, float *A, float *B,uint32_t output_dim);
//initi template float
template void add_kernel<float>(shared_ptr<PIMKernel> *kernel, float *A, float *B,uint32_t output_dim);

}



 // namespace kernel
} // namespace hbm
} // namespace pim
} // namespace brt