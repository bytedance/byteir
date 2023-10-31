#include "add.h"
// #include ""

// #pragma once

namespace brt {
namespace pim {
namespace hbm {

namespace kernel {

// declaration
template <typename T>
void add_kernel(PIMKernel *kernel, T *A, T *B,uint32_t output_dim) {
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

} // namespace kernel
} // namespace hbm
} // namespace pim
} // namespace brt