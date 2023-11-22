#include "gemv.h"

#include "brt/backends/pim/samsung/device/BurstTensor.h"
// #pragma once

namespace brt {
namespace pim {
namespace hbmpim {

namespace kernel {

// declaration
template <typename T>
void gemv_kernel(HBMPIMKernel *kernel, TensorBurstType *w_data,
                 TensorBurstType *i_data, bool is_tree) {

  kernel->executeGemv(w_data, i_data, is_tree);
  //   kernel->executeGEMV(dim_data->dimTobShape(dim_data->output_dim_),
  //                          pimBankType::ALL_BANK, KernelType::ADD, 0, 0,
  //                          0);

  //   kernel->readData(result, dim_data->dimTobShape(dim_data->output_dim_),
  //    0, 0);
};
// initiate template
// template void add_kernel<int>(HBMPIMKernel *kernel, TDataDim
// dim_Data,DRAMSim::BurstType *C
//                              );
// template float add_kernel<float>(HBMPIMKernel**kernel, float *A,
// float *B,uint32_t output_dim);
// initi template float
template void gemv_kernel<float>(HBMPIMKernel *kernel, TensorBurstType *w_data,
                                 TensorBurstType *i_data, bool is_tree

);
// template void add_kernel<__half>(HBMPIMKernel *kernel, TDataDim
// dim_Data,DRAMSim::BurstType *C
//                                 );
// template void add_kernel<int>(HBMPIMKernel *kernel, TDataDim
// dim_Data,DRAMSim::BurstType *C
//                               );

} // namespace kernel

// namespace kernel
} // namespace pim
} // namespace pim
} // namespace brt