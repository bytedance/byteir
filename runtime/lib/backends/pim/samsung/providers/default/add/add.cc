#include "add.h"
#include "FP16.h"
#include "brt/backends/pim/samsung/device/BurstTensor.h"
// #pragma once

namespace brt {
namespace pim {
namespace hbmpim {

namespace kernel {

// declaration
template <typename T>
void add_kernel(HBMPIMKernel kernel, TDataDim *dim_data,
                DRAMSim::BurstType *result) {

  kernel.executeEltwise(dim_data->dimTobShape(dim_data->output_dim_),
                        pimBankType::ALL_BANK, KernelType::ADD, 0, 0, 0);

  kernel.readData(result, dim_data->dimTobShape(dim_data->output_dim_), 0, 0);
};
// initiate template
// template void add_kernel<int>(HBMPIMKernel *kernel, TDataDim
// dim_Data,DRAMSim::BurstType *C
//                              );
// template float add_kernel<float>(HBMPIMKernel**kernel, float *A,
// float *B,uint32_t output_dim);
// initi template float
template void add_kernel<float>(HBMPIMKernel kernel, TDataDim *dim_data,
                                DRAMSim::BurstType *result);
template void add_kernel<int>(HBMPIMKernel kernel, TDataDim *dim_data,
                              DRAMSim::BurstType *result);
template void add_kernel<half_float::half>(HBMPIMKernel kernel,
                                           TDataDim *dim_data,
                                           DRAMSim::BurstType *result);

} // namespace kernel

// namespace kernel
} // namespace hbmpim
} // namespace pim
} // namespace brt