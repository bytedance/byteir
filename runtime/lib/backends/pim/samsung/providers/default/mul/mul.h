
#pragma once

// #include "Burst.h"
#include "brt/backends/pim/samsung/device/BurstTensor.h"
#include "brt/backends/pim/samsung/providers/default/HBMPIMKernel.h"
// #include "tests/HBMPIMKernel.h"
using namespace DRAMSim;
// using namespace brt::common;
namespace brt {
namespace pim {
namespace hbmpim {

namespace kernel {

// declaration
template <typename T>
void mul_kernel(  HBMPIMKernel *kernel, DRAMSim::TDataDim* data_dim,  DRAMSim::BurstType *C  );


} // namespace kernel
} // namespace hbm
} // namespace pim
} // namespace brt