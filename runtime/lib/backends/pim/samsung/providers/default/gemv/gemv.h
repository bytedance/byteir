
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
void gemv_kernel(HBMPIMKernel kernel, TensorBurstType *w_data,
                 TensorBurstType *i_data, bool is_tree);

} // namespace kernel
} // namespace hbmpim
} // namespace pim
} // namespace brt