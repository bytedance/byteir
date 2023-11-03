
#pragma once

// #include "Burst.h"
#include "brt/backends/pim/samsung/providers/default/PIMKernel.h"

namespace brt {
namespace pim {
namespace hbm {

namespace kernel {

// declaration
template <typename T>
void add_kernel(   shared_ptr<PIMKernel> *kernel, T *A, T *B, uint32_t output_dim
                );

} // namespace kernel
} // namespace hbm
} // namespace pim
} // namespace brt