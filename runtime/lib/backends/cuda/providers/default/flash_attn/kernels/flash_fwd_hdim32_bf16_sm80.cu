// Copyright (c) 2023, Tri Dao.

// Splitting the different head dimensions to different files to speed up
// compilation.

#include "flash_fwd_launch_template.h"
namespace brt {
namespace cuda {
namespace kernel {
template <>
void run_mha_fwd_<cutlass::bfloat16_t, 32>(Flash_fwd_params &params,
                                           cudaStream_t stream) {
  run_mha_fwd_hdim32<cutlass::bfloat16_t>(params, stream);
}
} // namespace kernel
} // namespace cuda
} // namespace brt