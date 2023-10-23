

#pragma once

#include "./dpu.h"
#include "./dpu_call.h"
#include "brt/backends/pim/upmem/device/upmem_worker_queue.h"

#include "brt/core/common/status.h"
#include "brt/core/context/execution_context.h"
#include "brt/core/context/execution_frame.h"
#include <cstdint>
// #include <string>

// Structures used by both the host and the dpu to communicate information
typedef struct {
  uint32_t n_size;
  uint32_t n_size_pad;
  uint32_t nr_rows;
  uint32_t max_rows;
} gemv_dpu_arguments_t;

// Specific information for each DPU
struct gemv_dpu_info_t {
  uint32_t rows_per_dpu;
  uint32_t rows_per_dpu_pad;
  uint32_t prev_rows_dpu;
};

struct dpu_info_t {
  struct dpu_set_t dpu_set;
  struct dpu_set_t dpu;
  uint32_t nr_of_dpus;
};

namespace brt {
namespace pim {
namespace upmem {
inline uint32_t MakeDPUSet(UpmemEnv &env,char * x){

  uint32_t nr_of_dpus;
  // Allocate DPUs and load binary
  DPU_ASSERT(dpu_alloc(env.GetNumDpus(), NULL, env.GetDpuSet()));
  DPU_ASSERT(dpu_load(*env.GetDpuSet(), x, NULL));
  DPU_ASSERT(dpu_get_nr_dpus(*env.GetDpuSet(), &nr_of_dpus));

  
  return nr_of_dpus;
}

} // namespace upmem
} // namespace pim
} // namespace brt