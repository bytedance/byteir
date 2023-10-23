


#include "brt/backends/pim/upmem/device/common.h"
namespace brt {
namespace pim {

namespace upmem {
namespace kernel {
void runadd(struct dpu_set_t* dpu_set, struct dpu_set_t dpu, uint32_t nr_of_dpus,
            void *A, void *B, void *C, int m, int n);
void runsub(struct dpu_set_t* dpu_set, struct dpu_set_t dpu, uint32_t nr_of_dpus,
            void *A, void *B, void *C, int m, int n);   
void runmul(struct dpu_set_t* dpu_set, struct dpu_set_t dpu, uint32_t nr_of_dpus,
            void *A, void *B, void *C, int m, int n);
void rundiv(struct dpu_set_t* dpu_set, struct dpu_set_t dpu, uint32_t nr_of_dpus,
            void *A, void *B, void *C, int m, int n);
} // namespace kernel
} // namespace upmem
} // namespace pim
} // namespace brt