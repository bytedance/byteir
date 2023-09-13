#include "flash.h"
#include <cuda_runtime.h>

namespace brt {
namespace cuda {
namespace kernel {

void print_Qkv_params(Qkv_params &params);
void print_Flash_fwd_params(Flash_fwd_params &params);
void print_Flash_bwd_params(Flash_bwd_params &params);

void run_mha(void *q_ptr, void *k_ptr, void *v_ptr, void *o_ptr,
             void *softmax_lse_ptr, void *softmax_ptr, void *rng_state_ptr,

             int32_t *cu_seqlens_q_ptr, int32_t *cu_seqlens_k_ptr,

             uint32_t q_batch_stride, uint32_t k_batch_stride,
             uint32_t v_batch_stride, uint32_t o_batch_stride,

             uint32_t q_row_stride, uint32_t k_row_stride,
             uint32_t v_row_stride, uint32_t o_row_stride,

             uint32_t q_head_stride, uint32_t k_head_stride,
             uint32_t v_head_stride, uint32_t o_head_stride,

             uint32_t b, uint32_t h, uint32_t h_k, uint32_t d,
             uint32_t d_rounded, float softmax_scale,

             uint32_t seqlen_q, uint32_t seqlen_k, uint32_t seqlen_q_rounded,
             uint32_t seqlen_k_rounded,

             int is_causal, cudaStream_t stream);

void run_mha_bwd(void *q_ptr, void *k_ptr, void *v_ptr, void *o_ptr,
                 void *dout_ptr, void *dq_ptr, void *dk_ptr, void *dv_ptr,
                 int *cu_seqlens_q_ptr, int *cu_seqlens_k_ptr,
                 void *dq_accum_ptr, void *dk_accum_ptr, void *dv_accum_ptr,
                 void *softmax_lse_ptr, void *dsoftmax_sum_ptr,
                 void *rng_state_ptr,

                 uint32_t q_batch_stride, uint32_t k_batch_stride,
                 uint32_t v_batch_stride, uint32_t o_batch_stride,

                 uint32_t q_row_stride, uint32_t k_row_stride,
                 uint32_t v_row_stride, uint32_t o_row_stride,

                 uint32_t q_head_stride, uint32_t k_head_stride,
                 uint32_t v_head_stride, uint32_t o_head_stride,

                 uint32_t b, uint32_t h, uint32_t h_k, uint32_t d,
                 uint32_t d_rounded, float softmax_scale,

                 uint32_t seqlen_q, uint32_t seqlen_k,
                 uint32_t seqlen_q_rounded, uint32_t seqlen_k_rounded,

                 int is_causal, cudaStream_t stream);

} // namespace kernel
} // namespace cuda
} // namespace brt
