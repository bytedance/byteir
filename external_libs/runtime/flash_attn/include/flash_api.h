#include "flash.h"
#include <cuda_runtime.h>

#if defined(_WIN32)

#ifndef EXPORT_API
#define EXPORT_API __declspec(dllexport)
#else
#define EXPORT_API __attribute__((visibility("default")))
#endif

void print_Qkv_params(Qkv_params &params);
void print_Flash_fwd_params(Flash_fwd_params &params);
void print_Flash_bwd_params(Flash_bwd_params &params);
void run_mha(void *q_ptr, void *k_ptr, void *v_ptr, void *o_ptr,
             void *softmax_lse_ptr, void *softmax_ptr, void *rng_state_ptr,

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

             float p_dropout, int window_size_left, int window_size_right,
             cudaStream_t stream);

void run_mha_bwd(void *q_ptr, void *k_ptr, void *v_ptr, void *o_ptr,
                 void *dout_ptr, void *dq_ptr, void *dk_ptr, void *dv_ptr,
                 void *dq_accum_ptr, void *softmax_lse_ptr,
                 void *dsoftmax_sum_ptr, void *rng_state_ptr,

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

                 float p_dropout, int window_size_left, int window_size_right,
                 cudaStream_t stream);

void run_mha_fwd_with_kvcache(
    void *q_ptr, void *k_ptr, void *v_ptr, void *knew_ptr, void *vnew_ptr,
    void *seqlens_k_, void *o_ptr, void *softmax_lse_ptr,

    uint32_t q_batch_stride, uint32_t k_batch_stride, uint32_t v_batch_stride,
    uint32_t knew_batch_stride, uint32_t vnew_batch_stride,
    uint32_t o_batch_stride,

    uint32_t q_row_stride, uint32_t k_row_stride, uint32_t v_row_stride,
    uint32_t knew_row_stride, uint32_t vnew_row_stride, uint32_t o_row_stride,

    uint32_t q_head_stride, uint32_t k_head_stride, uint32_t v_head_stride,
    uint32_t knew_head_stride, uint32_t vnew_head_stride,
    uint32_t o_head_stride,

    uint32_t b, uint32_t h, uint32_t h_k, uint32_t d, uint32_t d_rounded,
    uint32_t seqlen_knew, float softmax_scale,

    uint32_t seqlen_q, uint32_t seqlen_k, uint32_t seqlen_q_rounded,
    uint32_t seqlen_k_rounded,

    int window_size_left, int window_size_right, cudaStream_t stream);

#ifdef __cplusplus
extern "C" {
#endif
EXPORT_API void run_flash_attn_fwd(void **tensors, void *extra_args,
                                   cudaStream_t stream);

EXPORT_API void run_flash_attn_bwd(void **tensors, void *extra_args,
                                   cudaStream_t stream);

EXPORT_API void run_flash_attn_kvcache(void **tensors, void *extra_args,
                                       cudaStream_t stream);
#ifdef __cplusplus
}
#endif
