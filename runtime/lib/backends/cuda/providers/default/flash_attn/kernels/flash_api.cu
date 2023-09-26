#include "flash.h"
#include "flash_fwd_launch_template.h"
#include <iostream>

namespace brt {
namespace cuda {
namespace kernel {

// TODO: Switch back to handling bf16.
// void run_mha_fwd(Flash_fwd_params &params, cudaStream_t stream) {
//   FWD_HEADDIM_SWITCH(params.d, [&] {
//     run_mha_fwd_<cutlass::half_t, kHeadDim>(params, stream);
//   });
// }

// void run_mha_fwd(Flash_fwd_params &params, cudaStream_t stream) {
//     FP16_SWITCH(!params.is_bf16, [&] {
//         FWD_HEADDIM_SWITCH(params.d, [&] {
//             run_mha_fwd_<elem_type, kHeadDim>(params, stream);
//         });
//     });
// }

// for debug
void print_Qkv_params(Qkv_params &params) {
  std::cout << "q_batch_stride: " << params.q_batch_stride << std::endl;
  std::cout << "k_batch_stride: " << params.k_batch_stride << std::endl;
  std::cout << "v_batch_stride: " << params.v_batch_stride << std::endl;
  std::cout << "q_row_stride: " << params.q_row_stride << std::endl;
  std::cout << "k_row_stride: " << params.k_row_stride << std::endl;
  std::cout << "v_row_stride: " << params.v_row_stride << std::endl;
  std::cout << "q_head_stride: " << params.q_head_stride << std::endl;
  std::cout << "k_head_stride: " << params.k_head_stride << std::endl;
  std::cout << "v_head_stride: " << params.v_head_stride << std::endl;
  std::cout << "h: " << params.h << std::endl;
  std::cout << "h_k: " << params.h_k << std::endl;
  std::cout << "h_h_k_ratio: " << params.h_h_k_ratio << std::endl;
}

void print_Flash_fwd_params(Flash_fwd_params &params) {
  std::cout << "q_batch_stride: " << params.q_batch_stride << std::endl;
  std::cout << "k_batch_stride: " << params.k_batch_stride << std::endl;
  std::cout << "v_batch_stride: " << params.v_batch_stride << std::endl;
  std::cout << "q_row_stride: " << params.q_row_stride << std::endl;
  std::cout << "k_row_stride: " << params.k_row_stride << std::endl;
  std::cout << "v_row_stride: " << params.v_row_stride << std::endl;
  std::cout << "q_head_stride: " << params.q_head_stride << std::endl;
  std::cout << "k_head_stride: " << params.k_head_stride << std::endl;
  std::cout << "v_head_stride: " << params.v_head_stride << std::endl;
  std::cout << "h: " << params.h << std::endl;
  std::cout << "h_k: " << params.h_k << std::endl;
  std::cout << "h_h_k_ratio: " << params.h_h_k_ratio << std::endl;

  std::cout << "o_batch_stride: " << params.o_batch_stride << std::endl;
  std::cout << "o_row_stride: " << params.o_row_stride << std::endl;
  std::cout << "o_head_stride: " << params.o_head_stride << std::endl;
  std::cout << "b: " << params.b << std::endl;
  std::cout << "seqlen_q: " << params.seqlen_q << std::endl;
  std::cout << "seqlen_k: " << params.seqlen_k << std::endl;
  std::cout << "d: " << params.d << std::endl;
  std::cout << "seqlen_q_rounded: " << params.seqlen_q_rounded << std::endl;
  std::cout << "seqlen_k_rounded: " << params.seqlen_k_rounded << std::endl;
  std::cout << "d_rounded: " << params.d_rounded << std::endl;
  std::cout << "scale_softmax: " << params.scale_softmax << std::endl;
  std::cout << "scale_softmax_log2: " << params.scale_softmax_log2 << std::endl;
  std::cout << "p_dropout: " << params.p_dropout << std::endl;
  std::cout << "p_dropout_in_uint8_t: " << params.p_dropout_in_uint8_t
            << std::endl;
  std::cout << "rp_dropout: " << params.rp_dropout << std::endl;
  std::cout << "scale_softmax_rp_dropout: " << params.scale_softmax_rp_dropout
            << std::endl;
  std::cout << "is_bf16: " << params.is_bf16 << std::endl;
  std::cout << "is_causal: " << params.is_causal << std::endl;
}

void print_Flash_bwd_params(Flash_bwd_params &params) {
  std::cout << "q_batch_stride: " << params.q_batch_stride << std::endl;
  std::cout << "k_batch_stride: " << params.k_batch_stride << std::endl;
  std::cout << "v_batch_stride: " << params.v_batch_stride << std::endl;
  std::cout << "q_row_stride: " << params.q_row_stride << std::endl;
  std::cout << "k_row_stride: " << params.k_row_stride << std::endl;
  std::cout << "v_row_stride: " << params.v_row_stride << std::endl;
  std::cout << "q_head_stride: " << params.q_head_stride << std::endl;
  std::cout << "k_head_stride: " << params.k_head_stride << std::endl;
  std::cout << "v_head_stride: " << params.v_head_stride << std::endl;
  std::cout << "h: " << params.h << std::endl;
  std::cout << "h_k: " << params.h_k << std::endl;
  std::cout << "h_h_k_ratio: " << params.h_h_k_ratio << std::endl;

  std::cout << "o_batch_stride: " << params.o_batch_stride << std::endl;
  std::cout << "o_row_stride: " << params.o_row_stride << std::endl;
  std::cout << "o_head_stride: " << params.o_head_stride << std::endl;
  std::cout << "b: " << params.b << std::endl;
  std::cout << "seqlen_q: " << params.seqlen_q << std::endl;
  std::cout << "seqlen_k: " << params.seqlen_k << std::endl;
  std::cout << "d: " << params.d << std::endl;
  std::cout << "seqlen_q_rounded: " << params.seqlen_q_rounded << std::endl;
  std::cout << "seqlen_k_rounded: " << params.seqlen_k_rounded << std::endl;
  std::cout << "d_rounded: " << params.d_rounded << std::endl;
  std::cout << "scale_softmax: " << params.scale_softmax << std::endl;
  std::cout << "scale_softmax_log2: " << params.scale_softmax_log2 << std::endl;
  std::cout << "p_dropout: " << params.p_dropout << std::endl;
  std::cout << "p_dropout_in_uint8_t: " << params.p_dropout_in_uint8_t
            << std::endl;
  std::cout << "rp_dropout: " << params.rp_dropout << std::endl;
  std::cout << "scale_softmax_rp_dropout: " << params.scale_softmax_rp_dropout
            << std::endl;
  std::cout << "is_bf16: " << params.is_bf16 << std::endl;
  std::cout << "is_causal: " << params.is_causal << std::endl;

  std::cout << "do_batch_stride: " << params.do_batch_stride << std::endl;
  std::cout << "do_row_stride: " << params.do_row_stride << std::endl;
  std::cout << "do_head_stride: " << params.do_head_stride << std::endl;
  std::cout << "dq_batch_stride: " << params.dq_batch_stride << std::endl;
  std::cout << "dk_batch_stride: " << params.dk_batch_stride << std::endl;
  std::cout << "dv_batch_stride: " << params.dv_batch_stride << std::endl;
  std::cout << "dq_row_stride: " << params.dq_row_stride << std::endl;
  std::cout << "dk_row_stride: " << params.dk_row_stride << std::endl;
  std::cout << "dv_row_stride: " << params.dv_row_stride << std::endl;
  std::cout << "dq_head_stride: " << params.dq_head_stride << std::endl;
  std::cout << "dk_head_stride: " << params.dk_head_stride << std::endl;
  std::cout << "dv_head_stride: " << params.dv_head_stride << std::endl;
}

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

             float p_dropout, int is_causal, cudaStream_t stream) {
  Flash_fwd_params params;
  // Reset the parameters
  memset(&params, 0, sizeof(params));

  // Set the pointers and strides.
  params.q_ptr = q_ptr;
  params.k_ptr = k_ptr;
  params.v_ptr = v_ptr;
  params.o_ptr = o_ptr;

  params.softmax_lse_ptr = softmax_lse_ptr;

  // All stride are in elements, not bytes.
  params.q_batch_stride = q_batch_stride;
  params.k_batch_stride = k_batch_stride;
  params.v_batch_stride = v_batch_stride;
  params.o_batch_stride = o_batch_stride;

  params.q_row_stride = q_row_stride;
  params.k_row_stride = k_row_stride;
  params.v_row_stride = v_row_stride;
  params.o_row_stride = o_row_stride;
  params.q_head_stride = q_head_stride;
  params.k_head_stride = k_head_stride;
  params.v_head_stride = v_head_stride;
  params.o_head_stride = o_head_stride;

  // Set the dimensions.
  params.b = b;
  params.h = h;
  params.h_k = h_k;
  params.h_h_k_ratio = h / h_k;
  params.seqlen_q = seqlen_q;
  params.seqlen_k = seqlen_k;
  params.seqlen_q_rounded = seqlen_q_rounded;
  params.seqlen_k_rounded = seqlen_k_rounded;
  params.d = d;
  params.d_rounded = d_rounded;
  params.is_causal = is_causal;

  // Set the different scale values.
  params.scale_softmax = softmax_scale;
  params.scale_softmax_log2 = softmax_scale * M_LOG2E;

  params.p_dropout = 1.f - p_dropout; // probability to keep
  params.p_dropout_in_uint8_t = uint8_t(std::floor(params.p_dropout * 255.0));
  params.rp_dropout = 1.f / params.p_dropout;
  params.scale_softmax_rp_dropout = params.rp_dropout * params.scale_softmax;
  params.is_bf16 = 0;
  params.cu_seqlens_q = cu_seqlens_q_ptr;
  params.cu_seqlens_k = cu_seqlens_k_ptr;
  params.p_ptr = softmax_ptr; // used for `return_softmax`.
  params.rng_state = static_cast<uint64_t *>(rng_state_ptr);

  // print_Flash_fwd_params(params);

  FP16_SWITCH(!params.is_bf16, [&] {
    FWD_HEADDIM_SWITCH(
        params.d, [&] { run_mha_fwd_<elem_type, kHeadDim>(params, stream); });
  });
}

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

                 float p_dropout, int is_causal, cudaStream_t stream) {
  Flash_bwd_params params;
  // Reset the parameters
  memset(&params, 0, sizeof(params));

  // Set the pointers and strides.
  params.q_ptr = q_ptr;
  params.k_ptr = k_ptr;
  params.v_ptr = v_ptr;
  params.o_ptr = o_ptr;

  params.dq_ptr = dq_ptr;
  params.dk_ptr = dk_ptr;
  params.dv_ptr = dv_ptr;
  params.do_ptr = dout_ptr;

  params.dq_accum_ptr = dq_accum_ptr;
  params.dk_accum_ptr = dk_accum_ptr;
  params.dv_accum_ptr = dv_accum_ptr;

  params.softmax_lse_ptr = softmax_lse_ptr;

  // All stride are in elements, not bytes.
  params.q_batch_stride = q_batch_stride;
  params.k_batch_stride = k_batch_stride;
  params.v_batch_stride = v_batch_stride;
  params.o_batch_stride = o_batch_stride;

  params.q_row_stride = q_row_stride;
  params.k_row_stride = k_row_stride;
  params.v_row_stride = v_row_stride;
  params.o_row_stride = o_row_stride;
  params.q_head_stride = q_head_stride;
  params.k_head_stride = k_head_stride;
  params.v_head_stride = v_head_stride;
  params.o_head_stride = o_head_stride;

  params.dq_batch_stride = q_batch_stride;
  params.dk_batch_stride = k_batch_stride;
  params.dv_batch_stride = v_batch_stride;
  params.do_batch_stride = o_batch_stride;

  params.dq_row_stride = q_row_stride;
  params.dk_row_stride = k_row_stride;
  params.dv_row_stride = v_row_stride;
  params.do_row_stride = o_row_stride;
  params.dq_head_stride = q_head_stride;
  params.dk_head_stride = k_head_stride;
  params.dv_head_stride = v_head_stride;
  params.do_head_stride = o_head_stride;

  // Set the dimensions.
  params.b = b;
  params.h = h;
  params.h_k = h_k;
  params.h_h_k_ratio = h / h_k;
  params.seqlen_q = seqlen_q;
  params.seqlen_k = seqlen_k;
  params.seqlen_q_rounded = seqlen_q_rounded;
  params.seqlen_k_rounded = seqlen_k_rounded;
  params.d = d;
  params.d_rounded = d_rounded;
  params.is_causal = is_causal;

  // Set the different scale values.
  params.scale_softmax = softmax_scale;
  params.scale_softmax_log2 = softmax_scale * M_LOG2E;

  params.p_dropout = 1.f - p_dropout; // probability to keep
  params.p_dropout_in_uint8_t = uint8_t(std::floor(params.p_dropout * 255.0));
  params.rp_dropout = 1.f / params.p_dropout;
  params.scale_softmax_rp_dropout = params.rp_dropout * params.scale_softmax;
  params.is_bf16 = 0;
  params.cu_seqlens_q = cu_seqlens_q_ptr;
  params.cu_seqlens_k = cu_seqlens_k_ptr;
  params.p_ptr = nullptr; // used for `return_softmax`, no use in bwd
  params.dsoftmax_sum = dsoftmax_sum_ptr;
  params.rng_state = static_cast<uint64_t *>(rng_state_ptr);

  // print_Flash_bwd_params(params);

  bool configure = false;
  FP16_SWITCH(!params.is_bf16, [&] {
    if (params.d <= 32) {
      run_mha_bwd_<elem_type, 32>(params, stream, configure);
    } else if (params.d <= 64) {
      run_mha_bwd_<elem_type, 64>(params, stream, configure);
    } else if (params.d <= 96) {
      run_mha_bwd_<elem_type, 96>(params, stream, configure);
    } else if (params.d <= 128) {
      run_mha_bwd_<elem_type, 128>(params, stream, configure);
    } else if (params.d <= 160) {
      run_mha_bwd_<elem_type, 160>(params, stream, configure);
    } else if (params.d <= 192) {
      run_mha_bwd_<elem_type, 192>(params, stream, configure);
    } else if (params.d <= 224) {
      run_mha_bwd_<elem_type, 224>(params, stream, configure);
    } else if (params.d <= 256) {
      run_mha_bwd_<elem_type, 256>(params, stream, configure);
    }
  });
}

} // namespace kernel
} // namespace cuda
} // namespace brt