#include "flash.h"
#include "flash_api.h"
#include "static_switch.h"
#include <iostream>
#include <algorithm>

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

// Find the number of splits that maximizes the occupancy. For example, if we
// have batch * n_heads = 48 and we have 108 SMs, having 2 splits (efficiency =
// 0.89) is better than having 3 splits (efficiency = 0.67). However, we also
// don't want too many splits as that would incur more HBM reads/writes. So we
// find the best efficiency, then find the smallest number of splits that gets
// 85% of the best efficiency.
inline int num_splits_heuristic(int batch_nheads_mblocks, int num_SMs,
                                int num_n_blocks, int max_splits) {
  // If we have enough to almost fill the SMs, then just use 1 split
  if (batch_nheads_mblocks >= 0.8f * num_SMs) {
    return 1;
  }
  max_splits = std::min({max_splits, num_SMs, num_n_blocks});
  float max_efficiency = 0.f;
  std::vector<float> efficiency;
  efficiency.reserve(max_splits);
  auto ceildiv = [](int a, int b) { return (a + b - 1) / b; };
  // Some splits are not eligible. For example, if we have 64 blocks and choose
  // 11 splits, we'll have 6 * 10 + 4 blocks. If we choose 12 splits, we'll have
  // 6 * 11 + (-2) blocks (i.e. it's 11 splits anyway). So we check if the
  // number of blocks per split is the same as the previous num_splits.
  auto is_split_eligible = [&ceildiv, &num_n_blocks](int num_splits) {
    return num_splits == 1 || ceildiv(num_n_blocks, num_splits) !=
                                  ceildiv(num_n_blocks, num_splits - 1);
  };
  for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
    if (!is_split_eligible(num_splits)) {
      efficiency.push_back(0.f);
    } else {
      float n_waves = float(batch_nheads_mblocks * num_splits) / num_SMs;
      float eff = n_waves / ceil(n_waves);
      // printf("num_splits = %d, eff = %f\n", num_splits, eff);
      if (eff > max_efficiency) {
        max_efficiency = eff;
      }
      efficiency.push_back(eff);
    }
  }
  for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
    if (!is_split_eligible(num_splits)) {
      continue;
    }
    if (efficiency[num_splits - 1] >= 0.85 * max_efficiency) {
      // printf("num_splits chosen = %d\n", num_splits);
      return num_splits;
    }
  }
  return 1;
}

void run_fwd(Flash_fwd_params params, cudaStream_t stream) {
  auto head_dim = params.d;

  if (head_dim <= 32) {
    run_mha_fwd_<cutlass::half_t, 32>(params, stream);
  } else if (head_dim <= 64) {
    run_mha_fwd_<cutlass::half_t, 64>(params, stream);
  } else if (head_dim <= 96) {
    run_mha_fwd_<cutlass::half_t, 96>(params, stream);
  } else  if (head_dim <= 128) {
    run_mha_fwd_<cutlass::half_t, 128>(params, stream);
  } else  if (head_dim <= 160) {
    run_mha_fwd_<cutlass::half_t, 160>(params, stream);
  } else  if (head_dim <= 192) {
    run_mha_fwd_<cutlass::half_t, 192>(params, stream);
  } else  if (head_dim <= 224) {
    run_mha_fwd_<cutlass::half_t, 224>(params, stream);
  } else {
    run_mha_fwd_<cutlass::half_t, 256>(params, stream);
  }
}

void run_fwd_kvcache(Flash_fwd_params params, cudaStream_t stream) {
  auto head_dim = params.d;

  if (head_dim <= 32) {
    run_mha_fwd_splitkv_dispatch<cutlass::half_t, 32>(params, stream);
  } else if (head_dim <= 64) {
    run_mha_fwd_splitkv_dispatch<cutlass::half_t, 64>(params, stream);
  } else if (head_dim <= 96) {
    run_mha_fwd_splitkv_dispatch<cutlass::half_t, 96>(params, stream);
  } else  if (head_dim <= 128) {
    run_mha_fwd_splitkv_dispatch<cutlass::half_t, 128>(params, stream);
  } else  if (head_dim <= 160) {
    run_mha_fwd_splitkv_dispatch<cutlass::half_t, 160>(params, stream);
  } else  if (head_dim <= 192) {
    run_mha_fwd_splitkv_dispatch<cutlass::half_t, 192>(params, stream);
  } else  if (head_dim <= 224) {
    run_mha_fwd_splitkv_dispatch<cutlass::half_t, 224>(params, stream);
  } else {
    run_mha_fwd_splitkv_dispatch<cutlass::half_t, 256>(params, stream);
  }
}

void run_bwd(Flash_bwd_params params, cudaStream_t stream) {
  auto head_dim = params.d;

  if (head_dim <= 32) {
    run_mha_bwd_<cutlass::half_t, 32>(params, stream);
  } else if (head_dim <= 64) {
    run_mha_bwd_<cutlass::half_t, 64>(params, stream);
  } else if (head_dim <= 96) {
    run_mha_bwd_<cutlass::half_t, 96>(params, stream);
  } else  if (head_dim <= 128) {
    run_mha_bwd_<cutlass::half_t, 128>(params, stream);
  } else  if (head_dim <= 160) {
    run_mha_bwd_<cutlass::half_t, 160>(params, stream);
  } else  if (head_dim <= 192) {
    run_mha_bwd_<cutlass::half_t, 192>(params, stream);
  } else  if (head_dim <= 224) {
    run_mha_bwd_<cutlass::half_t, 224>(params, stream);
  } else {
    run_mha_bwd_<cutlass::half_t, 256>(params, stream);
  }
}

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
             cudaStream_t stream) {
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

  // Set the different scale values.
  params.scale_softmax = softmax_scale;
  params.scale_softmax_log2 = softmax_scale * M_LOG2E;

  params.p_dropout = 1.f - p_dropout; // probability to keep
  params.p_dropout_in_uint8_t = uint8_t(std::floor(params.p_dropout * 255.0));
  params.rp_dropout = 1.f / params.p_dropout;
  params.scale_softmax_rp_dropout = params.rp_dropout * params.scale_softmax;
  params.is_bf16 = 0;
  params.cu_seqlens_q = nullptr;
  params.cu_seqlens_k = nullptr;
  params.seqused_k = nullptr;
  params.p_ptr = softmax_ptr; // used for `return_softmax`.
  params.rng_state = static_cast<uint64_t *>(rng_state_ptr);
  params.is_causal = window_size_left < 0 && window_size_right == 0;

  if (window_size_left < 0 && window_size_right >= 0) {
    window_size_left = seqlen_k;
  }
  if (window_size_left >= 0 && window_size_right < 0) {
    window_size_right = seqlen_k;
  }
  params.window_size_left = window_size_left;
  params.window_size_right = window_size_right;
  params.is_seqlens_k_cumulative = true;
  params.alibi_slopes_ptr = nullptr;
  // print_Flash_fwd_params(params);

  run_fwd(params, stream);
}

void run_mha_bwd(void *q_ptr, void *k_ptr, void *v_ptr, void *o_ptr,
                 void *dout_ptr, void *dq_ptr, void *dk_ptr, void *dv_ptr,
                 void *dq_accum_ptr,
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

                 float p_dropout, int window_size_left, int window_size_right,
                 cudaStream_t stream) {
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
  params.dk_accum_ptr = nullptr;
  params.dv_accum_ptr = nullptr;

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
  params.is_causal = window_size_left < 0 && window_size_right == 0;
  if (window_size_left < 0 && window_size_right >= 0) {
    window_size_left = seqlen_k;
  }
  if (window_size_left >= 0 && window_size_right < 0) {
    window_size_right = seqlen_k;
  }
  params.window_size_left = window_size_left;
  params.window_size_right = window_size_right;
  params.is_seqlens_k_cumulative = true;

  // Set the different scale values.
  params.scale_softmax = softmax_scale;
  params.scale_softmax_log2 = softmax_scale * M_LOG2E;

  params.p_dropout = 1.f - p_dropout; // probability to keep
  params.p_dropout_in_uint8_t = uint8_t(std::floor(params.p_dropout * 255.0));
  params.rp_dropout = 1.f / params.p_dropout;
  params.scale_softmax_rp_dropout = params.rp_dropout * params.scale_softmax;
  params.is_bf16 = 0;
  params.cu_seqlens_q = nullptr;
  params.cu_seqlens_k = nullptr;
  params.seqused_k = nullptr;
  params.p_ptr = nullptr; // used for `return_softmax`, no use in bwd
  params.dsoftmax_sum = dsoftmax_sum_ptr;
  params.rng_state = static_cast<uint64_t *>(rng_state_ptr);
  params.alibi_slopes_ptr = nullptr;
  // print_Flash_bwd_params(params);

  run_bwd(params, stream);
}

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

    int window_size_left, int window_size_right, cudaStream_t stream) {
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

  // Set the different scale values.
  params.scale_softmax = softmax_scale;
  params.scale_softmax_log2 = softmax_scale * M_LOG2E;

  params.p_dropout = 1.f; // probability to keep
  params.p_dropout_in_uint8_t = uint8_t(std::floor(params.p_dropout * 255.0));
  params.rp_dropout = 1.f / params.p_dropout;
  params.scale_softmax_rp_dropout = params.rp_dropout * params.scale_softmax;

  params.is_bf16 = 0;
  params.cu_seqlens_q = nullptr;
  params.cu_seqlens_k = static_cast<int *>(seqlens_k_);
  params.seqused_k = nullptr;
  params.p_ptr = nullptr; // used for `return_softmax`.
  params.rng_state = nullptr;
  params.alibi_slopes_ptr = nullptr;
  params.page_block_size = 1;
  params.is_causal = window_size_left < 0 && window_size_right == 0;

  if (window_size_left < 0 && window_size_right >= 0) {
    window_size_left = seqlen_k;
  }
  if (window_size_left >= 0 && window_size_right < 0) {
    window_size_right = seqlen_k;
  }
  params.window_size_left = window_size_left;
  params.window_size_right = window_size_right;
  params.is_seqlens_k_cumulative = false;

  params.seqlen_knew = seqlen_knew;
  params.knew_ptr = knew_ptr;
  params.vnew_ptr = vnew_ptr;
  // All stride are in elements, not bytes.
  params.knew_batch_stride = knew_batch_stride;
  params.vnew_batch_stride = vnew_batch_stride;
  params.knew_row_stride = knew_row_stride;
  params.vnew_row_stride = vnew_row_stride;
  params.knew_head_stride = knew_head_stride;
  params.vnew_head_stride = vnew_head_stride;

  // TODO: ROPE support TBD
  params.rotary_dim = 0;

  // This needs to match with run_mha_fwd_splitkv_dispatch
  // const int head_size = round_multiple(head_size_og, 8);
  const int block_n = h <= 64 ? 256 : (h <= 128 ? 128 : 64);
  const int num_n_blocks = (seqlen_k + block_n - 1) / block_n;
  // Technically kBlockM = 64 only for the splitKV kernels, not the standard
  // kernel. In any case we don't expect seqlen_q to be larger than 64 for
  // inference.
  const int num_m_blocks = (seqlen_q + 64 - 1) / 64;
  // cudaDeviceProp dprops;
  // cudaGetDeviceProperties(&dprops, 0);
  // params.num_splits = num_splits_heuristic(
  //     b * h_k * num_m_blocks, dprops->multiProcessorCount, num_n_blocks,
  //     128);
  // static_assert(params.num_splits <= 128 && "num_splits > 128 not
  // supported");
  params.num_splits = 1;
  // TODO: support > 1 split
  // if (params.num_splits > 1) {
  //   at::Tensor softmax_lse_accum =
  //       torch::empty({params.num_splits, batch_size, num_heads, seqlen_q},
  //                    opts.dtype(at::kFloat));
  //   at::Tensor out_accum = torch::empty(
  //       {params.num_splits, batch_size, num_heads, seqlen_q,
  //       head_size_rounded}, opts.dtype(at::kFloat));
  //   params.softmax_lseaccum_ptr = softmax_lse_accum.data_ptr();
  //   params.oaccum_ptr = out_accum.data_ptr();
  // }

  run_fwd_kvcache(params, stream);
}

int64_t getIntFromVoidPtr(void *data, size_t &pos) {
  int64_t *intPtr =
      reinterpret_cast<int64_t *>(static_cast<char *>(data) + pos);
  pos += sizeof(int64_t);
  return *intPtr;
}

float getFloatFromVoidPtr(void *data, size_t &pos) {
  float *floatPtr = reinterpret_cast<float *>(static_cast<char *>(data) + pos);
  pos += sizeof(float);
  return *floatPtr;
}

#ifdef __cplusplus
extern "C" {
#endif

void run_flash_attn_fwd(void **tensors, void *extra_args, cudaStream_t stream) {
  size_t pos = 0;
  auto q_batch_stride =
      static_cast<uint32_t>(getIntFromVoidPtr(extra_args, pos));
  auto k_batch_stride =
      static_cast<uint32_t>(getIntFromVoidPtr(extra_args, pos));
  auto v_batch_stride =
      static_cast<uint32_t>(getIntFromVoidPtr(extra_args, pos));
  auto o_batch_stride =
      static_cast<uint32_t>(getIntFromVoidPtr(extra_args, pos));
  auto q_row_stride = static_cast<uint32_t>(getIntFromVoidPtr(extra_args, pos));
  auto k_row_stride = static_cast<uint32_t>(getIntFromVoidPtr(extra_args, pos));
  auto v_row_stride = static_cast<uint32_t>(getIntFromVoidPtr(extra_args, pos));
  auto o_row_stride = static_cast<uint32_t>(getIntFromVoidPtr(extra_args, pos));
  auto q_head_stride =
      static_cast<uint32_t>(getIntFromVoidPtr(extra_args, pos));
  auto k_head_stride =
      static_cast<uint32_t>(getIntFromVoidPtr(extra_args, pos));
  auto v_head_stride =
      static_cast<uint32_t>(getIntFromVoidPtr(extra_args, pos));
  auto o_head_stride =
      static_cast<uint32_t>(getIntFromVoidPtr(extra_args, pos));
  auto b = static_cast<uint32_t>(getIntFromVoidPtr(extra_args, pos));
  auto h = static_cast<uint32_t>(getIntFromVoidPtr(extra_args, pos));
  auto h_k = static_cast<uint32_t>(getIntFromVoidPtr(extra_args, pos));
  auto d = static_cast<uint32_t>(getIntFromVoidPtr(extra_args, pos));
  auto d_rounded = static_cast<uint32_t>(getIntFromVoidPtr(extra_args, pos));
  auto softmax_scale = static_cast<float>(getFloatFromVoidPtr(extra_args, pos));
  auto seqlen_q = static_cast<uint32_t>(getIntFromVoidPtr(extra_args, pos));
  auto seqlen_k = static_cast<uint32_t>(getIntFromVoidPtr(extra_args, pos));
  auto seqlen_q_rounded =
      static_cast<uint32_t>(getIntFromVoidPtr(extra_args, pos));
  auto seqlen_k_rounded =
      static_cast<uint32_t>(getIntFromVoidPtr(extra_args, pos));
  auto p_dropout = static_cast<float>(getFloatFromVoidPtr(extra_args, pos));
  auto window_size_left =
      static_cast<uint32_t>(getIntFromVoidPtr(extra_args, pos));
  auto window_size_right =
      static_cast<uint32_t>(getIntFromVoidPtr(extra_args, pos));

  // tensors: q, k, v, rng_state, o, softmax_lse, softmax_sum
  run_mha(tensors[0], tensors[1], tensors[2], tensors[4], tensors[5],
          tensors[6], tensors[3],
          /*q_batch_stride*/ q_batch_stride,
          /*k_batch_stride*/ k_batch_stride,
          /*v_batch_stride*/ v_batch_stride,
          /*o_batch_stride*/ o_batch_stride,
          /*q_row_stride*/ q_row_stride,
          /*k_row_stride*/ k_row_stride,
          /*v_row_stride*/ v_row_stride,
          /*o_row_stride*/ o_row_stride,
          /*q_head_stride*/ q_head_stride,
          /*k_head_stride*/ k_head_stride,
          /*v_head_stride*/ v_head_stride,
          /*o_head_stride*/ o_head_stride,
          /*b*/ b,
          /*h*/ h,
          /*h_k*/ h_k,
          /*d*/ d,
          /*d_rounded*/ d_rounded,
          /*softmax_scale*/ softmax_scale,
          /*seqlen_q*/ seqlen_q,
          /*seqlen_k*/ seqlen_k,
          /*seqlen_q_rounded*/ seqlen_q_rounded,
          /*seqlen_k_rounded*/ seqlen_k_rounded,
          /*p_dropout*/ p_dropout,
          /*window_size_left*/ window_size_left,
          /*window_size_right*/ window_size_right, stream);
}

void run_flash_attn_bwd(void **tensors, void *extra_args, cudaStream_t stream) {
  size_t pos = 0;
  auto q_batch_stride =
      static_cast<uint32_t>(getIntFromVoidPtr(extra_args, pos));
  auto k_batch_stride =
      static_cast<uint32_t>(getIntFromVoidPtr(extra_args, pos));
  auto v_batch_stride =
      static_cast<uint32_t>(getIntFromVoidPtr(extra_args, pos));
  auto o_batch_stride =
      static_cast<uint32_t>(getIntFromVoidPtr(extra_args, pos));
  auto q_row_stride = static_cast<uint32_t>(getIntFromVoidPtr(extra_args, pos));
  auto k_row_stride = static_cast<uint32_t>(getIntFromVoidPtr(extra_args, pos));
  auto v_row_stride = static_cast<uint32_t>(getIntFromVoidPtr(extra_args, pos));
  auto o_row_stride = static_cast<uint32_t>(getIntFromVoidPtr(extra_args, pos));
  auto q_head_stride =
      static_cast<uint32_t>(getIntFromVoidPtr(extra_args, pos));
  auto k_head_stride =
      static_cast<uint32_t>(getIntFromVoidPtr(extra_args, pos));
  auto v_head_stride =
      static_cast<uint32_t>(getIntFromVoidPtr(extra_args, pos));
  auto o_head_stride =
      static_cast<uint32_t>(getIntFromVoidPtr(extra_args, pos));
  auto b = static_cast<uint32_t>(getIntFromVoidPtr(extra_args, pos));
  auto h = static_cast<uint32_t>(getIntFromVoidPtr(extra_args, pos));
  auto h_k = static_cast<uint32_t>(getIntFromVoidPtr(extra_args, pos));
  auto d = static_cast<uint32_t>(getIntFromVoidPtr(extra_args, pos));
  auto d_rounded = static_cast<uint32_t>(getIntFromVoidPtr(extra_args, pos));
  auto softmax_scale = static_cast<float>(getFloatFromVoidPtr(extra_args, pos));
  auto seqlen_q = static_cast<uint32_t>(getIntFromVoidPtr(extra_args, pos));
  auto seqlen_k = static_cast<uint32_t>(getIntFromVoidPtr(extra_args, pos));
  auto seqlen_q_rounded =
      static_cast<uint32_t>(getIntFromVoidPtr(extra_args, pos));
  auto seqlen_k_rounded =
      static_cast<uint32_t>(getIntFromVoidPtr(extra_args, pos));
  auto p_dropout = static_cast<float>(getFloatFromVoidPtr(extra_args, pos));
  auto window_size_left =
      static_cast<uint32_t>(getIntFromVoidPtr(extra_args, pos));
  auto window_size_right =
      static_cast<uint32_t>(getIntFromVoidPtr(extra_args, pos));

  //tensors: dout, q, k, v, out, softmax_lse, rng_state, dq, dk, dv, d_softmax, dq_accum
  run_mha_bwd(tensors[1], tensors[2], tensors[3], tensors[4], tensors[0],
              tensors[7], tensors[8], tensors[9], tensors[11], tensors[5],
              tensors[10], tensors[6],
              /*q_batch_stride*/ q_batch_stride,
              /*k_batch_stride*/ k_batch_stride,
              /*v_batch_stride*/ v_batch_stride,
              /*o_batch_stride*/ o_batch_stride,
              /*q_row_stride*/ q_row_stride,
              /*k_row_stride*/ k_row_stride,
              /*v_row_stride*/ v_row_stride,
              /*o_row_stride*/ o_row_stride,
              /*q_head_stride*/ q_head_stride,
              /*k_head_stride*/ k_head_stride,
              /*v_head_stride*/ v_head_stride,
              /*o_head_stride*/ o_head_stride,
              /*b*/ b,
              /*h*/ h,
              /*h_k*/ h_k,
              /*d*/ d,
              /*d_rounded*/ d_rounded,
              /*softmax_scale*/ softmax_scale,
              /*seqlen_q*/ seqlen_q,
              /*seqlen_k*/ seqlen_k,
              /*seqlen_q_rounded*/ seqlen_q_rounded,
              /*seqlen_k_rounded*/ seqlen_k_rounded,
              /*p_dropout*/ p_dropout,
              /*window_size_left*/ window_size_left,
              /*window_size_right*/ window_size_right, stream);
}

void run_flash_attn_kvcache(void **tensors, void *extra_args,
                            cudaStream_t stream) {
  size_t pos = 0;
  auto q_batch_stride =
      static_cast<uint32_t>(getIntFromVoidPtr(extra_args, pos));
  auto k_batch_stride =
      static_cast<uint32_t>(getIntFromVoidPtr(extra_args, pos));
  auto v_batch_stride =
      static_cast<uint32_t>(getIntFromVoidPtr(extra_args, pos));
  auto knew_batch_stride =
      static_cast<uint32_t>(getIntFromVoidPtr(extra_args, pos));
  auto vnew_batch_stride =
      static_cast<uint32_t>(getIntFromVoidPtr(extra_args, pos));
  auto o_batch_stride =
      static_cast<uint32_t>(getIntFromVoidPtr(extra_args, pos));
  auto q_row_stride = static_cast<uint32_t>(getIntFromVoidPtr(extra_args, pos));
  auto k_row_stride = static_cast<uint32_t>(getIntFromVoidPtr(extra_args, pos));
  auto v_row_stride = static_cast<uint32_t>(getIntFromVoidPtr(extra_args, pos));
  auto knew_row_stride =
      static_cast<uint32_t>(getIntFromVoidPtr(extra_args, pos));
  auto vnew_row_stride =
      static_cast<uint32_t>(getIntFromVoidPtr(extra_args, pos));
  auto o_row_stride = static_cast<uint32_t>(getIntFromVoidPtr(extra_args, pos));
  auto q_head_stride =
      static_cast<uint32_t>(getIntFromVoidPtr(extra_args, pos));
  auto k_head_stride =
      static_cast<uint32_t>(getIntFromVoidPtr(extra_args, pos));
  auto v_head_stride =
      static_cast<uint32_t>(getIntFromVoidPtr(extra_args, pos));
  auto knew_head_stride =
      static_cast<uint32_t>(getIntFromVoidPtr(extra_args, pos));
  auto vnew_head_stride =
      static_cast<uint32_t>(getIntFromVoidPtr(extra_args, pos));
  auto o_head_stride =
      static_cast<uint32_t>(getIntFromVoidPtr(extra_args, pos));
  auto b = static_cast<uint32_t>(getIntFromVoidPtr(extra_args, pos));
  auto h = static_cast<uint32_t>(getIntFromVoidPtr(extra_args, pos));
  auto h_k = static_cast<uint32_t>(getIntFromVoidPtr(extra_args, pos));
  auto d = static_cast<uint32_t>(getIntFromVoidPtr(extra_args, pos));
  auto d_rounded = static_cast<uint32_t>(getIntFromVoidPtr(extra_args, pos));
  auto seqlen_knew = static_cast<uint32_t>(getIntFromVoidPtr(extra_args, pos));
  auto softmax_scale = static_cast<float>(getFloatFromVoidPtr(extra_args, pos));
  auto seqlen_q = static_cast<uint32_t>(getIntFromVoidPtr(extra_args, pos));
  auto seqlen_k = static_cast<uint32_t>(getIntFromVoidPtr(extra_args, pos));
  auto seqlen_q_rounded =
      static_cast<uint32_t>(getIntFromVoidPtr(extra_args, pos));
  auto seqlen_k_rounded =
      static_cast<uint32_t>(getIntFromVoidPtr(extra_args, pos));
  auto window_size_left = static_cast<int>(getIntFromVoidPtr(extra_args, pos));
  auto window_size_right = static_cast<int>(getIntFromVoidPtr(extra_args, pos));

  run_mha_fwd_with_kvcache(tensors[0], tensors[1], tensors[2], tensors[3],
                           tensors[4], tensors[5], tensors[6], tensors[7],
                           /*q_batch_stride*/ q_batch_stride,
                           /*k_batch_stride*/ k_batch_stride,
                           /*v_batch_stride*/ v_batch_stride,
                           /*knew_batch_stride*/ knew_batch_stride,
                           /*vnew_batch_stride*/ vnew_batch_stride,
                           /*o_batch_stride*/ o_batch_stride,
                           /*q_row_stride*/ q_row_stride,
                           /*k_row_stride*/ k_row_stride,
                           /*v_row_stride*/ v_row_stride,
                           /*knew_row_stride*/ knew_row_stride,
                           /*vnew_row_stride*/ vnew_row_stride,
                           /*o_row_stride*/ o_row_stride,
                           /*q_head_stride*/ q_head_stride,
                           /*k_head_stride*/ k_head_stride,
                           /*v_head_stride*/ v_head_stride,
                           /*knew_head_stride*/ knew_head_stride,
                           /*vnew_head_stride*/ vnew_head_stride,
                           /*o_head_stride*/ o_head_stride,
                           /*b*/ b,
                           /*h*/ h,
                           /*h_k*/ h_k,
                           /*d*/ d,
                           /*d_rounded*/ d_rounded,
                           /*seqlen_knew*/ seqlen_knew,
                           /*softmax_scale*/ softmax_scale,
                           /*seqlen_q*/ seqlen_q,
                           /*seqlen_k*/ seqlen_k,
                           /*seqlen_q_rounded*/ seqlen_q_rounded,
                           /*seqlen_k_rounded*/ seqlen_k_rounded,
                           /*window_size_left*/ window_size_left,
                           /*window_size_right*/ window_size_right, stream);
}
#ifdef __cplusplus
}
#endif
