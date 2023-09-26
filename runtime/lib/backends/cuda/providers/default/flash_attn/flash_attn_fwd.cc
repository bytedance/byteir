//===- flash_attn_fwd.cc -----------------------------------*---C++ -*-===//
//
// Copyright 2022 ByteDance Ltd. and/or its affiliates. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

#include "./flash_attn_fwd.h"
#include "./kernels/flash_api.h"
#include "brt/backends/cuda/device/common/util.h"
#include "brt/backends/cuda/device/cuda_allocator.h"
#include "brt/backends/cuda/device/cuda_work_queue.h"
#include "brt/core/common/common.h"
#include "brt/core/framework/op_accessor.h"
#include <cuda_fp16.h>
#include <iostream>

#define InvalidArgs(msg)                                                       \
  common::Status(common::StatusCategory::BRT,                                  \
                 common::StatusCode::INVALID_ARGUMENT, msg);

#define ShapeCheck(shape, batch_size, seqlen, num_heads, head_size_og)         \
  if (shape[0] != batch_size || shape[1] != seqlen || shape[2] != num_heads || \
      shape[3] != head_size_og) {                                              \
    return InvalidArgs("flash attn shape check failed");                       \
  }

namespace brt {
namespace cuda {
FlashAttnFwdOpKernel::FlashAttnFwdOpKernel(const OpKernelInfo &info)
    : OpKernel(info, false, false, false, false) {}

// byre.compute @byteir.flash_attn_fwd(q_padded, k_padded, v_padded, out_padded,
// softmax_lse, softmax_ptr, rng_state) {causal, dropout_p, softmax_scale,
// return_softmax} output: out, q_padded, k_padded, v_padded, out_padded,
// softmax_lse, softmax_ptr, rng_state(2xi64)
common::Status FlashAttnFwdOpKernel::RunImpl(const ExecutionContext &ctx) {
  OpAccessor accessor(info_, ctx.exec_frame);
  // args
  void *q_ptr = accessor.GetArgAsyncValueRef(0);
  void *k_ptr = accessor.GetArgAsyncValueRef(1);
  void *v_ptr = accessor.GetArgAsyncValueRef(2);
  void *rng_state_ptr = accessor.GetArgAsyncValueRef(3);
  void *o_ptr = accessor.GetArgAsyncValueRef(4);
  void *softmax_lse_ptr = accessor.GetArgAsyncValueRef(5);
  void *softmax_ptr = accessor.GetArgAsyncValueRef(6);

  // check rng_state
  // uint64_t *h_rng_state = new uint64_t[2];
  // cudaMemcpy(h_rng_state, rng_state_ptr, 2 * sizeof(uint64_t),
  // cudaMemcpyDeviceToHost); std::cout << h_rng_state[0] << "," <<
  // h_rng_state[1] << std::endl; cudaDeviceSynchronize();

  // attr
  const bool is_causal = accessor.GetAttrAsBool("causal");
  const float p_dropout = accessor.GetAttrAsFloat("dropout_p");
  const float softmax_scale = accessor.GetAttrAsFloat("softmax_scale");
  const bool return_softmax = accessor.GetAttrAsBool("return_softmax");

  softmax_ptr = return_softmax ? softmax_ptr : nullptr;

  const auto q_shape = accessor.GetArgShape(0);
  const auto k_shape = accessor.GetArgShape(1);
  const auto v_shape = accessor.GetArgShape(2);
  const auto o_shape = accessor.GetArgShape(4);
  int64_t o_rank = o_shape.size();
  int64_t q_rank = q_shape.size();
  int64_t k_rank = k_shape.size();
  int64_t v_rank = v_shape.size();
  if (o_rank != 4 || q_rank != 4 || k_rank != 4 || v_rank != 4) {
    return InvalidArgs("flash-attn expects input tensors of rank 4.");
  }

  // shape check
  const int batch_size_o = o_shape[0];
  const int seqlen_o = o_shape[1];
  const int num_heads_o = o_shape[2];
  const int head_size_og_o = o_shape[3];
  const int batch_size_q = q_shape[0];
  const int seqlen_q = q_shape[1];
  const int num_heads_q = q_shape[2];
  const int head_size_og_q = q_shape[3];
  const int batch_size_k = k_shape[0];
  const int seqlen_k = k_shape[1];
  const int num_heads_k = k_shape[2];
  const int head_size_og_k = k_shape[3];
  const int batch_size_v = v_shape[0];
  const int seqlen_v = v_shape[1];
  const int num_heads_v = v_shape[2];
  const int head_size_og_v = v_shape[3];
  if (batch_size_q <= 0) {
    return InvalidArgs("batch size must be postive");
  }
  if (head_size_og_q > 256) {
    return InvalidArgs(
        "FlashAttention forward only supports head dimension at most 256");
  }
  if (num_heads_q % num_heads_k != 0) {
    return InvalidArgs(
        "Number of heads in key/value must divide number of heads in query");
  }
  ShapeCheck(o_shape, batch_size_q, seqlen_q, num_heads_q, head_size_og_q);
  ShapeCheck(k_shape, batch_size_q, seqlen_k, num_heads_k, head_size_og_q);
  ShapeCheck(v_shape, batch_size_q, seqlen_k, num_heads_k, head_size_og_q);
  if (head_size_og_q % 8 != 0) {
    // TODO: Handle head sizes that are not a multiple of 8 via some padding.
    return InvalidArgs("only supports head sizes that are a multiple of 8");
  }

  // dtype check
  DTypeEnum q_dtype = accessor.GetArgDTypeEnum(0);
  DTypeEnum k_dtype = accessor.GetArgDTypeEnum(1);
  DTypeEnum v_dtype = accessor.GetArgDTypeEnum(2);
  DTypeEnum o_dtype = accessor.GetArgDTypeEnum(4);
  if (o_dtype != q_dtype || q_dtype != k_dtype || k_dtype != v_dtype) {
    return InvalidArgs(
        "query, key, value, and output must have the same dtype");
  }

  auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
  const int head_size = round_multiple(head_size_og_q, 8);
  const int head_size_rounded = round_multiple(head_size, 32);
  const int seqlen_q_rounded = round_multiple(seqlen_q, 128);
  const int seqlen_k_rounded = round_multiple(seqlen_k, 128);

  cudaStream_t stream =
      static_cast<CUDAWorkQueue *>(ctx.work_queue)->GetComputeStream();

  uint32_t q_batch_stride = q_shape[1] * q_shape[2] * q_shape[3];
  uint32_t k_batch_stride = k_shape[1] * k_shape[2] * k_shape[3];
  uint32_t v_batch_stride = v_shape[1] * v_shape[2] * v_shape[3];
  uint32_t o_batch_stride = o_shape[1] * o_shape[2] * o_shape[3];
  uint32_t q_row_stride = q_shape[2] * q_shape[3];
  uint32_t k_row_stride = k_shape[2] * k_shape[3];
  uint32_t v_row_stride = v_shape[2] * v_shape[3];
  uint32_t o_row_stride = o_shape[2] * o_shape[3];
  uint32_t q_head_stride = q_shape[3];
  uint32_t k_head_stride = k_shape[3];
  uint32_t v_head_stride = v_shape[3];
  uint32_t o_head_stride = o_shape[3];

  // std::cout << "params:" << std::endl;
  // std::cout << "q_batch_stride: " << q_batch_stride << std::endl;
  // std::cout << "k_batch_stride: " << k_batch_stride << std::endl;
  // std::cout << "v_batch_stride: " << v_batch_stride << std::endl;
  // std::cout << "o_batch_stride: " << o_batch_stride << std::endl;
  // std::cout << "q_row_stride: " << q_row_stride << std::endl;
  // std::cout << "k_row_stride: " << k_row_stride << std::endl;
  // std::cout << "v_row_stride: " << v_row_stride << std::endl;
  // std::cout << "o_row_stride: " << o_row_stride << std::endl;
  // std::cout << "q_head_stride: " << q_head_stride << std::endl;
  // std::cout << "k_head_stride: " << k_head_stride << std::endl;
  // std::cout << "v_head_stride: " << v_head_stride << std::endl;
  // std::cout << "o_head_stride: " << o_head_stride << std::endl;
  // std::cout << "batch_size_q: " << batch_size_q << std::endl;
  // std::cout << "num_heads_q: " << num_heads_q << std::endl;
  // std::cout << "num_heads_k: " << num_heads_k << std::endl;
  // std::cout << "head_size: " << head_size << std::endl;
  // std::cout << "head_size_rounded: " << head_size_rounded << std::endl;
  // std::cout << "softmax_scale: " << softmax_scale << std::endl;
  // std::cout << "seqlen_q: " << seqlen_q << std::endl;
  // std::cout << "seqlen_k: " << seqlen_k << std::endl;
  // std::cout << "seqlen_q_rounded: " << seqlen_q_rounded << std::endl;
  // std::cout << "seqlen_k_rounded: " << seqlen_k_rounded << std::endl;
  // std::cout << "is_causal: " << is_causal << std::endl;

  kernel::run_mha(q_ptr, k_ptr, v_ptr, o_ptr, softmax_lse_ptr, softmax_ptr,
                  rng_state_ptr,
                  /* cu_seqlens_q_ptr */ nullptr,
                  /* cu_seqlens_k_ptr */ nullptr,
                  /* q_batch_stride */ q_batch_stride,
                  /* k_batch_stride */ k_batch_stride,
                  /* v_batch_stride */ v_batch_stride,
                  /* o_batch_stride */ o_batch_stride,
                  /* q_row_stride   */ q_row_stride,
                  /* k_row_stride   */ k_row_stride,
                  /* v_row_stride   */ v_row_stride,
                  /* o_row_stride   */ o_row_stride,
                  /* q_head_stride  */ q_head_stride,
                  /* k_head_stride  */ k_head_stride,
                  /* v_head_stride  */ v_head_stride,
                  /* o_head_stride  */ o_head_stride,
                  /* b */ batch_size_q,
                  /* h */ num_heads_q,
                  /* h_k */ num_heads_k,
                  /* d */ head_size,
                  /* d_rounded */ head_size_rounded,
                  /* softmax_scale*/ softmax_scale,
                  /* seqlen_q */ seqlen_q,
                  /* seqlen_k */ seqlen_k,
                  /* seqlen_q_rounded */ seqlen_q_rounded,
                  /* seqlen_k_rounded */ seqlen_k_rounded,
                  /* p_dropout */ p_dropout,
                  /* is_causal */ is_causal,
                  /* stream */ stream);

  return common::Status::OK();
}

} // namespace cuda
} // namespace brt
