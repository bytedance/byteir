//===- flash_attn_bwd.cc -----------------------------------*---C++ -*-===//
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

#include "./flash_attn_bwd.h"
#include "./kernels/flash_api.h"
#include "brt/backends/cuda/device/common/util.h"
#include "brt/backends/cuda/device/cuda_allocator.h"
#include "brt/backends/cuda/device/cuda_work_queue.h"
#include "brt/core/common/common.h"
#include "brt/core/framework/op_accessor.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
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
FlashAttnBwdOpKernel::FlashAttnBwdOpKernel(const OpKernelInfo &info)
    : OpKernel(info, false, false, false, false) {}

// byre.compute @byteir.flash_attn_bwd(dout, q, k, v, out, softmax_lse,
// rng_state, dq, dk, dv, dsoftmax_sum_ptr) {causal,
// dropout_p,softmax_scale, dq_accum_ptr}
common::Status FlashAttnBwdOpKernel::RunImpl(const ExecutionContext &ctx) {
  OpAccessor accessor(info_, ctx.exec_frame);
  // args
  void *dout_ptr = accessor.GetArgAsyncValueRef(0);
  void *q_ptr = accessor.GetArgAsyncValueRef(1);
  void *k_ptr = accessor.GetArgAsyncValueRef(2);
  void *v_ptr = accessor.GetArgAsyncValueRef(3);
  void *out_ptr = accessor.GetArgAsyncValueRef(4);
  void *softmax_lse_ptr = accessor.GetArgAsyncValueRef(5);
  void *rng_state_ptr = accessor.GetArgAsyncValueRef(6); // TODO : handle rng
  void *dq_ptr = accessor.GetArgAsyncValueRef(7);
  void *dk_ptr = accessor.GetArgAsyncValueRef(8);
  void *dv_ptr = accessor.GetArgAsyncValueRef(9);
  void *dsoftmax_ptr = accessor.GetArgAsyncValueRef(10);
  void *dq_accum_ptr = accessor.GetArgAsyncValueRef(11);

  // attr
  const bool is_causal = accessor.GetAttrAsBool("causal");
  const float p_dropout = accessor.GetAttrAsFloat("dropout_p");
  const float softmax_scale = accessor.GetAttrAsFloat("softmax_scale");

  // device compute capability check
  int nDevices;

  cudaGetDeviceCount(&nDevices);
  bool support_sm8x = false;
  bool support_sm80 = false;
  bool support_sm90 = false;
  for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    support_sm8x = support_sm8x || (prop.major == 8 && prop.minor >= 0);
    support_sm80 = support_sm80 || (prop.major == 8 && prop.minor == 0);
    support_sm90 = support_sm90 || (prop.major == 9 && prop.minor == 0);
  }

  if (!support_sm8x && !support_sm90) {
    return InvalidArgs("FlashAttention only supports Ampere GPUs or newer.");
  }

  // dropout check
  // bool is_dropout = p_dropout > 0.0;
  // if (is_dropout) {
  //   return InvalidArgs("currently, we only support p_dropout == 0");
  // }

  // type check
  const auto dout_type = accessor.GetArgDTypeEnum(0);
  const auto q_type = accessor.GetArgDTypeEnum(1);
  const auto k_type = accessor.GetArgDTypeEnum(2);
  const auto v_type = accessor.GetArgDTypeEnum(3);
  const auto out_type = accessor.GetArgDTypeEnum(4);
  const auto dq_type = accessor.GetArgDTypeEnum(7);
  const auto dk_type = accessor.GetArgDTypeEnum(8);
  const auto dv_type = accessor.GetArgDTypeEnum(9);

  // if (q_type != DTypeEnum::Float16 || q_type != DTypeEnum::BFloat16) {
  //   return InvalidArgs("FlashAttention only support fp16 and bf16 data
  //   type");
  // }
  // if (dout_type != q_type || k_type != q_type || v_type != q_type ||
  //     out_type != q_type || dq_type != q_type || dk_type != q_type ||
  //     dv_type != q_type) {
  //   return InvalidArgs("Args must have the same dtype");
  // }

  // shepe check
  const auto dout_shape = accessor.GetArgShape(0);
  const auto q_shape = accessor.GetArgShape(1);
  const auto k_shape = accessor.GetArgShape(2);
  const auto v_shape = accessor.GetArgShape(3);
  const auto out_shape = accessor.GetArgShape(4);
  const auto dq_shape = accessor.GetArgShape(7);
  const auto dk_shape = accessor.GetArgShape(8);
  const auto dv_shape = accessor.GetArgShape(9);
  const auto dsoftmax_shape = accessor.GetArgShape(10);
  const auto dq_accum_shape = accessor.GetArgShape(11);
  int64_t o_rank = out_shape.size();
  int64_t q_rank = q_shape.size();
  int64_t k_rank = k_shape.size();
  int64_t v_rank = v_shape.size();
  if (o_rank != 4 || q_rank != 4 || k_rank != 4 || v_rank != 4) {
    return InvalidArgs("flash-attn expects input tensors of rank 4.");
  }

  const int batch_size_o = out_shape[0];
  const int seqlen_o = out_shape[1];
  const int num_heads_o = out_shape[2];
  const int head_size_og_o = out_shape[3];
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
        "FlashAttention backword only supports head dimension at most 256");
  }
  if (head_size_og_q > 192 && !support_sm80 && !support_sm90) {
    return InvalidArgs("FlashAttention backward for head dim > 192 requires "
                       "A100/A800 or H100/H800");
  }
  if (num_heads_q % num_heads_k != 0) {
    return InvalidArgs(
        "Number of heads in key/value must divide number of heads in query");
  }
  ShapeCheck(out_shape, batch_size_q, seqlen_q, num_heads_q, head_size_og_q);
  ShapeCheck(dq_shape, batch_size_q, seqlen_q, num_heads_q, head_size_og_q);
  ShapeCheck(dout_shape, batch_size_q, seqlen_q, num_heads_q, head_size_og_q);
  ShapeCheck(k_shape, batch_size_q, seqlen_k, num_heads_k, head_size_og_q);
  ShapeCheck(v_shape, batch_size_q, seqlen_k, num_heads_k, head_size_og_q);
  ShapeCheck(dk_shape, batch_size_q, seqlen_k, num_heads_k, head_size_og_q);
  ShapeCheck(dv_shape, batch_size_q, seqlen_k, num_heads_k, head_size_og_q);

  auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
  const int head_size = round_multiple(head_size_og_q, 8);
  const int head_size_rounded = round_multiple(head_size, 32);
  const int seqlen_q_rounded = round_multiple(seqlen_q, 128);
  const int seqlen_k_rounded = round_multiple(seqlen_k, 128);

  if (dsoftmax_shape[0] != batch_size_q || dsoftmax_shape[1] != num_heads_q ||
      dsoftmax_shape[2] != seqlen_q_rounded) {
    return InvalidArgs("dsoftmax shape check failed.");
  }
  ShapeCheck(dq_accum_shape, batch_size_q, num_heads_q, seqlen_q_rounded,
             head_size_rounded);

  if (head_size_og_q % 8 != 0) {
    // TODO: Handle head sizes that are not a multiple of 8 via some padding.
    return InvalidArgs("only supports head sizes that are a multiple of 8");
  }

  if (num_heads_k != num_heads_q) {
    // TODO: add compiler support when num_heads_k != num_heads_q
    // we need to create dk_expanded and dv_expanded when num_heads_k !=
    // num_heads_q reference in flash attn v2 as follows:
    // ======================================
    // at::Tensor dk_expanded, dv_expanded;
    // if (num_heads_k != num_heads) {  // MQA / GQA
    //     dk_expanded = torch::empty({batch_size, seqlen_k, num_heads,
    //     head_size}, opts); dv_expanded = torch::empty({batch_size, seqlen_k,
    //     num_heads, head_size}, opts);
    // } else {
    //     dk_expanded = dk;
    //     dv_expanded = dv;
    // }
    // ======================================
    return InvalidArgs("currently, we only support num_heads_k == num_heads_q");
  }

  // dtype check
  DTypeEnum o_dtype = accessor.GetArgDTypeEnum(0);
  DTypeEnum q_dtype = accessor.GetArgDTypeEnum(1);
  DTypeEnum k_dtype = accessor.GetArgDTypeEnum(2);
  DTypeEnum v_dtype = accessor.GetArgDTypeEnum(3);
  if (o_dtype != q_dtype || q_dtype != k_dtype || k_dtype != v_dtype) {
    return InvalidArgs(
        "query, key, value, and output must have the same dtype");
  }

  // bool loop = seqlen_k > blocksize_c;
  // TODO: change later, for now set to true for simplicity
  bool loop = true;

  cudaStream_t stream =
      static_cast<CUDAWorkQueue *>(ctx.work_queue)->GetComputeStream();

  uint32_t q_batch_stride = q_shape[1] * q_shape[2] * q_shape[3];
  uint32_t k_batch_stride = k_shape[1] * k_shape[2] * k_shape[3];
  uint32_t v_batch_stride = v_shape[1] * v_shape[2] * v_shape[3];
  uint32_t o_batch_stride = out_shape[1] * out_shape[2] * out_shape[3];
  uint32_t q_row_stride = q_shape[2] * q_shape[3];
  uint32_t k_row_stride = k_shape[2] * k_shape[3];
  uint32_t v_row_stride = v_shape[2] * v_shape[3];
  uint32_t o_row_stride = out_shape[2] * out_shape[3];
  uint32_t q_head_stride = q_shape[3];
  uint32_t k_head_stride = k_shape[3];
  uint32_t v_head_stride = v_shape[3];
  uint32_t o_head_stride = out_shape[3];

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
  // std::cout << "head_size_og_q: " << head_size_og_q << std::endl;
  // std::cout << "head_size_rounded: " << head_size_rounded << std::endl;
  // std::cout << "softmax_scale: " << softmax_scale << std::endl;
  // std::cout << "seqlen_q: " << seqlen_q << std::endl;
  // std::cout << "seqlen_k: " << seqlen_k << std::endl;
  // std::cout << "seqlen_q_rounded: " << seqlen_q_rounded << std::endl;
  // std::cout << "seqlen_k_rounded: " << seqlen_k_rounded << std::endl;
  // std::cout << "is_causal: " << is_causal << std::endl;

  kernel::run_mha_bwd(
      q_ptr, k_ptr, v_ptr, out_ptr, dout_ptr, dq_ptr, dk_ptr, dv_ptr,
      /* cu_seqlens_q_ptr */ nullptr,
      /* cu_seqlens_k_ptr */ nullptr,
      /* dq_accum_ptr */ loop ? dq_accum_ptr : nullptr,
      /* dk_accum_ptr */ nullptr,
      /* dv_accum_ptr */ nullptr, softmax_lse_ptr, dsoftmax_ptr, rng_state_ptr,
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
      /* d */ head_size_og_q,
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
