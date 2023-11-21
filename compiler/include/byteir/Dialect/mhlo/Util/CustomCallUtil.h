//===- CustomCallUtil.h ---------------------------------------*--- C++ -*-===//
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

#ifndef BYTEIR_DIALECT_MHLO_UTIL_CUSTOMCALLUTIL_H
#define BYTEIR_DIALECT_MHLO_UTIL_CUSTOMCALLUTIL_H

#include "llvm/ADT/StringRef.h"

#define CUSTOM_CALL_NAME_PREFIX "byteir."
#define TF_NAME_PREFIX "tf."
#define PYTORCH_NAME_PREFIX "pytorch."

namespace mlir {
constexpr llvm::StringRef getGemvhbmpimName() {
  return PYTORCH_NAME_PREFIX "gemv_hbmpim";
}

constexpr llvm::StringRef getAddUpmemName() {
  return PYTORCH_NAME_PREFIX "add_upmem";
}
constexpr llvm::StringRef getSubUpmemName() {
  return PYTORCH_NAME_PREFIX "sub_upmem";
}
constexpr llvm::StringRef getMulUpmemName() {
  return PYTORCH_NAME_PREFIX "mul_upmem";
}
constexpr llvm::StringRef getDivUpmemName() {
  return PYTORCH_NAME_PREFIX "div_upmem";
}
constexpr llvm::StringRef getRsqrtUpmemName() {
  return PYTORCH_NAME_PREFIX "rsqrt_upmem";
}
constexpr llvm::StringRef getTanhUpmemName() {
  return PYTORCH_NAME_PREFIX "tanh_upmem";
}
constexpr llvm::StringRef getReluUpmemName() {
  return PYTORCH_NAME_PREFIX "relu_upmem";
}
constexpr llvm::StringRef getSoftmaxUpmemName() {
  return PYTORCH_NAME_PREFIX "softmax_upmem";
}

constexpr llvm::StringRef getGemvUpmemName() {
  return PYTORCH_NAME_PREFIX "gemv_upmem";
}
constexpr llvm::StringRef getGemvhbmpimName() {
  return PYTORCH_NAME_PREFIX "gemv_hbmpim";
}




constexpr llvm::StringRef getCustomCallAttrName() { return "byteir_attrs"; }

constexpr llvm::StringRef getNonZeroName() {
  return CUSTOM_CALL_NAME_PREFIX "non_zero";
}

constexpr llvm::StringRef getSoftmaxName() {
  return CUSTOM_CALL_NAME_PREFIX "softmax";
}

constexpr llvm::StringRef getLogSoftmaxName() {
  return CUSTOM_CALL_NAME_PREFIX "log_softmax";
}

constexpr llvm::StringRef getGeLUName() {
  return CUSTOM_CALL_NAME_PREFIX "gelu";
}

constexpr llvm::StringRef getErfName() { return CUSTOM_CALL_NAME_PREFIX "erf"; }

constexpr llvm::StringRef getTopKName() {
  return CUSTOM_CALL_NAME_PREFIX "top_k";
}

constexpr llvm::StringRef getArgMaxName() {
  return CUSTOM_CALL_NAME_PREFIX "arg_max";
}

constexpr llvm::StringRef getArgMinName() {
  return CUSTOM_CALL_NAME_PREFIX "arg_min";
}

constexpr llvm::StringRef getLayerNormName() {
  return CUSTOM_CALL_NAME_PREFIX "layer_norm";
}

constexpr llvm::StringRef getL2NormName() {
  return CUSTOM_CALL_NAME_PREFIX "l2_norm";
}

constexpr llvm::StringRef getOneHotName() {
  return CUSTOM_CALL_NAME_PREFIX "one_hot";
}

constexpr llvm::StringRef getAddNName() {
  return CUSTOM_CALL_NAME_PREFIX "addn";
}

constexpr llvm::StringRef getQuantizeName() {
  return CUSTOM_CALL_NAME_PREFIX "quantize";
}

constexpr llvm::StringRef getDequantizeName() {
  return CUSTOM_CALL_NAME_PREFIX "dequantize";
}

constexpr llvm::StringRef getRngUniformName() {
  return CUSTOM_CALL_NAME_PREFIX "rng_uniform";
}

constexpr llvm::StringRef getFlashAttnFwdName() {
  return CUSTOM_CALL_NAME_PREFIX "flash_attn_fwd";
}

constexpr llvm::StringRef getFlashAttnBwdName() {
  return CUSTOM_CALL_NAME_PREFIX "flash_attn_bwd";
}

constexpr llvm::StringRef getDynamicPartitionName() {
  return TF_NAME_PREFIX "DynamicPartition";
}

constexpr llvm::StringRef getDynamicStitchName() {
  return TF_NAME_PREFIX "DynamicStitch";
}

constexpr llvm::StringRef getDynamicMaskStitchName() {
  return TF_NAME_PREFIX "DynamicMaskStitch";
}

constexpr llvm::StringRef getWhereName() { return TF_NAME_PREFIX "Where"; }

} // namespace mlir

#undef TF_NAME_PREFIX
#undef CUSTOM_CALL_NAME_PREFIX
#undef PYTORCH_NAME_PREFIX

#endif // BYTEIR_DIALECT_MHLO_UTIL_CUSTOMCALLUTIL_H
