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

#ifndef TORCH_FRONTEND_UTILS_CUSTOMCALLUTIL_H
#define TORCH_FRONTEND_UTILS_CUSTOMCALLUTIL_H

#include "llvm/ADT/StringRef.h"

#define CUSTOM_CALL_NAME_PREFIX "byteir."
#define TF_NAME_PREFIX "tf."

namespace mlir {

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

constexpr llvm::StringRef getNllLossForwardName() {
  return CUSTOM_CALL_NAME_PREFIX "nll_loss_forward";
}

constexpr llvm::StringRef getNllLossBackwardName() {
  return CUSTOM_CALL_NAME_PREFIX "nll_loss_backward";
}

constexpr llvm::StringRef getGeLUName() {
  return CUSTOM_CALL_NAME_PREFIX "gelu";
}

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

constexpr llvm::StringRef getResizeName() {
  return CUSTOM_CALL_NAME_PREFIX "resize";
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

constexpr llvm::StringRef getFlashAttnFwdName() {
  return CUSTOM_CALL_NAME_PREFIX "flash_attn_fwd";
}

constexpr llvm::StringRef getFlashAttnKVCacheName() {
  return CUSTOM_CALL_NAME_PREFIX "flash_attn_kvcache";
}

constexpr llvm::StringRef getFlashAttnBwdName() {
  return CUSTOM_CALL_NAME_PREFIX "flash_attn_bwd";
}
} // namespace mlir

#endif // TORCH_FRONTEND_UTILS_CUSTOMCALLUTIL_H
