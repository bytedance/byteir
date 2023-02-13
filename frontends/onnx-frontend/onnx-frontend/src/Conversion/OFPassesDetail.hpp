//===- OFPassesDetail.hpp -------------------------------------------------===//
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

#pragma once

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mhlo/IR/hlo_ops.h"

#include "third_party/onnx-mlir/src/Dialect/ONNX/ONNXOps.hpp"

#include "onnx-frontend/src/Support/OFConstants.hpp"
#include "onnx-frontend/src/Support/OFUtils.hpp"

namespace onnx_frontend {

#define GEN_PASS_CLASSES
#include "onnx-frontend/src/Conversion/OFPasses.inc"

} // namespace onnx_frontend
