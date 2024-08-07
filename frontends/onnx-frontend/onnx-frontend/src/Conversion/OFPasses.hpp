//===- OFPasses.hpp -------------------------------------------------------===//
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

#include "mlir/Pass/Pass.h"

#include "onnx-frontend/src/Conversion/OFCanonicalizer.hpp"
#include "onnx-frontend/src/Conversion/OFCheckNonLowered.hpp"
#include "onnx-frontend/src/Conversion/OFInsertNecessaryCast.hpp"
#include "onnx-frontend/src/Conversion/OFModifyEntryPoint.hpp"
#include "onnx-frontend/src/Conversion/OFRewriteCustomOnnxOps.hpp"
#include "onnx-frontend/src/Conversion/OFRewriteToCustomCall.hpp"

namespace onnx_frontend {

#define GEN_PASS_REGISTRATION
#include "onnx-frontend/src/Conversion/OFPasses.inc"

} // namespace onnx_frontend
