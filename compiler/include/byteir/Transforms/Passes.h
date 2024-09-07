//===- Passes.h ----------------------------------------------*--- C++ -*-===//
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

#ifndef BYTEIR_TRANSFORMS_PASSES_H
#define BYTEIR_TRANSFORMS_PASSES_H

#include "byteir/Transforms/AnchoredPipeline.h"
#include "byteir/Transforms/ApplyPDLPatterns.h"
#include "byteir/Transforms/Bufferize.h"
#include "byteir/Transforms/CMAE.h"
#include "byteir/Transforms/CanonicalizeExt.h"
#include "byteir/Transforms/CollectFunc.h"
#include "byteir/Transforms/CondCanonicalize.h"
#include "byteir/Transforms/FuncTag.h"
#include "byteir/Transforms/GenericDeviceConfig.h"
#include "byteir/Transforms/GraphClusteringByDevice.h"
#include "byteir/Transforms/InsertUniqueId.h"
#include "byteir/Transforms/LoopTag.h"
#include "byteir/Transforms/LoopUnroll.h"
#include "byteir/Transforms/MemoryPlanning.h"
#include "byteir/Transforms/ModuleTag.h"
#include "byteir/Transforms/RemoveFuncBody.h"
#include "byteir/Transforms/RewriteOpToStdCall.h"
#include "byteir/Transforms/SetArgShape.h"
#include "byteir/Transforms/SetSpace.h"
#include "byteir/Transforms/ShapeFuncOutlining.h"
#include "byteir/Transforms/TryCatchModulePipeline.h"

namespace mlir {

/// Generate the code for registering transforms passes.
#define GEN_PASS_REGISTRATION
#include "byteir/Transforms/Passes.h.inc"

} // namespace mlir

#endif // BYTEIR_TRANSFORMS_PASSES_H
