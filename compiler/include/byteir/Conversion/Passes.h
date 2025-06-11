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

#ifndef BYTEIR_CONVERSION_PASSES_H
#define BYTEIR_CONVERSION_PASSES_H

#include "byteir/Conversion/FuncToByre/FuncToByre.h"
#include "byteir/Conversion/GPUToNVVM/GPUToNVVM.h"
#include "byteir/Conversion/HloToByreTensor/HloToByreTensor.h"
#include "byteir/Conversion/HloToCat/ConvertHloToCat.h"
#include "byteir/Conversion/HloToCat/FuseHloToCat.h"
#include "byteir/Conversion/HloToCat/HloToCat.h"
#include "byteir/Conversion/HloToTensor/ConvertHloToTensor.h"
#include "byteir/Conversion/LcclToByre/LcclToByre.h"
#include "byteir/Conversion/MemrefToByre/MemrefToByre.h"
#include "byteir/Conversion/ToAIT/ToAIT.h"
#include "byteir/Conversion/ToTIT/ToTIT.h"
#include "byteir/Conversion/ToAce/MhloToAce.h"
#include "byteir/Conversion/ToByre/ToByre.h"
#include "byteir/Conversion/ToGPU/ToGPU.h"
#include "byteir/Conversion/ToHlo/ArithToMhlo.h"
#include "byteir/Conversion/ToLLVM/ToLLVM.h"
#include "byteir/Conversion/ToLinalg/ToLinalg.h"
#include "byteir/Conversion/ToPTX/ToPTX.h"

namespace mlir {

/// Generate the code for registering conversion passes.
#define GEN_PASS_REGISTRATION
#include "byteir/Conversion/Passes.h.inc"

} // namespace mlir

#endif // BYTEIR_CONVERSION_PASSES_H
