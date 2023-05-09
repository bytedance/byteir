//===- Passes.cpp ---------------------------------------------*--- C++ -*-===//
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

#include "byteir-c/Passes.h"

#include "byteir/Conversion/Passes.h"
#include "byteir/Dialect/Ace/Passes.h"
#include "byteir/Dialect/Affine/Passes.h"
#include "byteir/Dialect/Byre/Passes.h"
#include "byteir/Dialect/Linalg/Passes.h"
#include "byteir/Dialect/MemRef/Passes.h"
#include "byteir/Dialect/SCF/Passes.h"
#include "byteir/Dialect/Shape/Passes.h"
#include "byteir/Dialect/Transform/Passes.h"
#include "byteir/Dialect/mhlo/Passes.h"
#include "byteir/Pipelines/InitAllPipelines.h"
#include "byteir/Target/CUDA/ToCUDA.h"
#include "byteir/Target/Cpp/ToCpp.h"
#include "byteir/Target/PTX/ToPTX.h"
#include "byteir/Transforms/Passes.h"
#include "mlir/InitAllTranslations.h"

using namespace mlir;

void byteirRegisterAllPasses() {
  // conversion and common transform passes
  registerByteIRConversionPasses();
  registerByteIRTransformsPasses();

  // dialect specific passes
  registerByteIRAcePasses();
  registerByteIRAffinePasses();
  registerByteIRByrePasses();
  registerByteIRLinalgPasses();
  registerByteIRMemRefPasses();
  registerByteIRMhloPassesExt();
  registerByteIRSCFPasses();
  registerByteIRShapePasses();
  registerByteIRTransformPasses();

  // pipelines
  registerAllByteIRCommonPipelines();
  registerAllByteIRGPUPipelines();
  registerAllByteIRHostPipelines();
}

void byteirRegisterAllTranslations() {
  registerAllTranslations();
  registerToPTXTranslation();
  byteir::registerToCppTranslation();
  byteir::registerToCUDATranslation();
}
