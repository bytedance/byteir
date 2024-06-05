//===- Dialects.cpp -------------------------------------------*--- C++ -*-===//
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

#include "byteir-c/Dialects.h"

#include "byteir/Dialect/Ace/AceDialect.h"
#include "byteir/Dialect/Byre/ByreDialect.h"
#include "byteir/Dialect/Byre/Serialization/ByreSerialOps.h"
#include "byteir/Dialect/Cat/IR/CatDialect.h"
#include "byteir/Dialect/Ccl/IR/CclOps.h"
#include "byteir/Dialect/Ccl/TransformOps/CclTransformOps.h"
#include "byteir/Dialect/GPU/TransformOps/GPUExtTransformOps.h"
#include "byteir/Dialect/Linalg/TransformOps/LinalgExtTransformOps.h"
#include "byteir/Dialect/Tensor/IR/TilingInterfaceImpl.h"
#include "byteir/Dialect/Transform/IR/TransformExtOps.h"
#include "byteir/Utils/OpInterfaceUtils.h"
#include "byteir/Utils/PatternMatch.h"
#include "mlir/CAPI/Registration.h"

using namespace mlir;

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Cat, cat, mlir::cat::CatDialect)
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Ace, ace, mlir::ace::AceDialect)
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Ccl, ccl, mlir::ccl::CclDialect)
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Byre, byre, mlir::byre::ByreDialect)
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(
    ByreSerial, byre_serial, mlir::byre::serialization::ByreSerialDialect)

void byteirRegisterDialectExtensions(MlirContext context) {
  DialectRegistry registry;
  registeOpInterfaceExtensions(registry);
  registerPDLPatternHooksInterface(registry);
  ccl::registerTransformDialectExtension(registry);
  linalg_ext::registerTransformDialectExtension(registry);
  transform_ext::registerTransformDialectExtension(registry);
  tensor_ext::registerTilingInterfaceExternalModels(registry);
  gpu_ext::registerTransformDialectExtension(registry);
  unwrap(context)->appendDialectRegistry(registry);
}
