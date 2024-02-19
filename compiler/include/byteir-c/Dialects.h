//===- Dialects.h ---------------------------------------------*--- C++ -*-===//
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

#ifndef BYTEIR_C_DIALECTS_H
#define BYTEIR_C_DIALECTS_H

#include "mlir-c/RegisterEverything.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Cat, cat);
MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Ace, ace);
MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Ccl, ccl);

MLIR_CAPI_EXPORTED void byteirRegisterDialectExtensions(MlirContext context);

#ifdef __cplusplus
}
#endif

#endif // BYTEIR_C_DIALECTS_H
