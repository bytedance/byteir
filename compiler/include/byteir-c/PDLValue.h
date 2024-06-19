//===- PDLValue.h --------------------------------------------*--- C++ -*-===//
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

#ifndef BYTEIR_C_PDLVALUE_H
#define BYTEIR_C_PDLVALUE_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

enum MlirPDLValueKind {
  MlirPDLValueAttribute,
  MlirPDLValueOperation,
  MlirPDLValueType,
  MlirPDLValueTypeRange,
  MlirPDLValueValue,
  MlirPDLValueValueRange
};
typedef enum MlirPDLValueKind MlirPDLValueKind;

struct MlirPDLValue {
  void *ptr;
  MlirPDLValueKind kind;
};
typedef struct MlirPDLValue MlirPDLValue;

struct MlirPDLResultListRef {
  void *ptr;
};
typedef struct MlirPDLResultListRef MlirPDLResultListRef;

MLIR_CAPI_EXPORTED MlirAttribute
mlirPDLValueCastToMlirAttribute(MlirPDLValue pdlValue);

MLIR_CAPI_EXPORTED MlirOperation
mlirPDLValueCastToMlirOperation(MlirPDLValue pdlValue);

MLIR_CAPI_EXPORTED MlirType mlirPDLValueCastToMlirType(MlirPDLValue pdlValue);

MLIR_CAPI_EXPORTED void mlirPDLValueCastToMlirTypeRange(MlirPDLValue pdlValue,
                                                        MlirType **types,
                                                        intptr_t *ntypes);

MLIR_CAPI_EXPORTED MlirValue mlirPDLValueCastToMlirValue(MlirPDLValue pdlValue);

MLIR_CAPI_EXPORTED void mlirPDLValueCastToMlirValueRange(MlirPDLValue pdlValue,
                                                         MlirValue **types,
                                                         intptr_t *nvalues);

MLIR_CAPI_EXPORTED void
mlirPDLResultListEmplaceAttribute(MlirPDLResultListRef pdlResults,
                                  MlirAttribute attr);
MLIR_CAPI_EXPORTED void
mlirPDLResultListEmplaceOperation(MlirPDLResultListRef pdlResults,
                                  MlirOperation op);
MLIR_CAPI_EXPORTED void
mlirPDLResultListEmplaceType(MlirPDLResultListRef pdlResults, MlirType type);
MLIR_CAPI_EXPORTED void
mlirPDLResultListEmplaceTypes(MlirPDLResultListRef pdlResults, MlirType *types,
                              intptr_t ntypes);
MLIR_CAPI_EXPORTED void
mlirPDLResultListEmplaceValue(MlirPDLResultListRef pdlResults, MlirValue value);
MLIR_CAPI_EXPORTED void
mlirPDLResultListEmplaceValues(MlirPDLResultListRef pdlResults,
                               MlirValue *values, intptr_t nvalues);

// fn -> std::function<bool(std::vector<MlirPDLValue>)>
MLIR_CAPI_EXPORTED bool mlirRegisterPDLConstraintFn(MlirContext ctx,
                                                    MlirStringRef name,
                                                    void *pfn, bool override);

// fn -> std::function<bool(MlirOperation,
//                          MlirPDLResultList,
//                          std::vector<MlirPDLValue>,
//                          std::function<void(MlirOperation)>)>
MLIR_CAPI_EXPORTED bool mlirRegisterPDLRewriteFn(MlirContext ctx,
                                                 MlirStringRef name, void *pfn,
                                                 bool override);

#ifdef __cplusplus
}
#endif

#endif // BYTEIR_C_PDLVALUE_H
