//===- PDLValue.cpp -------------------------------------------*--- C++ -*-===//
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

#include "byteir-c/PDLValue.h"
#include "byteir/Utils/PatternMatch.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"

using namespace mlir;

inline PDLValue unwrap(MlirPDLValue value) {
  switch (value.kind) {
  case MlirPDLValueAttribute:
    return Value::getFromOpaquePointer(value.ptr);
  case MlirPDLValueOperation:
    return reinterpret_cast<Operation *>(value.ptr);
  case MlirPDLValueType:
    return Type::getFromOpaquePointer(value.ptr);
  case MlirPDLValueTypeRange:
    return reinterpret_cast<TypeRange *>(value.ptr);
  case MlirPDLValueValue:
    return Value::getFromOpaquePointer(value.ptr);
  case MlirPDLValueValueRange:
    return reinterpret_cast<ValueRange *>(value.ptr);
  }
  llvm_unreachable("unknown pdl value kind");
}

inline MlirPDLValue wrap(PDLValue value) {
  return {
      const_cast<void *>(value.getAsOpaquePointer()),
      static_cast<MlirPDLValueKind>(static_cast<uint32_t>(value.getKind()))};
}

inline PDLResultList &unwrap(MlirPDLResultListRef value) {
  assert(value.ptr);
  return *reinterpret_cast<PDLResultList *>(value.ptr);
}

inline MlirPDLResultListRef wrap(PDLResultList &value) {
  return {reinterpret_cast<void *>(&value)};
}

MlirAttribute mlirPDLValueCastToMlirAttribute(MlirPDLValue pdlValue) {
  return wrap(unwrap(pdlValue).cast<Attribute>());
}

MlirOperation mlirPDLValueCastToMlirOperation(MlirPDLValue pdlValue) {
  return wrap(unwrap(pdlValue).cast<Operation *>());
}

MlirType mlirPDLValueCastToMlirType(MlirPDLValue pdlValue) {
  return wrap(unwrap(pdlValue).cast<Type>());
}

void mlirPDLValueCastToMlirTypeRange(MlirPDLValue pdlValue, MlirType **types,
                                     intptr_t *ntypes) {
  auto typeRange = unwrap(pdlValue).cast<TypeRange>();
  *ntypes = typeRange.size();
  *types = new MlirType[typeRange.size()];
  for (size_t i = 0; i < typeRange.size(); ++i) {
    (*types)[i] = wrap(typeRange[i]);
  }
}

MlirValue mlirPDLValueCastToMlirValue(MlirPDLValue pdlValue) {
  return wrap(unwrap(pdlValue).cast<Value>());
}

void mlirPDLValueCastToMlirValueRange(MlirPDLValue pdlValue, MlirValue **values,
                                      intptr_t *nvalues) {

  auto valueRange = unwrap(pdlValue).cast<ValueRange>();
  *nvalues = valueRange.size();
  *values = new MlirValue[valueRange.size()];
  for (size_t i = 0; i < valueRange.size(); ++i) {
    (*values)[i] = wrap(valueRange[i]);
  }
}

void mlirPDLResultListEmplaceAttribute(MlirPDLResultListRef pdlResults,
                                       MlirAttribute attr) {
  unwrap(pdlResults).push_back(unwrap(attr));
}

void mlirPDLResultListEmplaceOperation(MlirPDLResultListRef pdlResults,
                                       MlirOperation op) {
  unwrap(pdlResults).push_back(unwrap(op));
}

void mlirPDLResultListEmplaceType(MlirPDLResultListRef pdlResults,
                                  MlirType type) {
  unwrap(pdlResults).push_back(unwrap(type));
}

void mlirPDLResultListEmplaceTypes(MlirPDLResultListRef pdlResults,
                                   MlirType *types, intptr_t ntypes) {
  std::vector<Type> typeRange;
  for (intptr_t i = 0; i < ntypes; ++i) {
    typeRange.push_back(unwrap(types[i]));
  }
  unwrap(pdlResults).push_back(TypeRange(typeRange));
}

void mlirPDLResultListEmplaceValue(MlirPDLResultListRef pdlResults,
                                   MlirValue value) {
  unwrap(pdlResults).push_back(unwrap(value));
}

void mlirPDLResultListEmplaceValues(MlirPDLResultListRef pdlResults,
                                    MlirValue *values, intptr_t nvalues) {
  std::vector<Value> valueRange;
  for (intptr_t i = 0; i < nvalues; ++i) {
    valueRange.push_back(unwrap(values[i]));
  }
  unwrap(pdlResults).push_back(ValueRange(valueRange));
}

void mlirRegisterPDLConstraintFn(MlirContext ctx, MlirStringRef name,
                                 void *pfn) {
  registerPDLConstraintFunction(
      unwrap(ctx), unwrap(name),
      [fn = *reinterpret_cast<std::function<bool(std::vector<MlirPDLValue>)> *>(
           pfn)](PatternRewriter &,
                 ArrayRef<PDLValue> pdlValues) -> LogicalResult {
        std::vector<MlirPDLValue> wrapped;
        wrapped.reserve(pdlValues.size());
        for (auto &&i : pdlValues) {
          wrapped.push_back(wrap(i));
        }
        return success(fn(wrapped));
      });
}

void mlirRegisterPDLRewriteFn(MlirContext ctx, MlirStringRef name, void *pfn) {
  registerPDLRewriteFunction(
      unwrap(ctx), unwrap(name),
      [fn = *reinterpret_cast<std::function<bool(
           MlirOperation, MlirPDLResultListRef, std::vector<MlirPDLValue>,
           std::function<void(MlirOperation)>)> *>(pfn)](
          PatternRewriter &rewriter, PDLResultList &resultList,
          ArrayRef<PDLValue> pdlValues) -> LogicalResult {
        std::vector<MlirPDLValue> wrapped;
        wrapped.reserve(pdlValues.size());
        for (auto &&i : pdlValues) {
          wrapped.push_back(wrap(i));
        }

        MlirOperation insertionPoint;
        if (rewriter.getInsertionPoint() != rewriter.getBlock()->end())
          insertionPoint = wrap(&*rewriter.getInsertionPoint());

        auto onOperationInserted = [&](MlirOperation op) {
          rewriter.getListener()->notifyOperationInserted(unwrap(op));
        };

        if (!fn(insertionPoint, wrap(resultList), wrapped, onOperationInserted))
          return failure();

        return success();
      });
}
