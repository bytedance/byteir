//===- AceDialect.cpp -----------------------------------------------------===//
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
//
// Some code comes from openxla/stablehlo project, the original license:
// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
// Copyright 2022 The StableHLO Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/Ace/AceDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::ace;

#include "byteir/Dialect/Ace/AceOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// ace Dialect Interfaces
//===----------------------------------------------------------------------===//

namespace {
struct AceInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  // Operations in ace dialect are always legal to inline
  bool isLegalToInline(Operation *, Region *, bool, IRMapping &) const final {
    return true;
  }
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// ace dialect.
//===----------------------------------------------------------------------===//

void AceDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "byteir/Dialect/Ace/AceOps.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "byteir/Dialect/Ace/AceOpsTypes.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "byteir/Dialect/Ace/AceOpsAttributes.cpp.inc"
      >();
  addInterfaces<AceInlinerInterface>();
}

#define GET_OP_CLASSES
#include "byteir/Dialect/Ace/AceOps.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "byteir/Dialect/Ace/AceOpsTypes.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "byteir/Dialect/Ace/AceOpsAttributes.cpp.inc"

//===----------------------------------------------------------------------===//
// ConstOp
//===----------------------------------------------------------------------===//

OpFoldResult mlir::ace::ConstOp::fold(FoldAdaptor) { return getValue(); }

//===----------------------------------------------------------------------===//
// ReshapeOp
//===----------------------------------------------------------------------===//

LogicalResult mlir::ace::ReshapeOp::verify() {
  auto operandTy = dyn_cast<RankedTensorType>(getOperand().getType());
  // If the operand type is dynamically shaped there is nothing to verify.
  if (!operandTy || !operandTy.hasStaticShape())
    return success();

  // If the operand type is statically shaped (not required) the number of
  // elements must match that of the result type.
  auto resultTy = cast<RankedTensorType>(getResult().getType());
  assert(resultTy && resultTy.hasStaticShape() &&
         "result type must be statically shaped");
  int64_t numResultElements = resultTy.getNumElements();
  int64_t numOperandElements = operandTy.getNumElements();
  if (numResultElements != numOperandElements)
    return emitOptionalError(getLoc(), "number of output elements (",
                             numResultElements,
                             ") doesn't match expected number of elements (",
                             numOperandElements, ")");

  return success();
}
