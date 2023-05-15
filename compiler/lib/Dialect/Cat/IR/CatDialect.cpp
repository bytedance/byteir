//===- CatDialect.cpp -----------------------------------------------------===//
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

#include "byteir/Dialect/Cat/IR/CatDialect.h"

using namespace mlir;
using namespace mlir::cat;

#include "byteir/Dialect/Cat/IR/CatOpsDialect.cpp.inc"

void CatDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "byteir/Dialect/Cat/IR/CatOps.cpp.inc"
      >();
}

#define GET_OP_CLASSES
#include "byteir/Dialect/Cat/IR/CatOps.cpp.inc"

LogicalResult VerifyBMMLayout(Value lhs, Value rhs, Value out,
                              llvm::StringRef layoutStr) {
  auto lhsType = lhs.getType().cast<ShapedType>();
  auto rhsType = rhs.getType().cast<ShapedType>();
  auto outType = out.getType().cast<ShapedType>();
  if (lhsType.getRank() != 3 || rhsType.getRank() != 3)
    return failure();

  auto lhsShape = lhsType.getShape();
  auto rhsShape = rhsType.getShape();
  if (layoutStr == "rrr" && lhsShape[2] == rhsShape[1])
    return success();
  if (layoutStr == "rcr" && lhsShape[2] == rhsShape[2])
    return success();
  if (layoutStr == "crr" && lhsShape[1] == rhsShape[1])
    return success();
  if (layoutStr == "ccr" && lhsShape[1] == rhsShape[2])
    return success();

  return failure();
}

LogicalResult VerifyGemmLayout(Value lhs, Value rhs, Value out,
                               llvm::StringRef layoutStr) {
  auto lhsShape = lhs.getType().cast<ShapedType>().getShape();
  auto rhsShape = rhs.getType().cast<ShapedType>().getShape();
  auto outShape = out.getType().cast<ShapedType>().getShape();
  if (layoutStr == "rrr" && lhsShape[1] == rhsShape[0])
    return success();
  if (layoutStr == "rcr" && lhsShape[1] == rhsShape[1])
    return success();
  if (layoutStr == "crr" && lhsShape[0] == rhsShape[0])
    return success();
  if (layoutStr == "ccr" && lhsShape[0] == rhsShape[1])
    return success();

  return failure();
}

//===----------------------------------------------------------------------===//
// BatchMatmulOp
//===----------------------------------------------------------------------===/

LogicalResult BatchMatmulOp::verify() {
  return VerifyBMMLayout(this->getLhs(), this->getRhs(), this->getOutput(),
                         this->getLayout());
}

//===----------------------------------------------------------------------===//
// BMMPermuteOp
//===----------------------------------------------------------------------===/

LogicalResult BMMPermuteOp::verify() {
  return VerifyBMMLayout(this->getLhs(), this->getRhs(), this->getOutput(),
                         this->getLayout());
}

//===----------------------------------------------------------------------===//
// GemmOp
//===----------------------------------------------------------------------===/

LogicalResult GemmOp::verify() {
  return VerifyGemmLayout(this->getLhs(), this->getRhs(), this->getOutput(),
                          this->getLayout());
}

//===----------------------------------------------------------------------===//
// GemmBiasOp
//===----------------------------------------------------------------------===/

LogicalResult GemmBiasOp::verify() {
  return VerifyGemmLayout(this->getLhs(), this->getRhs(), this->getOutput(),
                          this->getLayout());
}
