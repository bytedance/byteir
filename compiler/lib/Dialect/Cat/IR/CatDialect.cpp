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
  auto lhsType = cast<ShapedType>(lhs.getType());
  auto rhsType = cast<ShapedType>(rhs.getType());
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
  auto lhsShape = cast<ShapedType>(lhs.getType()).getShape();
  auto rhsShape = cast<ShapedType>(rhs.getType()).getShape();
  // auto outShape = cast<ShapedType>(out.getType()).getShape();
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

LogicalResult VerifyGemmPermute0213Layout(Value lhs, Value rhs, Value out,
                                          int64_t t1, int64_t t2,
                                          llvm::StringRef layoutStr) {
  auto lhsShape = cast<ShapedType>(lhs.getType()).getShape();
  auto rhsShape = cast<ShapedType>(rhs.getType()).getShape();
  auto outShape = cast<ShapedType>(out.getType()).getShape();
  if (t1 != outShape[2] || t2 != outShape[1])
    return failure();
  if (layoutStr == "rrr" && lhsShape[1] == rhsShape[0]) {
    if (outShape[0] * t1 == lhsShape[0] && t2 * outShape[3] == rhsShape[1])
      return success();
    else
      return failure();
  }
  if (layoutStr == "rcr" && lhsShape[1] == rhsShape[1]) {
    if (outShape[0] * t1 == lhsShape[0] && t2 * outShape[3] == rhsShape[0])
      return success();
    else
      return failure();
  }
  if (layoutStr == "crr" && lhsShape[0] == rhsShape[0]) {
    if (outShape[0] * t1 == lhsShape[1] && t2 * outShape[3] == rhsShape[1])
      return success();
    else
      return failure();
  }
  if (layoutStr == "ccr" && lhsShape[0] == rhsShape[1]) {
    if (outShape[0] * t1 == lhsShape[1] && t2 * outShape[3] == rhsShape[0])
      return success();
    else
      return failure();
  }

  return failure();
}

//===----------------------------------------------------------------------===//
// BMMPermuteOp
//===----------------------------------------------------------------------===/

LogicalResult BMMRRRPermuteOp::verify() {
  return VerifyBMMLayout(this->getLhs(), this->getRhs(), this->getOutput(),
                         "rrr");
}

LogicalResult BMMRCRPermuteOp::verify() {
  return VerifyBMMLayout(this->getLhs(), this->getRhs(), this->getOutput(),
                         "rcr");
}

//===----------------------------------------------------------------------===//
// GemmOp
//===----------------------------------------------------------------------===/

LogicalResult GemmRRROp::verify() {
  return VerifyGemmLayout(this->getLhs(), this->getRhs(), this->getOutput(),
                          "rrr");
}

LogicalResult GemmRCROp::verify() {
  return VerifyGemmLayout(this->getLhs(), this->getRhs(), this->getOutput(),
                          "rcr");
}

//===----------------------------------------------------------------------===//
// GemmBiasOp
//===----------------------------------------------------------------------===/

LogicalResult GemmRRRBiasOp::verify() {
  return VerifyGemmLayout(this->getLhs(), this->getRhs(), this->getOutput(),
                          "rrr");
}

LogicalResult GemmRCRBiasOp::verify() {
  return VerifyGemmLayout(this->getLhs(), this->getRhs(), this->getOutput(),
                          "rcr");
}

//===----------------------------------------------------------------------===//
// GemmPermuteOp
//===----------------------------------------------------------------------===/

LogicalResult GemmRCRPermuteOp::verify() {
  return VerifyGemmPermute0213Layout(this->getLhs(), this->getRhs(),
                                     this->getOutput(), this->getT1(),
                                     this->getT2(), "rcr");
}

LogicalResult GemmRRRPermuteOp::verify() {
  return VerifyGemmPermute0213Layout(this->getLhs(), this->getRhs(),
                                     this->getOutput(), this->getT1(),
                                     this->getT2(), "rrr");
}
