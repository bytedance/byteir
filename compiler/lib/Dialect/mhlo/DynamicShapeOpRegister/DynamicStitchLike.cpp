//===- DynamicStitchLike.h ------------------------------------*--- C++ -*-===//
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

#include "byteir/Dialect/mhlo/DynamicShapeOpRegister/Register.h"
#include "byteir/Dialect/mhlo/Util/CustomCallUtil.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"

#define DEBUG_TYPE "dynamic-shape-op-register"

using namespace mlir;

void mlir::registerDynamicMaskStitchReifyReturnTypeShapes() {
  static ReifyReturnTypeShapesRegistration shapeRegister(
      getDynamicMaskStitchName(),
      [](Operation *op, OpBuilder &builder, ValueRange operands,
         SmallVectorImpl<::mlir::Value> &reifiedReturnShapes) {
        unsigned numOperands = op->getNumOperands();

        Value dim0 = builder.create<tensor::DimOp>(
            op->getLoc(), op->getOperand(numOperands - 1), 0);
        SmallVector<Value> dims;
        dims.push_back(dim0);
        for (int64_t i = 1;
             i < cast<RankedTensorType>(op->getOperand(0).getType()).getRank();
             ++i) {
          dims.push_back(builder.create<tensor::DimOp>(op->getLoc(),
                                                       op->getOperand(0), i));
        }
        reifiedReturnShapes.push_back(
            builder.create<tensor::FromElementsOp>(op->getLoc(), dims));

        return success();
      });
}

void mlir::registerDynamicStitchReifyReturnTypeShapes() {
  static ReifyReturnTypeShapesRegistration shapeRegister(
      getDynamicStitchName(),
      [](Operation *op, OpBuilder &builder, ValueRange operands,
         SmallVectorImpl<::mlir::Value> &reifiedReturnShapes) {
        unsigned numOperands = op->getNumOperands();
        unsigned halfNum = numOperands / 2;
        SmallVector<Value> data;
        data.reserve(halfNum);
        for (unsigned i = 0; i < halfNum; ++i) {
          data.push_back(op->getOperand(i));
        }

        bool allRankedTensor = llvm::all_of(
            data, [](Value v) { return isa<RankedTensorType>(v.getType()); });
        if (!allRankedTensor)
          return failure();

        Value dim0 = builder.create<tensor::DimOp>(op->getLoc(), data[0], 0);
        for (unsigned i = 1; i < halfNum; ++i) {
          Value ithDim0 =
              builder.create<tensor::DimOp>(op->getLoc(), data[i], 0)
                  .getResult();
          dim0 = builder.create<shape::AddOp>(op->getLoc(), dim0, ithDim0);
        }
        SmallVector<Value> dims;
        dims.push_back(dim0);
        for (int64_t i = 1;
             i < cast<RankedTensorType>(data[0].getType()).getRank(); ++i) {
          dims.push_back(
              builder.create<tensor::DimOp>(op->getLoc(), data[0], i));
        }
        reifiedReturnShapes.push_back(
            builder.create<tensor::FromElementsOp>(op->getLoc(), dims));

        return success();
      });
}