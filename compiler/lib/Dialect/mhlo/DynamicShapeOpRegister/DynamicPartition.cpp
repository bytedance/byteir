//===- DynamicPartition.cpp -----------------------------------*--- C++ -*-===//
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

#include "byteir/Dialect/Shape/IR/ShapeExtOps.h"
#include "byteir/Dialect/mhlo/DynamicShapeOpRegister/Register.h"
#include "byteir/Dialect/mhlo/Util/CustomCallUtil.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"

#define DEBUG_TYPE "dynamic-shape-op-register"

using namespace mlir;

void mlir::registerDynamicPartitionShapeConstraints() {
  static InsertShapeConstraintRegistration shapeRegister(
      getDynamicPartitionName(), [](Operation *op, OpBuilder &builder) {
        // init builder position
        builder.setInsertionPointAfter(op);
        SmallVector<Value> dim0OfResults;
        for (Value res : op->getResults()) {
          dim0OfResults.push_back(
              builder.create<tensor::DimOp>(op->getLoc(), res, 0));
        }
        Value sum = dim0OfResults[0];
        for (size_t i = 1; i < dim0OfResults.size(); ++i) {
          sum =
              builder.create<shape::AddOp>(op->getLoc(), sum, dim0OfResults[i]);
        }
        Value dim0OfOperand =
            builder.create<tensor::DimOp>(op->getLoc(), op->getOperand(0), 0);
        builder.create<shape_ext::MeetOp>(op->getLoc(), sum, dim0OfOperand);
        return success();
      });
}

/// See DynamicPartition's signature on
/// https://www.tensorflow.org/api_docs/python/tf/dynamic_partition
void mlir::registerDynamicPartitionInferBoundedReturnTypeComponents() {
  static InferBoundedReturnTypeComponentsRegistration shapeRegister(
      getDynamicPartitionName(),
      [](MLIRContext *context, std::optional<Location>,
         ValueShapeRange operands, DictionaryAttr attr, RegionRange,
         SmallVectorImpl<ShapedTypeComponents> &inferredReturnTypes) {
        auto numPartition = attr.getAs<DictionaryAttr>(getCustomCallAttrName())
                                .getAs<IntegerAttr>("num_partitions")
                                .getInt();
        if (ShapedType shapedType =
                operands[0].getType().dyn_cast_or_null<ShapedType>()) {
          inferredReturnTypes.append(numPartition, shapedType);
          return success();
        }
        return failure();
      });
}