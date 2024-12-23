//===- StridedSlice.cpp ---------------------------------------*--- C++ -*-===//
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
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"

#define DEBUG_TYPE "dynamic-shape-op-register"

using namespace mlir;

llvm::SmallVector<int64_t> stridedSliceShapeInfer(
    llvm::ArrayRef<int64_t> inputShapeRef, int64_t beginMask, int64_t endMask,
    int64_t newAxisMask, int64_t shrinkAxisMask,
    mlir::DenseIntElementsAttr beginAttr, mlir::DenseIntElementsAttr endAttr,
    mlir::DenseIntElementsAttr strideAttr) {

  llvm::SmallVector<int> newAxis;
  auto newAxisMaskCopy = newAxisMask;
  int index = 0;
  while (newAxisMaskCopy > 0) {
    if (newAxisMaskCopy & 1) {
      newAxis.push_back(index);
    }
    index++;
    newAxisMaskCopy = newAxisMaskCopy >> 1;
  }

  int rank = newAxis.size() + inputShapeRef.size();
  llvm::SmallVector<int64_t> inputShape(rank);
  int newAxisIndex = 0;
  int inputShapeIndex = 0;
  for (int i = 0; i < rank; ++i) {
    if (newAxisIndex < newAxis.size() &&
        (inputShapeIndex + newAxisIndex) == newAxis[newAxisIndex]) {
      inputShape[i] = 1;
      newAxisIndex++;
    } else {
      inputShape[i] = inputShapeRef[inputShapeIndex];
      inputShapeIndex++;
    }
  }

  llvm::SmallVector<int> beginValue(beginAttr.getValues<int>().begin(),
                                    beginAttr.getValues<int>().end());
  llvm::SmallVector<int> endValue(endAttr.getValues<int>().begin(),
                                  endAttr.getValues<int>().end());
  llvm::SmallVector<int> strideValue(strideAttr.getValues<int>().begin(),
                                     strideAttr.getValues<int>().end());

  assert(beginValue.size() == endValue.size());
  assert(beginValue.size() == strideValue.size());
  assert(beginValue.size() <= inputShape.size());

  llvm::SmallVector<int64_t> outputShape;
  for (size_t i = 0; i < beginValue.size(); ++i) {
    if (((1 << i) & newAxisMask) != 0) {
      assert(inputShape[i] == 1);
      outputShape.push_back(inputShape[i]);
      continue;
    }
    int64_t from = beginValue[i];
    int64_t to = endValue[i];
    int64_t step = strideValue[i];
    if (from < 0) {
      from += inputShape[i];
    }
    if (to < 0) {
      to += inputShape[i];
    }
    assert(step != 0);
    if (((1 << i) & beginMask) != 0) {
      from = (step > 0) ? 0 : inputShape[i];
    }
    if (((1 << i) & endMask) != 0) {
      to = (step > 0) ? inputShape[i] : 0;
    }
    int64_t range = std::abs(to - from);
    step = std::abs(step);
    int64_t len = (range - 1) / step + 1;
    if (((1 << i) & shrinkAxisMask) == 0) {
      outputShape.push_back(len);
    } else {
      assert(len == 1);
    }
  }
  for (size_t i = beginValue.size(); i < inputShape.size(); ++i) {
    outputShape.push_back(inputShape[i]);
  }
  return outputShape;
}

LogicalResult stridedSliceShapeInferReturnType(
    MLIRContext *context, ValueShapeRange operands, DictionaryAttr attr,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnTypes) {
  Value input = operands[0];
  Value begin = operands[1];
  Value end = operands[2];
  Value stride = operands[3];

  ShapedType inputShapeType = dyn_cast<ShapedType>(input.getType());
  if (!inputShapeType || !inputShapeType.hasStaticShape()) {
    llvm::outs() << "input shape of tf.StridedSlice not static"
                 << "\n";
    return failure();
  }

  int64_t beginMask = attr.getAs<DictionaryAttr>(getCustomCallAttrName())
                          .getAs<IntegerAttr>("begin_mask")
                          .getInt();
  int64_t endMask = attr.getAs<DictionaryAttr>(getCustomCallAttrName())
                        .getAs<IntegerAttr>("end_mask")
                        .getInt();
  int64_t ellipsisMask = attr.getAs<DictionaryAttr>(getCustomCallAttrName())
                             .getAs<IntegerAttr>("ellipsis_mask")
                             .getInt();
  int64_t newAxisMask = attr.getAs<DictionaryAttr>(getCustomCallAttrName())
                            .getAs<IntegerAttr>("new_axis_mask")
                            .getInt();
  int64_t shrinkAxisMask = attr.getAs<DictionaryAttr>(getCustomCallAttrName())
                               .getAs<IntegerAttr>("shrink_axis_mask")
                               .getInt();

  // TODO: support ellipsis_mask
  if (ellipsisMask != 0) {
    llvm::outs() << "ellipsis mask not equal to 0"
                 << "\n";
    return failure();
  }

  mlir::DenseIntElementsAttr beginAttr;
  mlir::DenseIntElementsAttr endAttr;
  mlir::DenseIntElementsAttr strideAttr;
  if (!matchPattern(begin, m_Constant(&beginAttr))) {
    llvm::outs() << "begin of tf.StridedSlice not const value"
                 << "\n";
    return failure();
  }
  if (!matchPattern(end, m_Constant(&endAttr))) {
    llvm::outs() << "end  of tf.StridedSlice not const value"
                 << "\n";
    return failure();
  }
  if (!matchPattern(stride, m_Constant(&strideAttr))) {
    llvm::outs() << "stride of tf.StridedSlice not const value"
                 << "\n";
    return failure();
  }
  auto inputShapeRef = inputShapeType.getShape();
  auto outputShape =
      stridedSliceShapeInfer(inputShapeRef, beginMask, endMask, newAxisMask,
                             shrinkAxisMask, beginAttr, endAttr, strideAttr);

  Type type = RankedTensorType::get(outputShape, IntegerType::get(context, 64));
  inferredReturnTypes.push_back(cast<ShapedType>(type));
  return success();
}

void mlir::registerStridedSliceReifyReturnTypeShapes() {
  static ReifyReturnTypeShapesRegistration shapeRegister(
      getStridedSliceName(),
      [](Operation *op, OpBuilder &builder, ValueRange operands,
         SmallVectorImpl<::mlir::Value> &reifiedReturnShapes) {
        Value input = op->getOperand(0);
        Value begin = op->getOperand(1);
        Value end = op->getOperand(2);
        Value stride = op->getOperand(3);

        ShapedType inputShapeType = dyn_cast<ShapedType>(input.getType());
        if (!inputShapeType) {
          return failure();
        }

        DictionaryAttr attr = op->getAttrDictionary();
        int64_t beginMask = attr.getAs<DictionaryAttr>(getCustomCallAttrName())
                                .getAs<IntegerAttr>("begin_mask")
                                .getInt();
        int64_t endMask = attr.getAs<DictionaryAttr>(getCustomCallAttrName())
                              .getAs<IntegerAttr>("end_mask")
                              .getInt();
        int64_t ellipsisMask =
            attr.getAs<DictionaryAttr>(getCustomCallAttrName())
                .getAs<IntegerAttr>("ellipsis_mask")
                .getInt();
        int64_t newAxisMask =
            attr.getAs<DictionaryAttr>(getCustomCallAttrName())
                .getAs<IntegerAttr>("new_axis_mask")
                .getInt();
        int64_t shrinkAxisMask =
            attr.getAs<DictionaryAttr>(getCustomCallAttrName())
                .getAs<IntegerAttr>("shrink_axis_mask")
                .getInt();

        // TODO: support ellipsis_mask
        if (ellipsisMask != 0) {
          llvm::outs() << "ellipsis mask not equal to 0"
                       << "\n";
          return failure();
        }

        mlir::DenseIntElementsAttr beginAttr;
        mlir::DenseIntElementsAttr endAttr;
        mlir::DenseIntElementsAttr strideAttr;
        if (!matchPattern(begin, m_Constant(&beginAttr))) {
          llvm::outs() << "begin of tf.StridedSlice not const value"
                       << "\n";
          return failure();
        }
        if (!matchPattern(end, m_Constant(&endAttr))) {
          llvm::outs() << "end  of tf.StridedSlice not const value"
                       << "\n";
          return failure();
        }
        if (!matchPattern(stride, m_Constant(&strideAttr))) {
          llvm::outs() << "stride of tf.StridedSlice not const value"
                       << "\n";
          return failure();
        }
        llvm::SmallVector<int> newAxis;
        auto newAxisMaskCopy = newAxisMask;
        int index = 0;
        while (newAxisMaskCopy > 0) {
          if (newAxisMaskCopy & 1) {
            newAxis.push_back(index);
          }
          index++;
          newAxisMaskCopy = newAxisMaskCopy >> 1;
        }

        Value zeroV = builder.create<arith::ConstantIndexOp>(op->getLoc(), 0);
        Value oneV = builder.create<arith::ConstantIndexOp>(op->getLoc(), 1);
        Value negV = builder.create<arith::ConstantIndexOp>(op->getLoc(), -1);
        int rank = newAxis.size() + inputShapeType.getRank();
        llvm::SmallVector<Value> inputShape(rank);
        int newAxisIndex = 0;
        int inputShapeIndex = 0;
        for (int i = 0; i < rank; ++i) {
          if (newAxisIndex < newAxis.size() &&
              (inputShapeIndex + newAxisIndex) == newAxis[newAxisIndex]) {
            inputShape[i] = oneV;
            newAxisIndex++;
          } else {
            inputShape[i] = builder.create<tensor::DimOp>(op->getLoc(), input,
                                                          inputShapeIndex);
            inputShapeIndex++;
          }
        }

        llvm::SmallVector<int> beginValue(beginAttr.getValues<int>().begin(),
                                          beginAttr.getValues<int>().end());
        llvm::SmallVector<int> endValue(endAttr.getValues<int>().begin(),
                                        endAttr.getValues<int>().end());
        llvm::SmallVector<int> strideValue(strideAttr.getValues<int>().begin(),
                                           strideAttr.getValues<int>().end());

        assert(beginValue.size() == endValue.size());
        assert(beginValue.size() == strideValue.size());
        assert(beginValue.size() <= inputShape.size());

        llvm::SmallVector<Value> outputShape;
        for (size_t i = 0; i < beginValue.size(); ++i) {
          if (((1 << i) & newAxisMask) != 0) {
            assert(inputShape[i] == oneV);
            outputShape.push_back(inputShape[i]);
            continue;
          }
          if (((1 << i) & shrinkAxisMask) != 0) {
            continue;
          }

          int64_t from = beginValue[i];
          int64_t to = endValue[i];
          int64_t step = strideValue[i];
          Value fromV =
              builder.create<shape::GetExtentOp>(op->getLoc(), begin, i);
          Value toV = builder.create<shape::GetExtentOp>(op->getLoc(), end, i);
          Value stepV = builder.create<arith::ConstantIndexOp>(op->getLoc(),
                                                               std::abs(step));
          if (from < 0) {
            fromV = builder.create<shape::AddOp>(op->getLoc(), fromV,
                                                 inputShape[i]);
          }
          if (to < 0) {
            toV =
                builder.create<shape::AddOp>(op->getLoc(), toV, inputShape[i]);
          }
          assert(step != 0);
          if (((1 << i) & beginMask) != 0) {
            if (step > 0) {
              fromV = zeroV;
            } else {
              fromV = inputShape[i];
            }
          }
          if (((1 << i) & endMask) != 0) {
            if (step > 0) {
              toV = inputShape[i];
            } else {
              toV = zeroV;
            }
          }
          Value rangeV;
          if (step > 0) {
            Value negFromV =
                builder.create<shape::MulOp>(op->getLoc(), fromV, negV);
            rangeV = builder.create<shape::AddOp>(op->getLoc(), toV, negFromV);
          } else {
            Value negToV =
                builder.create<shape::MulOp>(op->getLoc(), toV, negV);
            rangeV = builder.create<shape::AddOp>(op->getLoc(), fromV, negToV);
          }
          Value lenV = builder.create<shape::AddOp>(op->getLoc(), rangeV, negV);
          lenV = builder.create<shape::DivOp>(op->getLoc(), lenV, stepV);
          lenV = builder.create<shape::AddOp>(op->getLoc(), lenV, oneV);
          outputShape.push_back(lenV);
        }
        for (size_t i = beginValue.size(); i < inputShape.size(); ++i) {
          outputShape.push_back(inputShape[i]);
        }

        reifiedReturnShapes.push_back(
            builder.create<tensor::FromElementsOp>(op->getLoc(), outputShape));

        return success();
      });
}

void mlir::registerStridedSliceInferReturnTypeComponents() {
  static InferReturnTypeComponentsRegistration shapeRegister(
      getStridedSliceName(),
      [](MLIRContext *context, std::optional<Location>,
         ValueShapeRange operands, DictionaryAttr attr, OpaqueProperties,
         RegionRange,
         SmallVectorImpl<ShapedTypeComponents> &inferredReturnTypes) {
        return stridedSliceShapeInferReturnType(context, operands, attr,
                                                inferredReturnTypes);
      });
}

/// See StridedSlice's signature on
/// https://www.tensorflow.org/api_docs/python/tf/raw_ops/StridedSlice
void mlir::registerStridedSliceInferBoundedReturnTypeComponents() {
  static InferBoundedReturnTypeComponentsRegistration shapeRegister(
      getStridedSliceName(),
      [](MLIRContext *context, std::optional<Location>,
         ValueShapeRange operands, DictionaryAttr attr, RegionRange,
         SmallVectorImpl<ShapedTypeComponents> &inferredReturnTypes) {
        return stridedSliceShapeInferReturnType(context, operands, attr,
                                                inferredReturnTypes);
      });
}
