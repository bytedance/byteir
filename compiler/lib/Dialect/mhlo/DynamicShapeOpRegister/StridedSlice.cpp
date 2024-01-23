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
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"

#define DEBUG_TYPE "dynamic-shape-op-register"

using namespace mlir;

/// See StridedSlice's signature on
/// https://www.tensorflow.org/api_docs/python/tf/raw_ops/StridedSlice
void mlir::registerStridedSliceInferBoundedReturnTypeComponents() {
  static InferBoundedReturnTypeComponentsRegistration shapeRegister(
      getStridedSliceName(),
      [](MLIRContext *context, std::optional<Location>,
         ValueShapeRange operands, DictionaryAttr attr, RegionRange,
         SmallVectorImpl<ShapedTypeComponents> &inferredReturnTypes) {
        Value input = operands[0];
        Value begin = operands[1];
        Value end = operands[2];
        Value stride = operands[3];

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
        assert(ellipsisMask == 0);

        ShapedType inputShapeType = input.getType().dyn_cast<ShapedType>();
        if (!inputShapeType || !inputShapeType.hasStaticShape()) {
          llvm::outs() << "input shape of tf.StridedSlice not static"
                       << "\n";
          return failure();
        }

        auto inputShapeRef = inputShapeType.getShape();
        llvm::SmallVector<int64_t> inputShape;
        if (newAxisMask != 0) {
          // insert 1 to inputshape according to newAxisMask
          llvm::SmallVector<int> newAxis;
          auto newAxisMaskCopy = newAxisMask;
          int index = 0;
          while (newAxisMaskCopy > 0) {
            if (newAxisMaskCopy & 1) {
              newAxis.push_back(index);
            }
            newAxisMaskCopy = newAxisMaskCopy >> 1;
            index++;
          }
          int rank = newAxis.size() + inputShapeRef.size();
          inputShape.resize(rank, 0);
          int newAxisIndex = 0;
          int inputShapeIndex = 0;
          for (int i = 0; i < rank; ++i) {
            if (newAxisIndex < newAxis.size()) {
              if ((inputShapeIndex + newAxisIndex) < newAxis[newAxisIndex]) {
                inputShape[i] = inputShapeRef[inputShapeIndex];
                inputShapeIndex++;
              } else {
                inputShape[i] = 1;
                newAxisIndex++;
              }
            } else {
              inputShape[i] = inputShapeRef[inputShapeIndex];
              inputShapeIndex++;
            }
          }
        } else {
          std::copy(inputShapeRef.begin(), inputShapeRef.end(),
                    std::back_inserter(inputShape));
        }

        mlir::DenseIntElementsAttr beginAttr;
        if (!matchPattern(begin, m_Constant(&beginAttr))) {
          // TODO: support non const begin
          Type type =
              RankedTensorType::get(inputShape, IntegerType::get(context, 64));
          inferredReturnTypes.push_back(type.cast<ShapedType>());
          llvm::outs() << "begin of tf.StridedSlice not const value"
                       << "\n";
          return success();
        }
        mlir::DenseIntElementsAttr endAttr;
        if (!matchPattern(end, m_Constant(&endAttr))) {
          // TODO: support non const end
          Type type =
              RankedTensorType::get(inputShape, IntegerType::get(context, 64));
          inferredReturnTypes.push_back(type.cast<ShapedType>());
          llvm::outs() << "end  of tf.StridedSlice not const value"
                       << "\n";
          return success();
        }
        mlir::DenseIntElementsAttr strideAttr;
        if (!matchPattern(stride, m_Constant(&strideAttr))) {
          // TODO: support non const stride
          Type type =
              RankedTensorType::get(inputShape, IntegerType::get(context, 64));
          inferredReturnTypes.push_back(type.cast<ShapedType>());
          llvm::outs() << "stride of tf.StridedSlice not const value"
                       << "\n";
          return success();
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
        Type type =
            RankedTensorType::get(outputShape, IntegerType::get(context, 64));
        inferredReturnTypes.push_back(type.cast<ShapedType>());
        return success();
      });
}
