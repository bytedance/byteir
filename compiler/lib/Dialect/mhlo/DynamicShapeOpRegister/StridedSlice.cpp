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

        ShapedType inputShapeType = input.getType().dyn_cast<ShapedType>();
        if (!inputShapeType || !inputShapeType.hasStaticShape()) {
          llvm::outs() << "input shape of tf.StridedSlice not static"
                       << "\n";
          return failure();
        }
        mlir::DenseIntElementsAttr beginAttr;
        if (!matchPattern(begin, m_Constant(&beginAttr))) {
          llvm::outs() << "begin of tf.StridedSlice not const value"
                       << "\n";
          return failure();
        }
        mlir::DenseIntElementsAttr endAttr;
        if (!matchPattern(end, m_Constant(&endAttr))) {
          llvm::outs() << "end  of tf.StridedSlice not const value"
                       << "\n";
          return failure();
        }
        mlir::DenseIntElementsAttr strideAttr;
        if (!matchPattern(stride, m_Constant(&strideAttr))) {
          llvm::outs() << "stride of tf.StridedSlice not const value"
                       << "\n";
          return failure();
        }

        auto inputShape = inputShapeType.getShape();
        llvm::SmallVector<int> beginValue(beginAttr.getValues<int>().begin(),
                                          beginAttr.getValues<int>().end());
        llvm::SmallVector<int> endValue(endAttr.getValues<int>().begin(),
                                        endAttr.getValues<int>().end());
        llvm::SmallVector<int> strideValue(strideAttr.getValues<int>().begin(),
                                           strideAttr.getValues<int>().end());

        assert(inputShape.size() == beginValue.size());
        assert(inputShape.size() == endValue.size());
        assert(inputShape.size() == strideValue.size());

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
        // TODO: support ellipsis_mask and new_axis_mask
        assert(ellipsisMask == 0);
        assert(newAxisMask == 0);

        llvm::SmallVector<int64_t> outputShape;
        for (size_t i = 0; i < inputShape.size(); ++i) {
          int64_t from = beginValue[i];
          int64_t to = endValue[i];
          int64_t step = strideValue[i];
          // TODO: support negative
          assert(from >= 0);
          assert(to >= 0);
          assert(step >= 0);
          if (((1 << i) & beginMask) == 1) {
            from = 0;
          }
          if (((1 << i) & endMask) == 1) {
            to = inputShape[i];
          }
          if (((1 << i) & shrinkAxisMask) == 0) {
            outputShape.push_back((to - from) / step);
          }
        }
        Type type =
            RankedTensorType::get(outputShape, inputShapeType.getElementType());
        inferredReturnTypes.push_back(type.cast<ShapedType>());
        return success();
      });
}
