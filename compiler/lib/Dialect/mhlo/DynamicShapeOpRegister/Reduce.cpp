//===- Reduce.cpp ---------------------------------------------*--- C++ -*-===//
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
#include "byteir/Dialect/mhlo/Util/ShapeInferUtil.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"

#define DEBUG_TYPE "dynamic-shape-op-register"

using namespace mlir;

// TODO: this should be removed when push to upstream
void mlir::registerReduceInferReturnTypeComponents() {
  static InferReturnTypeComponentsRegistration shapeRegister(
      mhlo::ReduceOp::getOperationName(),
      [](MLIRContext *context, std::optional<Location>,
         ValueShapeRange operands, DictionaryAttr attr, RegionRange,
         SmallVectorImpl<ShapedTypeComponents> &inferredReturnTypes) {
        auto inputType = dyn_cast<RankedTensorType>(operands[0].getType());
        if (inputType == nullptr || !inputType.hasStaticShape()) {
          return failure();
        }
        auto dimensions =
            dyn_cast<DenseIntElementsAttr>(attr.get("dimensions"));
        if (dimensions == nullptr) {
          return failure();
        }
        int64_t rank = inputType.getRank();
        llvm::SmallVector<bool, 4> dimsMask(rank, false);
        for (int64_t dim : dimensions.getValues<int64_t>())
          dimsMask[dim] = true;

        SmallVector<int64_t, 4> shape;
        for (int64_t i = 0; i < rank; ++i) {
          if (!dimsMask[i])
            shape.push_back(inputType.getDimSize(i));
        }
        Type type = RankedTensorType::get(shape, IntegerType::get(context, 64));
        inferredReturnTypes.push_back(cast<ShapedType>(type));
        return success();
      });
}
