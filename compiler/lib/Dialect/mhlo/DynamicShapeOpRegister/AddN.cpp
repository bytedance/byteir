//===- AddN.cpp -----------------------------------------------*--- C++ -*-===//
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
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "dynamic-shape-op-register"

using namespace mlir;

void mlir::registerAddNInferReturnTypeComponents() {
  static InferReturnTypeComponentsRegistration shapeRegister(
      getAddNName(),
      [](MLIRContext *context, std::optional<Location> loc,
         ValueShapeRange operands, DictionaryAttr attr,
         OpaqueProperties properties, RegionRange,
         SmallVectorImpl<ShapedTypeComponents> &inferredReturnTypes) {
        ShapedType dataType = dyn_cast<ShapedType>(operands[0].getType());
        if (!dataType) {
          LLVM_DEBUG(llvm::dbgs() << loc << ": get dataType failed\n");
          return failure();
        }
        auto dataShape = dataType.getShape();
        for (auto operand : operands) {
          ShapedType type = dyn_cast<ShapedType>(operand.getType());
          if (!dataType)
            return failure();
          auto shape = type.getShape();
          if (dataShape.size() != shape.size())
            return failure();
          if (!std::equal(dataShape.begin(), dataShape.end(), shape.begin())) {
            return failure();
          }
        }
        inferredReturnTypes.emplace_back(dataShape, dataType.getElementType());
        return success();
      });
}
