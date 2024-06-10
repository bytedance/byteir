//===- ScatterNd.cpp ------------------------------------------*--- C++ -*-===//
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

#define DEBUG_TYPE "dynamic-shape-op-register"

using namespace mlir;

/// See ScatterNd's signature on
/// https://www.tensorflow.org/api_docs/python/tf/raw_ops/ScatterNd
void mlir::registerScatterNdInferBoundedReturnTypeComponents() {
  static InferBoundedReturnTypeComponentsRegistration shapeRegister(
      getScatterNdName(),
      [](MLIRContext *context, std::optional<Location>,
         ValueShapeRange operands, DictionaryAttr, RegionRange,
         SmallVectorImpl<ShapedTypeComponents> &inferredReturnTypes) {
        auto adaptor = operands.getValueAsShape(2);
        if (!adaptor || !adaptor.hasStaticShape())
          return failure();

        llvm::SmallVector<int64_t> dims;
        adaptor.getDims(dims);
        Type type = RankedTensorType::get(dims, IntegerType::get(context, 64));
        inferredReturnTypes.push_back(cast<ShapedType>(type));
        return success();
      });
}
