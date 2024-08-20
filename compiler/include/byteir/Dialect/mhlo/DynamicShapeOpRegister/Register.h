//===- Register.h ---------------------------------------------*--- C++ -*-===//
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

#ifndef BYTEIR_DIALECT_MHLO_DYNAMICSHAPEOPREGISTER_REGISTER_H
#define BYTEIR_DIALECT_MHLO_DYNAMICSHAPEOPREGISTER_REGISTER_H

#include "byteir/Dialect/mhlo/Util/ShapeInferUtil.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
//===----------------------------------------------------------------------===//
// StaticShapeInfer Registration
//===----------------------------------------------------------------------===//
void registerConvolutionInferReturnTypeComponents();
void registerDotInferReturnTypeComponents();
void registerDotGeneralInferReturnTypeComponents();
void registerDynamicBroadcastInDimInferReturnTypeComponents();
void registerDynamicReshapeInferReturnTypeComponents();
void registerReshapeInferReturnTypeComponents();
void registerRealDynamicSliceInferReturnTypeComponents();
void registerReduceInferReturnTypeComponents();
void registerSoftmaxInferReturnTypeComponents();
void registerAddNInferReturnTypeComponents();
void registerOneHotInferReturnTypeComponents();
void registerTorchIndexSelectInferReturnTypeComponents();
void registerGeLUInferReturnTypeComponents();
void registerLayerNormInferReturnTypeComponents();
void registerBatchMatMulInferReturnTypeComponents();

inline void registerAllMhloInferReturnTypeComponents() {
  registerConvolutionInferReturnTypeComponents();
  registerDotInferReturnTypeComponents();
  registerDotGeneralInferReturnTypeComponents();
  registerDynamicBroadcastInDimInferReturnTypeComponents();
  registerDynamicReshapeInferReturnTypeComponents();
  registerReshapeInferReturnTypeComponents();
  registerRealDynamicSliceInferReturnTypeComponents();
  registerReduceInferReturnTypeComponents();
  registerSoftmaxInferReturnTypeComponents();
  registerAddNInferReturnTypeComponents();
  registerOneHotInferReturnTypeComponents();
  registerTorchIndexSelectInferReturnTypeComponents();
  registerGeLUInferReturnTypeComponents();
  registerLayerNormInferReturnTypeComponents();
  registerBatchMatMulInferReturnTypeComponents();
}

//===----------------------------------------------------------------------===//
// BoundedShapeInfer Registration
//===----------------------------------------------------------------------===//
void registerDynamicPartitionInferBoundedReturnTypeComponents();
void registerNonZeroInferBoundedReturnTypeComponents();
void registerScatterNdInferBoundedReturnTypeComponents();
void registerStridedSliceInferBoundedReturnTypeComponents();
void registerRepeatInferBoundedReturnTypeComponents();

inline void registerAllMhloInferBoundedReturnTypeComponents() {
  registerDynamicPartitionInferBoundedReturnTypeComponents();
  registerNonZeroInferBoundedReturnTypeComponents();
  registerScatterNdInferBoundedReturnTypeComponents();
  registerStridedSliceInferBoundedReturnTypeComponents();
  registerRepeatInferBoundedReturnTypeComponents();
}

//===----------------------------------------------------------------------===//
// ShapeReification Registration
//===----------------------------------------------------------------------===//
void registerDotReifyReturnTypeShapes();
void registerDynamicStitchReifyReturnTypeShapes();
void registerDynamicMaskStitchReifyReturnTypeShapes();
void registerDynamicBroadcastInDimReifyReturnTypeShapes();
void registerSoftmaxReifyReturnTypeShapes();
void registerTorchIndexSelectReifyReturnTypeShapes();
void registerGeLUReifyReturnTypeShapes();

inline void registerAllMhloReifyReturnTypeShapes() {
  registerDotReifyReturnTypeShapes();
  registerDynamicStitchReifyReturnTypeShapes();
  registerDynamicMaskStitchReifyReturnTypeShapes();
  registerDynamicBroadcastInDimReifyReturnTypeShapes();
  registerSoftmaxReifyReturnTypeShapes();
  registerTorchIndexSelectReifyReturnTypeShapes();
  registerGeLUReifyReturnTypeShapes();
}

//===----------------------------------------------------------------------===//
// ShapeConstraint Registration
//===----------------------------------------------------------------------===//
void registerConcatenateShapeConstraints();
void registerDotGeneralShapeConstraints();
void registerDynamicPartitionShapeConstraints();
void registerDynamicReshapeShapeConstraints();
void registerEinsumShapeConstraints();
void registerReshapeShapeConstraints();

inline void registerAllMhloShapeConstraints() {
  registerConcatenateShapeConstraints();
  registerDotGeneralShapeConstraints();
  registerDynamicPartitionShapeConstraints();
  registerDynamicReshapeShapeConstraints();
  registerEinsumShapeConstraints();
  registerReshapeShapeConstraints();
}

} // namespace mlir

#endif // BYTEIR_DIALECT_MHLO_DYNAMICSHAPEOPREGISTER_REGISTER_H
