//===- RemoveSingleIterationLoop.h --------------------------- C++ --===//
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

#ifndef BYTEIR_DIALECT_SCF_TRANSFORMS_REMOVESINGLEITERATIONLOOP_H
#define BYTEIR_DIALECT_SCF_TRANSFORMS_REMOVESINGLEITERATIONLOOP_H

namespace mlir {
using GetMinMaxExprFn =
    std::function<std::optional<std::pair<AffineExpr, AffineExpr>>(
        Value value, SmallVectorImpl<Value> &dims,
        SmallVectorImpl<Value> &symbols)>;

/// Insert pattern to remove single iteration loop. The pattern will detect
/// single iteration loops based on the range returned by the lambda
/// |getMinMaxFn| for some know values.
void populateRemoveSingleIterationLoopPattern(RewritePatternSet &patterns,
                                              GetMinMaxExprFn getMinMaxFn);

} // namespace mlir
#endif // BYTEIR_DIALECT_SCF_TRANSFORMS_INSERTTRIVIALSCFLOOP_H