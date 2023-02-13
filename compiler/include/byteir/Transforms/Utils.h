//===- Utils.h ------------------------------------------------*--- C++ -*-===//
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

#ifndef BYTEIR_TRANSFORMS_UTILS_H
#define BYTEIR_TRANSFORMS_UTILS_H

#include "llvm/ADT/StringRef.h"
#include <functional>

// This file includes all RewritePattern-form of utils.
// It is similiar to ones in byteir/Utils but in RewritePattern.
namespace mlir {
class DominanceInfo;
class Operation;
class PostDominanceInfo;
class RewritePatternSet;

void populateHoistUpInBlockPatterns(
    RewritePatternSet &patterns, DominanceInfo &domInfo,
    const std::function<bool(Operation *)> &checkFunc);

void populateHoistDownInBlockPatterns(
    RewritePatternSet &patterns, PostDominanceInfo &postDomInfo,
    const std::function<bool(Operation *)> &checkFunc);

void populateRemoveAttrPatterns(RewritePatternSet &patterns,
                                llvm::StringRef attrName);

} // namespace mlir

#endif // BYTEIR_TRANSFORMS_UTILS_H
