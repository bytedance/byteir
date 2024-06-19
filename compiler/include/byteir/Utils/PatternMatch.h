//===- PatternMatch.h ------------------------------------ -*- C++ ------*-===//
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
#pragma once

#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
bool registerPDLConstraintFunction(MLIRContext *ctx, StringRef name,
                                   PDLConstraintFunction constraintFn,
                                   bool override);

template <typename ConstraintFnT>
bool registerPDLConstraintFunction(MLIRContext *ctx, StringRef name,
                                   ConstraintFnT &&constraintFn,
                                   bool override) {
  return registerPDLConstraintFunction(
      ctx, name,
      detail::pdl_function_builder::buildConstraintFn(
          std::forward<ConstraintFnT>(constraintFn)),
      override);
}

bool registerPDLRewriteFunction(MLIRContext *ctx, StringRef name,
                                PDLRewriteFunction rewriteFn, bool override);

template <typename RewriteFnT>
bool registerPDLRewriteFunction(MLIRContext *ctx, StringRef name,
                                RewriteFnT &&rewriteFn, bool override) {
  return registerPDLRewriteFunction(
      ctx, name,
      detail::pdl_function_builder::buildRewriteFn(
          std::forward<RewriteFnT>(rewriteFn)),
      override);
}

void applyPDLPatternHooks(PDLPatternModule &pdlPattern);

void registerPDLPatternHooksInterface(DialectRegistry &registry);
} // namespace mlir
