//===- CanonicalizeExt.h --------------------------------------*--- C++ -*-===//
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

#ifndef TORCH_FRONTEND_TRANSFORMS_CANONICALIZE_EXT
#define TORCH_FRONTEND_TRANSFORMS_CANONICALIZE_EXT

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {

/// Creates an instance of the CanonicalizeExt pass, configured with default
/// settings (which can be overridden by pass options on the command line).
std::unique_ptr<Pass> createCanonicalizeExtPass();

/// Creates an instance of the CanonicalizeExt pass with the specified config.
std::unique_ptr<Pass>
createCanonicalizeExtPass(const GreedyRewriteConfig &config,
                          ArrayRef<std::string> disabledPatterns = std::nullopt,
                          ArrayRef<std::string> enabledPatterns = std::nullopt);

} // namespace mlir

#endif // TORCH_FRONTEND_TRANSFORMS_CANONICALIZE_EXT
