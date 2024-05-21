//===- CollectMhloOps.cpp -------------------------------------*--- C++-* -===//
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

#include "byteir/Dialect/mhlo/Transforms/HloFuser.h"

#include "byteir/Dialect/mhlo/Transforms/GenericFusionCommon.h"
#include "byteir/Dialect/mhlo/Util/CustomCallUtil.h"
#include "byteir/Dialect/mhlo/Util/FusionUtil.h"
#include "byteir/Dialect/mhlo/Util/Util.h"
#include "byteir/Utils/IRRewrite.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"

#include "PassDetail.h"

using namespace llvm;
using namespace mlir;
using namespace mlir::mhlo;

namespace {
namespace aggressive_fusion {

bool isCustomMhloRngUniformOp(Operation *op) {
  if (auto customOp = llvm::dyn_cast_or_null<mhlo::CustomCallOp>(op)) {
    return customOp.getCallTargetName() == getRngUniformName();
  }
  return false;
}

bool isCustomMhloByteirRepeatOp(Operation *op) {
  if (auto customOp = llvm::dyn_cast_or_null<mhlo::CustomCallOp>(op)) {
    return customOp.getCallTargetName() == getRepeatName();
  }
  return false;
}

bool isFusibleCandidate(Operation *op) {
  if (isCustomMhloRngUniformOp(op) || isCustomMhloByteirRepeatOp(op))
    return true;
  return isMhlo(op) && !llvm::isa<mhlo::CustomCallOp>(op);
}

bool isFusibleStart(Operation *) { return true; }

bool isFusibleTrigger(Operation *) { return true; }

bool isFusibleWith(Operation *, Operation *) { return true; }

bool isValidSingleOp(Operation *op) {
  if (llvm::isa<mhlo::ReshapeOp>(op))
    return false;
  else
    return true;
}

bool isValidFusionPattern(const MhloFusionPattern &) { return true; }

static GenericFuserConfig config{getByteIRHloAggressiveFusionAttrName(),
                                 aggressive_fusion::isFusibleCandidate,
                                 aggressive_fusion::isFusibleStart,
                                 aggressive_fusion::isFusibleTrigger,
                                 aggressive_fusion::isFusibleWith,
                                 aggressive_fusion::isValidSingleOp,
                                 aggressive_fusion::isValidFusionPattern};

} // namespace aggressive_fusion

// A derived fusion pass for hlo aggressive fusion, which would fuse mhlo ops
// into mhlo.fusion group as much as possible
struct HloAggressiveFusionPass
    : public GenericFusionPass<HloAggressiveFusionPass> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(HloAggressiveFusionPass)

  HloAggressiveFusionPass() : GenericFusionPass(true) {}

  /// Returns the command-line argument attached to this pass.
  static constexpr ::llvm::StringLiteral getArgumentName() {
    return ::llvm::StringLiteral("hlo-aggressive-fusion");
  }
  ::llvm::StringRef getArgument() const override {
    return "hlo-aggressive-fusion";
  }

  ::llvm::StringRef getDescription() const override {
    return "Do aggressive fusion on mhlo dialect, fuse mhlo ops into "
           "mhlo.fusion group as much as possible.";
  }

  /// Returns the derived pass name.
  static constexpr ::llvm::StringLiteral getPassName() {
    return ::llvm::StringLiteral("HloAggressiveFusion");
  }
  ::llvm::StringRef getName() const override { return "HloAggressiveFusion"; }

  const GenericFuserConfig &getConfig() { return aggressive_fusion::config; }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createHloAggressiveFusionPass() {
  return std::make_unique<HloAggressiveFusionPass>();
}
