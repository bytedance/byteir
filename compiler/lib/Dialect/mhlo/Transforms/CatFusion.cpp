//===- CatFusion.cpp ------------------------------------------*--- C++ -*-===//
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

#include "byteir/Dialect/Cat/IR/CatDialect.h"
#include "byteir/Dialect/mhlo/Transforms/GenericFusionCommon.h"
#include "byteir/Dialect/mhlo/Util/FusionUtil.h"
#include "byteir/Dialect/mhlo/Util/Util.h"
#include "byteir/Utils/IRRewrite.h"
#include "byteir/Utils/Utils.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"

#include "PassDetail.h"

using namespace llvm;
using namespace mlir;
using namespace mlir::cat;

namespace {

bool matchPermute(mhlo::TransposeOp op, ArrayRef<int64_t> target_perm) {
  auto perm = op.getPermutation().getValues<int64_t>();
  if (perm.size() != target_perm.size())
    return false;
  for (size_t i = 0; i < target_perm.size(); ++i)
    if (perm[i] != target_perm[i])
      return false;
  return true;
}

namespace cat_fusion {

bool isFusibleCandidate(Operation *op) {
  if (isa<cat::CatOpInterface>(op))
    return true;
  if (isa<mhlo::TransposeOp>(op)) {
    auto transOp = cast<mhlo::TransposeOp>(*op);
    if (!matchPermute(transOp, {0, 2, 3, 1}) &&
        !matchPermute(transOp, {0, 3, 1, 2}) && !matchPermute(transOp, {1, 0}))
      // BRT support TBD, offload to AIT
      return true;
  }
  return false;
}

bool isFusibleStart(Operation *op) { return true; }

bool isFusibleTrigger(Operation *op) { return true; }

bool isFusibleWith(Operation *target, Operation * /*start*/) { return true; }

bool isValidSingleOp(Operation *op) { return true; }

bool isValidFusionPattern(const MhloFusionPattern &) { return true; }

bool isFusibleCandidateAggressive(Operation *op) {
  if (isa<cat::CatOpInterface>(op))
    return true;
  if (isa<mhlo::TransposeOp>(op)) {
    auto transOp = cast<mhlo::TransposeOp>(*op);
    if (!matchPermute(transOp, {0, 2, 3, 1}) &&
        !matchPermute(transOp, {0, 3, 1, 2}) && !matchPermute(transOp, {1, 0}))
      // BRT support TBD, offload to AIT
      return true;
  }
  if (isa<mhlo::ReshapeOp>(op))
    return true;
  if (auto constantOp = dyn_cast<mhlo::ConstantOp>(op)) {
    auto elemTy = cast<RankedTensorType>(constantOp.getOutput().getType())
                      .getElementType();
    auto width = elemTy.getIntOrFloatBitWidth();
    // only support int1/8/16/32/64, float32/64
    if (elemTy.isa<IntegerType>())
      return width == 1 || width == 8 || width == 16 || width == 32 ||
             width == 64;
    if (elemTy.isa<FloatType>())
      return width == 32 || width == 64;
    return false;
  }
  return false;
}

bool isValidSingleOpAggressive(Operation *op) {
  return !isa<mhlo::ReshapeOp>(op) && !isa<mhlo::ConstantOp>(op);
}

static GenericFuserConfig config{
    getByteIRCatFusionAttrName(),    cat_fusion::isFusibleCandidate,
    cat_fusion::isFusibleStart,      cat_fusion::isFusibleTrigger,
    cat_fusion::isFusibleWith,       cat_fusion::isValidSingleOp,
    cat_fusion::isValidFusionPattern};

static GenericFuserConfig aggressiveConfig{
    getByteIRCatFusionAttrName(),    cat_fusion::isFusibleCandidateAggressive,
    cat_fusion::isFusibleStart,      cat_fusion::isFusibleTrigger,
    cat_fusion::isFusibleWith,       cat_fusion::isValidSingleOpAggressive,
    cat_fusion::isValidFusionPattern};

} // namespace cat_fusion

// a derived fusion pass for cat op
struct CatFusionPass : public GenericFusionPass<CatFusionPass> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CatFusionPass)

  CatFusionPass(bool aggressiveMode) : GenericFusionPass(true) {
    this->aggressiveMode = aggressiveMode;
  }

  CatFusionPass(const CatFusionPass &other)
      : GenericFusionPass<CatFusionPass>(other) {}

  /// Returns the command-line argument attached to this pass.
  static constexpr ::llvm::StringLiteral getArgumentName() {
    return ::llvm::StringLiteral("cat-fusion");
  }
  ::llvm::StringRef getArgument() const override { return "cat-fusion"; }

  ::llvm::StringRef getDescription() const override {
    return "Fuse cat subgraph";
  }

  /// Returns the derived pass name.
  static constexpr ::llvm::StringLiteral getPassName() {
    return ::llvm::StringLiteral("CatFusion");
  }
  ::llvm::StringRef getName() const override { return "CatFusion"; }

  const GenericFuserConfig &getConfig() {
    return this->aggressiveMode ? cat_fusion::aggressiveConfig
                                : cat_fusion::config;
  }

  ::mlir::Pass::Option<bool> aggressiveMode{
      *this, "aggressive-mode",
      ::llvm::cl::desc("whether to fuse CAT aggressively"),
      ::llvm::cl::init(false)};
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createCatFusionPass(bool aggressiveMode) {
  return std::make_unique<CatFusionPass>(aggressiveMode);
}
