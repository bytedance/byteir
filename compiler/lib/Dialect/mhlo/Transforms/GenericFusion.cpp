//===- GenericFusion.cpp --------------------------------------*--- C++ -*-===//
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
#include "byteir/Dialect/mhlo/Util/CustomCallUtil.h"
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
using namespace mlir::mhlo;

namespace {

bool isCustomMhloRngOp(Operation *op) {
  if (auto customOp = llvm::dyn_cast_or_null<mhlo::CustomCallOp>(op)) {
    return (customOp.getCallTargetName() == getRngUniformName()) ||
           (customOp.getCallTargetName() == getRngNormalName());
  }
  return false;
}

bool isCustomMhloByteirRepeatOp(Operation *op) {
  if (auto customOp = llvm::dyn_cast_or_null<mhlo::CustomCallOp>(op)) {
    return customOp.getCallTargetName() == getRepeatName();
  }
  return false;
}

//===----------------------------------------------------------------------===//
// ElementwiseFusion
//===----------------------------------------------------------------------===//
namespace elementwise {

// TODO: maybe we should support non-splat constant on device in future
bool isFusibleCandidate(Operation *op) {
  return isMhlo(op) &&
         (op->hasTrait<::mlir::OpTrait::Elementwise>() ||
          op->hasTrait<hlo::OpTrait::BroadcastingElementwise>() ||
          isSplatMhloConstantLike(op) ||
          isa<mhlo::BroadcastInDimOp, mhlo::BroadcastOp, mhlo::ReshapeOp>(op) ||
          isCustomMhloRngOp(op));
}

// every candidate can start
bool isFusibleStart(Operation *op) { return true; }

bool isFusibleTrigger(Operation *op) {
  if (op->hasTrait<::mlir::OpTrait::Elementwise>() ||
      op->hasTrait<hlo::OpTrait::BroadcastingElementwise>() ||
      isa<mhlo::ReshapeOp>(op) || isCustomMhloRngOp(op)) {
    return true;
  }

  // if broadcast, check whether its operand is only used in broadcast
  if (isa<mhlo::BroadcastInDimOp, mhlo::BroadcastOp>(op)) {
    auto src = op->getOperand(0);
    // is foldable we just allow
    if (isDeepMhloFoldable(src.getDefiningOp())) {
      return true;
    }
    // otherwise, check it is only used in broadcast
    // return useCount(src) == 1;
    // LWC FIXME: change back to above after broadcast fusion resolve.
    return false;
  }

  return false;
}

bool isFusibleWith(Operation *target, Operation * /*start*/) {
  return target->hasTrait<::mlir::OpTrait::Elementwise>() ||
         target->hasTrait<hlo::OpTrait::BroadcastingElementwise>() ||
         isSplatMhloConstantLike(target) ||
         isa<mhlo::BroadcastInDimOp, mhlo::BroadcastOp, mhlo::ReshapeOp>(
             target) ||
         isCustomMhloRngOp(target);
}

bool isFusibleWithNoElementwiseFuse(Operation *target, Operation * /*start*/) {
  return isSplatMhloConstantLike(target) ||
         isa<mhlo::BroadcastInDimOp, mhlo::BroadcastOp, mhlo::ReshapeOp>(
             target);
}

bool isValidSingleOp(Operation *op) {
  return op->hasTrait<::mlir::OpTrait::Elementwise>() ||
         op->hasTrait<hlo::OpTrait::BroadcastingElementwise>() ||
         isa<mhlo::BroadcastInDimOp, mhlo::BroadcastOp, mhlo::IotaOp>(op) ||
         isCustomMhloRngOp(op);
}

bool isValidFusionPattern(const MhloFusionPattern &) { return true; }

static GenericFuserConfig config{
    getByteIRElementwiseFusionAttrName(), elementwise::isFusibleCandidate,
    elementwise::isFusibleStart,          elementwise::isFusibleTrigger,
    elementwise::isFusibleWith,           elementwise::isValidSingleOp,
    elementwise::isValidFusionPattern};

static GenericFuserConfig config_no_elementwise_fuse{
    getByteIRElementwiseFusionAttrName(),
    elementwise::isFusibleCandidate,
    elementwise::isFusibleStart,
    elementwise::isFusibleTrigger,
    elementwise::isFusibleWithNoElementwiseFuse,
    elementwise::isValidSingleOp,
    elementwise::isValidFusionPattern};

namespace concat_slice {
// TODO: fuse elementwise with concat
bool isFusibleCandidate(Operation *op) {
  return llvm::isa<mhlo::ConcatenateOp, mhlo::ReshapeOp, mhlo::SliceOp>(op);
}

bool isFusibleStart(Operation *op) {
  return llvm::isa<mhlo::ConcatenateOp, mhlo::ReshapeOp, mhlo::SliceOp>(op);
}

bool isFusibleTrigger(Operation *op) {
  return llvm::isa<mhlo::ConcatenateOp, mhlo::ReshapeOp, mhlo::SliceOp>(op);
}

bool isFusibleWith(Operation *target, Operation *start) {
  if (llvm::isa<mhlo::ConcatenateOp, mhlo::ReshapeOp, mhlo::SliceOp>(start)) {
    return llvm::isa<mhlo::SliceOp, mhlo::ReshapeOp>(target);
  }
  return false;
}

bool isValidSingleOp(Operation *op) {
  return llvm::isa<mhlo::ConcatenateOp>(op);
}

bool isValidFusionPattern(const MhloFusionPattern &pattern) {
  if (!llvm::any_of(pattern, [](Operation *op) {
        return llvm::isa<mhlo::ConcatenateOp>(op);
      })) {
    return false;
  }

  llvm::DenseSet<Operation *> opSet(pattern.begin(), pattern.end());
  for (auto op : pattern) {
    if (llvm::isa<mhlo::ConcatenateOp>(op)) {
      bool isLegal = true;
      for (auto operand : op->getOperands()) {
        if (auto defOp = operand.getDefiningOp()) {
          while (defOp) {
            if (opSet.find(defOp) == opSet.end()) {
              break;
            }
            if (!llvm::isa<mhlo::ReshapeOp, mhlo::SliceOp>(defOp)) {
              isLegal = false;
              break;
            } else {
              defOp = defOp->getOperand(0).getDefiningOp();
            }
          }
        }
        if (!isLegal) {
          return false;
        }
      }
    }
  }
  return true;
};
static GenericFuserConfig config_concat_slice_fuse{
    getByteIRElementwiseFusionAttrName(),
    elementwise::concat_slice::isFusibleCandidate,
    elementwise::concat_slice::isFusibleStart,
    elementwise::concat_slice::isFusibleTrigger,
    elementwise::concat_slice::isFusibleWith,
    elementwise::concat_slice::isValidSingleOp,
    elementwise::concat_slice::isValidFusionPattern};
} // namespace concat_slice
} // namespace elementwise

//===----------------------------------------------------------------------===//
// MatmulEpilogueFusion
//===----------------------------------------------------------------------===//

namespace matmul_epilogue {

bool isFusibleCandidate(Operation *op) {
  return isMhlo(op) && (op->hasTrait<::mlir::OpTrait::Elementwise>() ||
                        op->hasTrait<hlo::OpTrait::BroadcastingElementwise>() ||
                        isMhloConstantLike(op) ||
                        isa<mhlo::BroadcastInDimOp, mhlo::BroadcastOp,
                            mhlo::ReshapeOp, mhlo::DotOp>(op));
}

bool isFusibleStart(Operation *op) { return isa<mhlo::DotOp>(op); }

bool isFusibleTrigger(Operation *op) {
  // trigger fuse for anything but dot
  return !isa<mhlo::DotOp>(op);
}

bool isFusibleWith(Operation * /*target*/, Operation * /*start*/) {
  return true;
}

bool isValidSingleOp(Operation *op) { return false; }

bool isValidFusionPattern(const MhloFusionPattern &) { return true; }

static GenericFuserConfig config{getByteIRMatmulEpilogueFusionAttrName(),
                                 matmul_epilogue::isFusibleCandidate,
                                 matmul_epilogue::isFusibleStart,
                                 matmul_epilogue::isFusibleTrigger,
                                 matmul_epilogue::isFusibleWith,
                                 matmul_epilogue::isValidSingleOp,
                                 matmul_epilogue::isValidFusionPattern};

} // namespace matmul_epilogue

//===----------------------------------------------------------------------===//
// ReductionFusion
//===----------------------------------------------------------------------===//

namespace reduction {
// TODO: maybe we should support non-splat constant on device in future
bool isFusibleCandidate(Operation *op) {
  return isMhlo(op) && (op->hasTrait<::mlir::OpTrait::Elementwise>() ||
                        op->hasTrait<hlo::OpTrait::BroadcastingElementwise>() ||
                        isSplatMhloConstantLike(op) ||
                        isa<mhlo::BroadcastInDimOp, mhlo::BroadcastOp,
                            mhlo::ReshapeOp, mhlo::ReduceOp>(op));
}

// every candidate can start
bool isFusibleStart(Operation *op) { return true; }

bool isFusibleTrigger(Operation *op) {
  if (op->hasTrait<::mlir::OpTrait::Elementwise>() ||
      op->hasTrait<hlo::OpTrait::BroadcastingElementwise>() ||
      isa<mhlo::ReshapeOp>(op)) {
    return true;
  }

  // if broadcast, check whether its operand is only used in broadcast
  if (isa<mhlo::BroadcastInDimOp, mhlo::BroadcastOp>(op)) {
    auto src = op->getOperand(0);
    // is foldable we just allow
    if (isDeepMhloFoldable(src.getDefiningOp())) {
      return true;
    }
    // otherwise, check it is only used in broadcast
    // return useCount(src) == 1;
    // LWC FIXME: change back to above after broadcast fusion resolve.
    return false;
  }

  if (isa<mhlo::ReduceOp>(op))
    return true;

  return false;
}

bool isFusibleWith(Operation *target, Operation * /*start*/) {
  return (target->hasTrait<::mlir::OpTrait::Elementwise>() ||
          target->hasTrait<hlo::OpTrait::BroadcastingElementwise>() ||
          isSplatMhloConstantLike(target) ||
          isa<mhlo::BroadcastInDimOp, mhlo::BroadcastOp, mhlo::ReshapeOp>(
              target)) &&
         target->hasOneUse();
}

bool isValidSingleOp(Operation *op) { return isa<mhlo::ReduceOp>(op); }

bool isValidFusionPattern(const MhloFusionPattern &pattern) {
  SmallVector<Value, 4> outputs = getOutputsOfCluster(pattern);
  if (outputs.size() == 1) {
    if (auto reduceOp = outputs[0].getDefiningOp<mhlo::ReduceOp>()) {
      ValueRange inputs = reduceOp.getInputs();
      auto reduceDims = reduceOp.getDimensionsAttr();
      for (Value in : inputs) {
        auto inputShape = cast<ShapedType>(in.getType()).getShape();
        for (auto iter = reduceDims.begin(); iter != reduceDims.end(); iter++) {
          APInt reDim = *iter;
          if (inputShape[reDim.getSExtValue()] != 1) {
            return true;
          }
        }
      }
      return false;
    }
  }
  return false;
}

static GenericFuserConfig config{
    getByteIRReductionFusionAttrName(), reduction::isFusibleCandidate,
    reduction::isFusibleStart,          reduction::isFusibleTrigger,
    reduction::isFusibleWith,           reduction::isValidSingleOp,
    reduction::isValidFusionPattern};

} // namespace reduction

//===----------------------------------------------------------------------===//
// AggressiveFusion
//===----------------------------------------------------------------------===//

namespace aggressive_fusion {

bool isFusibleCandidate(Operation *op) {
  if (isCustomMhloRngOp(op) || isCustomMhloByteirRepeatOp(op))
    return true;
  return isMhlo(op) && !llvm::isa<mhlo::CustomCallOp>(op);
}

bool isFusibleStart(Operation *) { return true; }

bool isFusibleTrigger(Operation *) { return true; }

bool isFusibleWith(Operation *, Operation *) { return true; }

bool isFusibleWithNoDenseFuse(Operation *target, Operation * /*start*/) {
  return isSplatMhloConstantLike(target) ||
         isa<mhlo::BroadcastInDimOp, mhlo::BroadcastOp, mhlo::ReshapeOp>(
             target);
}

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

static GenericFuserConfig no_fuse_config{
    getByteIRHloAggressiveFusionAttrName(),
    aggressive_fusion::isFusibleCandidate,
    aggressive_fusion::isFusibleStart,
    aggressive_fusion::isFusibleTrigger,
    aggressive_fusion::isFusibleWithNoDenseFuse,
    aggressive_fusion::isValidSingleOp,
    aggressive_fusion::isValidFusionPattern};

} // namespace aggressive_fusion

//===----------------------------------------------------------------------===//
// CatFusion
//===----------------------------------------------------------------------===//

namespace cat_fusion {

bool isFusibleCandidate(Operation *op) {
  if (isa<cat::CatOpInterface>(op))
    return true;
  return false;
}

bool isFusibleStart(Operation *op) { return true; }

bool isFusibleTrigger(Operation *op) { return false; }

bool isFusibleWith(Operation *target, Operation * /*start*/) { return false; }

bool isValidSingleOp(Operation *op) { return true; }

bool isValidFusionPattern(const MhloFusionPattern &) { return true; }

static GenericFuserConfig config{
    getByteIRCatFusionAttrName(),    cat_fusion::isFusibleCandidate,
    cat_fusion::isFusibleStart,      cat_fusion::isFusibleTrigger,
    cat_fusion::isFusibleWith,       cat_fusion::isValidSingleOp,
    cat_fusion::isValidFusionPattern};

} // namespace cat_fusion

//===----------------------------------------------------------------------===//
// Fusion Passes
//===----------------------------------------------------------------------===//

// a derived fusion pass for elementwise
struct ElementwiseFusionPass : public GenericFusionPass<ElementwiseFusionPass> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ElementwiseFusionPass)

  ElementwiseFusionPass(bool clusterSingleOp, bool disableElementwiseFusion)
      : GenericFusionPass(clusterSingleOp) {
    this->disableElementwiseFusion = disableElementwiseFusion;
  }

  ElementwiseFusionPass(const ElementwiseFusionPass &other)
      : GenericFusionPass<ElementwiseFusionPass>(other) {}

  /// Returns the command-line argument attached to this pass.
  static constexpr ::llvm::StringLiteral getArgumentName() {
    return ::llvm::StringLiteral("fuse-element");
  }
  ::llvm::StringRef getArgument() const override { return "fuse-element"; }

  ::llvm::StringRef getDescription() const override {
    return "Fuse elementwise op";
  }

  /// Returns the derived pass name.
  static constexpr ::llvm::StringLiteral getPassName() {
    return ::llvm::StringLiteral("ElementFusion");
  }
  ::llvm::StringRef getName() const override { return "ElementFusion"; }

  const GenericFuserConfig &getConfig() {
    return this->disableElementwiseFusion
               ? elementwise::config_no_elementwise_fuse
               : elementwise::config;
  }

  ::mlir::Pass::Option<bool> disableElementwiseFusion{
      *this, "disable-elementwise-fusion",
      ::llvm::cl::desc(
          "disable fusion strategy, only outline single operation"),
      ::llvm::cl::init(false)};
};

struct ConcatSliceFusionPass : public GenericFusionPass<ConcatSliceFusionPass> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConcatSliceFusionPass)

  ConcatSliceFusionPass() : GenericFusionPass(true) {}

  /// Returns the command-line argument attached to this pass.
  static constexpr ::llvm::StringLiteral getArgumentName() {
    return ::llvm::StringLiteral("fuse-concat-slice");
  }
  ::llvm::StringRef getArgument() const override { return "fuse-concat-slice"; }

  ::llvm::StringRef getDescription() const override {
    return "Fuse concat with slice op";
  }

  /// Returns the derived pass name.
  static constexpr ::llvm::StringLiteral getPassName() {
    return ::llvm::StringLiteral("ConcatSliceFusion");
  }
  ::llvm::StringRef getName() const override { return "ConcaSliceFusion"; }

  const GenericFuserConfig &getConfig() {
    return elementwise::concat_slice::config_concat_slice_fuse;
  }
};

// a derived fusion pass for matmul epilogue fusion
struct MatmulEpilogueFusionPass
    : public GenericFusionPass<MatmulEpilogueFusionPass> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MatmulEpilogueFusionPass)

  MatmulEpilogueFusionPass() : GenericFusionPass(false) {}

  /// Returns the command-line argument attached to this pass.
  static constexpr ::llvm::StringLiteral getArgumentName() {
    return ::llvm::StringLiteral("fuse-matmul-epilogue");
  }
  ::llvm::StringRef getArgument() const override {
    return "fuse-matmul-epilogue";
  }

  ::llvm::StringRef getDescription() const override {
    return "Fuse Matmul with elementwise epilogue op";
  }

  /// Returns the derived pass name.
  static constexpr ::llvm::StringLiteral getPassName() {
    return ::llvm::StringLiteral("MatmulEpilogueFusion");
  }
  ::llvm::StringRef getName() const override { return "MatmulEpilogueFusion"; }

  const GenericFuserConfig &getConfig() { return matmul_epilogue::config; }
};

// a derived fusion pass for reduction fusion
struct ReductionFusionPass : public GenericFusionPass<ReductionFusionPass> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReductionFusionPass)

  ReductionFusionPass() : GenericFusionPass(false) {}

  /// Returns the command-line argument attached to this pass.
  static constexpr ::llvm::StringLiteral getArgumentName() {
    return ::llvm::StringLiteral("fuse-reduction");
  }
  ::llvm::StringRef getArgument() const override { return "fuse-reduction"; }

  ::llvm::StringRef getDescription() const override {
    return "Fuse reduction with its producer";
  }

  /// Returns the derived pass name.
  static constexpr ::llvm::StringLiteral getPassName() {
    return ::llvm::StringLiteral("ReductionFusion");
  }
  ::llvm::StringRef getName() const override { return "ReductionFusion"; }

  const GenericFuserConfig &getConfig() { return reduction::config; }
};

// A derived fusion pass for hlo aggressive fusion, which would fuse mhlo ops
// into mhlo.fusion group as much as possible
struct HloAggressiveFusionPass
    : public GenericFusionPass<HloAggressiveFusionPass> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(HloAggressiveFusionPass)

  HloAggressiveFusionPass(bool disableFusion) : GenericFusionPass(true) {
    this->disableFusion = disableFusion;
  }

  HloAggressiveFusionPass(const HloAggressiveFusionPass &other)
      : GenericFusionPass<HloAggressiveFusionPass>(other) {}

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

  const GenericFuserConfig &getConfig() {
    return disableFusion ? aggressive_fusion::no_fuse_config
                         : aggressive_fusion::config;
  }

  ::mlir::Pass::Option<bool> disableFusion{
      *this, "disable-fusion",
      ::llvm::cl::desc(
          "disable fusion strategy, only outline single operation"),
      ::llvm::cl::init(false)};
};

// a derived fusion pass for cat op
struct CatFusionPass : public GenericFusionPass<CatFusionPass> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CatFusionPass)

  CatFusionPass() : GenericFusionPass(true) {}

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

  const GenericFuserConfig &getConfig() { return cat_fusion::config; }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createElementFusionPass(bool clusterSingleElemwiseOp,
                              bool disableElementwiseFuse) {
  return std::make_unique<ElementwiseFusionPass>(clusterSingleElemwiseOp,
                                                 disableElementwiseFuse);
}

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createConcatSliceFusionPass() {
  return std::make_unique<ConcatSliceFusionPass>();
}

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createMatmulEpilogueFusionPass() {
  return std::make_unique<MatmulEpilogueFusionPass>();
}

std::unique_ptr<OperationPass<func::FuncOp>> mlir::createReductionFusionPass() {
  return std::make_unique<ReductionFusionPass>();
}

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createHloAggressiveFusionPass(bool disableFusion) {
  return std::make_unique<HloAggressiveFusionPass>(disableFusion);
}

std::unique_ptr<OperationPass<func::FuncOp>> mlir::createCatFusionPass() {
  return std::make_unique<CatFusionPass>();
}
