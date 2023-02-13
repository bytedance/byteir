//===- GenericFusionCommon.h ---------------------------------*--- C++ -*-===//
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

#ifndef BYTEIR_DIALECT_MHLO_TRANSFORMS_GENERICFUSIONCOMMON_H
#define BYTEIR_DIALECT_MHLO_TRANSFORMS_GENERICFUSIONCOMMON_H

#include "byteir/Dialect/mhlo/Util/FusionUtil.h"
#include "byteir/Dialect/mhlo/Util/Util.h"
#include "byteir/Utils/IRRewrite.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include <functional>
#include <memory>
#include <string>

namespace mlir {
class Operation;
namespace func {
class FuncOp;
} // namespace func

struct GenericFuserConfig {
  StringRef fuse_attr;
  std::function<bool(Operation *)> fuse_candidate;
  std::function<bool(Operation *)> fuse_start;
  std::function<bool(Operation *)> fuse_trigger;
  std::function<bool(Operation *, Operation *)> fuse_with;
  std::function<bool(Operation *)> valid_single_op;
};

//===----------------------------------------------------------------------===//
// GenericFusion template
//===----------------------------------------------------------------------===//

template <typename DerivedT>
class GenericFusionBase : public ::mlir::OperationPass<mlir::func::FuncOp> {
public:
  using Base = GenericFusionBase;

  GenericFusionBase()
      : ::mlir::OperationPass<mlir::func::FuncOp>(
            ::mlir::TypeID::get<DerivedT>()) {}

  GenericFusionBase(const GenericFusionBase &other)
      : ::mlir::OperationPass<mlir::func::FuncOp>(other) {}

  /// Support isa/dyn_cast functionality for the derived pass class.
  static bool classof(const ::mlir::Pass *pass) {
    return pass->getTypeID() == ::mlir::TypeID::get<DerivedT>();
  }

  /// A clone method to create a copy of this pass.
  std::unique_ptr<::mlir::Pass> clonePass() const override {
    return std::make_unique<DerivedT>(*static_cast<const DerivedT *>(this));
  }

  // Note please add the following member func for derived pass

  // static constexpr ::llvm::StringLiteral getArgumentName()
  // ::llvm::StringRef getArgument() const override
  // ::llvm::StringRef getDescription() const override
  // static constexpr ::llvm::StringLiteral getPassName()
  // ::llvm::StringRef getName() const override
  // getDependentDialects(::mlir::DialectRegistry &registry)

public:
  ::mlir::Pass::Option<bool> clusterSingleOp{
      *this, "cluster-single-op",
      ::llvm::cl::desc(
          "whether to cluster single operation into mhlo.fusion op"),
      ::llvm::cl::init(false)};
};

template <typename DerivedT>
class GenericFusionPass : public GenericFusionBase<DerivedT> {
public:
  GenericFusionPass(GenericFuserConfig config, bool clusterSingleOp)
      : GenericFusionBase<DerivedT>() {
    fuse_config = config;
    this->clusterSingleOp = clusterSingleOp;
  }

  void runOnOperation() override {
    func::FuncOp funcOp = this->getOperation();
    // skip private
    if (funcOp.isPrivate())
      return;

    for (auto &block : funcOp.getBlocks()) {
      replicateDefiningOp(&block, isMhloConstantLike);
    }

    ProducerFusionPlanner planner(
        funcOp, fuse_config.fuse_candidate, fuse_config.fuse_start,
        fuse_config.fuse_trigger, fuse_config.fuse_with);
    planner.run();

    const MhloFusionPlan &plan = planner.getFusionPlan();

    for (auto it = plan.rbegin(); it != plan.rend(); ++it) {
      auto &pattern = *it;
      if (pattern.size() > 1) {
        applyMhloFusionPattern(pattern, fuse_config.fuse_attr);
      } else if (this->clusterSingleOp.getValue()) {
        if (pattern.size() == 1 && fuse_config.valid_single_op(pattern[0])) {
          applyMhloFusionPattern(pattern, fuse_config.fuse_attr);
        }
      }
    }
  }

protected:
  // member variable
  GenericFuserConfig fuse_config;
};

} // namespace mlir

#endif // BYTEIR_DIALECT_MHLO_TRANSFORMS_GENERICFUSIONCOMMON_H
