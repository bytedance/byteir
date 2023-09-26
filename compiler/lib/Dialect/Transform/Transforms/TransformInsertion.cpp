//===- TransformInsertion.cpp ---------------------------------*--- C++ -*-===//
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

#include "byteir/Dialect/Transform/Transforms/TransformInsertion.h"

#include "byteir/Dialect/Linalg/TransformOps/LinalgExtTransformOps.h"
#include "byteir/Dialect/Transform/IR/TransformExtOps.h"
#include "byteir/Utils/IRRewrite.h"
#include "byteir/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/TransformOps/DialectExtension.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include <optional>
#include <vector>

#include "PassDetail.h"

using namespace mlir;
using namespace llvm;

namespace {
inline std::string getAnnotationUniqueIdentifier(const std::string &prefix) {
  static size_t cnt = 0;
  return prefix + "_" + std::to_string(cnt++);
}

// TODO maybe move to public
void insertTransformIR(func::FuncOp funcOp, OpBuilder &builder,
                       const TransformInsertionConfig &config) {
  funcOp->walk([&](Operation *op) {
    if (config.opFilter(op)) {
      ImplicitLocOpBuilder b(op->getLoc(), builder);
      MLIRContext *ctx = b.getContext();

      auto annotation = getAnnotationUniqueIdentifier(config.matchPrefix);
      op->setAttr(annotation, UnitAttr::get(ctx));

      auto pdlOperationType = pdl::OperationType::get(ctx);
      b.create<transform::SequenceOp>(
          TypeRange(), transform::FailurePropagationMode::Propagate,
          pdlOperationType, [&](OpBuilder &b, Location loc, Value blockArg) {
            auto annotationAttr = DictionaryAttr::get(
                ctx, b.getNamedAttr(annotation, UnitAttr::get(ctx)));
            auto match = b.create<transform::MatchOp>(
                loc, blockArg.getType(), blockArg, ArrayAttr(),
                transform::MatchInterfaceEnumAttr(), annotationAttr,
                TypeAttr());
            ImplicitLocOpBuilder ib(loc, b);
            config.transformBuilder(ib, op, match);
            b.create<transform::YieldOp>(loc);
          });
    }
  });
}

void insertTransformIR(ModuleOp m, const TransformInsertionConfig &config) {
  OpBuilder builder = OpBuilder::atBlockEnd(m.getBody());
  for (auto funcOp : m.getOps<func::FuncOp>()) {
    // only tile on device functions
    if (!config.funcAnchor.empty() && !funcOp->hasAttr(config.funcAnchor)) {
      continue;
    }

    insertTransformIR(funcOp, builder, config);
  }
}

struct DetensorizeTransformInsertionPass
    : public DetensorizeTransformInsertionBase<
          DetensorizeTransformInsertionPass> {
  explicit DetensorizeTransformInsertionPass(const std::string &funcAnchor,
                                             const std::string &matchPrefix)
      : DetensorizeTransformInsertionBase() {
    this->funcAnchorAttr = funcAnchor;
    this->matchPrefix = matchPrefix;
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<transform::TransformDialect>();
    linalg::registerTransformDialectExtension(registry);
  }

  static bool isScalarTensorOp(linalg::LinalgOp linalgOp) {
    if (!linalgOp.hasTensorSemantics())
      return false;

    if (linalgOp.getNumLoops() != 0)
      return false;

    auto isScalarOrScalarTensorOperand = [&](OpOperand &operand) {
      if (linalgOp.isScalar(&operand))
        return true;

      auto tensorType =
          llvm::dyn_cast<RankedTensorType>(operand.get().getType());
      if (!tensorType)
        return false;

      return tensorType.getRank() == 0;
    };
    return llvm::all_of(linalgOp->getOpOperands(),
                        isScalarOrScalarTensorOperand);
  }

  void runOnOperation() override {
    auto opFilter = [](Operation *op) {
      if (auto linalgOp = llvm::dyn_cast_or_null<linalg::LinalgOp>(op)) {
        return isScalarTensorOp(linalgOp);
      }
      return false;
    };

    auto transformBuilder = [](ImplicitLocOpBuilder &b, Operation *,
                               Value pdlValue) {
      b.create<transform::DetensorizeOp>(pdlValue);
    };

    insertTransformIR(getOperation(), {funcAnchorAttr, matchPrefix, opFilter,
                                       transformBuilder});
  }
};

struct FuseExtTransformInsertionPass
    : public FuseExtTransformInsertionBase<FuseExtTransformInsertionPass> {
  explicit FuseExtTransformInsertionPass(
      const std::string &funcAnchor, const std::string &matchPrefix,
      const std::string &tileSizeAttrName,
      const std::string &tileInterchangeAttrName, const bool keepIntermediates)
      : FuseExtTransformInsertionBase() {
    this->funcAnchorAttr = funcAnchor;
    this->matchPrefix = matchPrefix;
    this->tileSizeAttrName = tileSizeAttrName;
    this->tileInterchangeAttrName = tileInterchangeAttrName;
    this->keepIntermediates = keepIntermediates;
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<transform::TransformDialect>();
    linalg::registerTransformDialectExtension(registry);
  }

  void runOnOperation() override {
    auto opFilter = [](Operation *op) {
      if (auto funcOp = op->getParentOfType<func::FuncOp>()) {
        Operation *retOp = funcOp.getBody().front().getTerminator();
        if (retOp->getOperand(0).getDefiningOp() == op)
          return true;
      }
      return false;
    };

    auto transformBuilder = [&](ImplicitLocOpBuilder &b, Operation *op,
                                Value pdlValue) {
      SmallVector<int64_t, 4> tileSizes = {};
      SmallVector<int64_t, 4> tileInterchange = {};
      if (auto tileSizesAttr = op->getAttrOfType<ArrayAttr>(tileSizeAttrName)) {
        tileSizes = extractFromIntegerArrayAttr<int64_t>(tileSizesAttr);
      }
      if (auto tileInterchangeAttr =
              op->getAttrOfType<ArrayAttr>(tileInterchangeAttrName)) {
        tileInterchange =
            extractFromIntegerArrayAttr<int64_t>(tileInterchangeAttr);
      }

      if (tileSizes.empty())
        tileSizes = {1};

      auto pdlType = pdl::OperationType::get(b.getContext());
      SmallVector<Type> resultTypes(1 + tileSizes.size(), pdlType);

      b.create<transform::FuseExtOp>(
          resultTypes, pdlValue, nullptr, b.getI64ArrayAttr(tileSizes),
          b.getI64ArrayAttr(tileInterchange), b.getBoolAttr(keepIntermediates));
      b.create<transform_ext::CleanupOp>(std::nullopt, std::nullopt);
    };

    insertTransformIR(getOperation(), {funcAnchorAttr, matchPrefix, opFilter,
                                       transformBuilder});
  }
};

struct GenericTransformInsertionPass
    : public PassWrapper<GenericTransformInsertionPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GenericTransformInsertionPass)

  GenericTransformInsertionPass(const TransformInsertionConfig &config)
      : config(config) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<transform::TransformDialect>();
    linalg::registerTransformDialectExtension(registry);
  }

  void runOnOperation() override { insertTransformIR(getOperation(), config); }

protected:
  TransformInsertionConfig config;
};

struct RewriteInDPSTransformInsertionPass
    : public RewriteInDPSTransformInsertionBase<
          RewriteInDPSTransformInsertionPass> {
  explicit RewriteInDPSTransformInsertionPass(const std::string &funcAnchor,
                                              const std::string &matchPrefix)
      : RewriteInDPSTransformInsertionBase() {
    this->funcAnchorAttr = funcAnchor;
    this->matchPrefix = matchPrefix;
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<transform::TransformDialect>();
    linalg::registerTransformDialectExtension(registry);
  }

  void runOnOperation() override {
    auto opFilter = [](Operation *op) {
      return llvm::isa<tensor::FromElementsOp>(op);
    };

    auto transformBuilder = [](ImplicitLocOpBuilder &b, Operation *,
                               Value pdlValue) {
      b.create<transform::RewriteInDestinationPassingStyleOp>(
          pdlValue.getType(), pdlValue);
    };

    insertTransformIR(getOperation(), {funcAnchorAttr, matchPrefix, opFilter,
                                       transformBuilder});
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createDetensorizeTransformInsertionPass(const std::string &funcAnchor,
                                              const std::string &matchPrefix) {
  return std::make_unique<DetensorizeTransformInsertionPass>(funcAnchor,
                                                             matchPrefix);
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createFuseExtTransformInsertionPass(
    const std::string &funcAnchor, const std::string &matchPrefix,
    const std::string &tileSizeAttrName,
    const std::string &tileInterchangeAttrName, const bool keepIntermediates) {
  return std::make_unique<FuseExtTransformInsertionPass>(
      funcAnchor, matchPrefix, tileSizeAttrName, tileInterchangeAttrName,
      keepIntermediates);
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createGenericTransformInsertionPass(
    const TransformInsertionConfig &config) {
  return std::make_unique<GenericTransformInsertionPass>(config);
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createRewriteInDPSTransformInsertionPass(const std::string &funcAnchor,
                                               const std::string &matchPrefix) {
  return std::make_unique<RewriteInDPSTransformInsertionPass>(funcAnchor,
                                                              matchPrefix);
}