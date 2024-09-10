//===- FuseOpOnTorch.cpp --------------------------------------*--- C++ -*-===//
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

#include "torch-frontend/Dialect/Torch/Transforms/FuseOpOnTorch.h"
#include "./PassDetail.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "torch-frontend/Utils/CustomCallUtil.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "llvm/ADT/StringSet.h"

using namespace mlir;
using namespace mlir::torch;

namespace {

Value createGeluTanh(PatternRewriter &rewriter, Location loc, Value output,
                     Value input) {
  Value tanhStr = rewriter.create<Torch::ConstantStrOp>(loc, "tanh");
  return rewriter.create<Torch::AtenGeluOp>(loc, output.getType(), input,
                                            tanhStr);
}

Value createGeluErf(PatternRewriter &rewriter, Location loc, Value output,
                    Value input) {
  Value noneStr = rewriter.create<Torch::ConstantStrOp>(loc, "none");
  return rewriter.create<Torch::AtenGeluOp>(loc, output.getType(), input,
                                            noneStr);
}

Value createLayerNormEpsilon(PatternRewriter &rewriter, Location loc,
                             ElementsAttr epsilon) {
  return rewriter.create<Torch::ConstantFloatOp>(
      loc, epsilon.getSplatValue<FloatAttr>());
}

Value createLayerNorm(PatternRewriter &rewriter, Location loc, Value output,
                      Value input, Value list, Value weight, Value bias,
                      Value epsilon, Value cudnn_enable) {
  Torch::AtenLayerNormOp layerNormOp = rewriter.create<Torch::AtenLayerNormOp>(
      loc, output.getType(), input, list, weight, bias, epsilon, cudnn_enable);
  layerNormOp->setAttr("eps_outside_sqrt", rewriter.getBoolAttr(true));
  return layerNormOp.getResult();
}

Value createL2Norm(PatternRewriter &rewriter, Location loc, Value output,
                   Value input, Value dims, Value eps) {
  auto op = rewriter.create<Torch::OperatorOp>(loc, TypeRange{output.getType()},
                                               "byteir.l2_norm",
                                               ValueRange{input, dims, eps},
                                               /*regionsCount=*/0);
  op->setAttr("eps_outside_sqrt", rewriter.getBoolAttr(true));
  return op->getResults()[0];
}

bool isValueLeastInfoTorchTensor(Value value) {
  if (auto ty = dyn_cast<Torch::NonValueTensorType>(value.getType())) {
    if (!ty.hasSizes() && !ty.hasDtype())
      return true;
  }
  return false;
}

bool isValueFullInfoTorchValueTensor(Value value) {
  if (auto ty = dyn_cast<Torch::ValueTensorType>(value.getType())) {
    if (ty.hasSizes() && ty.hasDtype())
      return true;
  }
  return false;
}

// copied from compiler/lib/Utils/Utils.cpp
bool isSplatValue(DenseIntElementsAttr attr, int64_t value) {
  if (!attr) {
    return false;
  }
  if (!attr.isSplat()) {
    return false;
  }
  return attr.getSplatValue<APInt>() == value;
}

bool isSplatValue(DenseFPElementsAttr attr, double value) {
  if (!attr)
    return false;
  return attr.isSplat() &&
         attr.getSplatValue<FloatAttr>().getValueAsDouble() == value;
}

bool isSplatCloseToValue(DenseFPElementsAttr attr, double value,
                         double EPSILON = 0.00001) {
  if (!attr)
    return false;
  if (!attr.isSplat())
    return false;
  double x = attr.getSplatValue<FloatAttr>().getValueAsDouble() - value;
  if ((x >= -EPSILON) && (x <= EPSILON))
    return true;
  return false;
}

#include "FuseOpOnTorchPattern.inc"

struct FuseOpOnTorchPass : public FuseOpOnTorchBase<FuseOpOnTorchPass> {
  FuseOpOnTorchPass(ArrayRef<std::string> validCustomCallOps) {
    this->validCustomCallOps = validCustomCallOps;
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<Torch::TorchDialect>();
  }

  void runOnOperation() override {
    validCustomCallOpsSet.clear();
    validCustomCallOpsSet.insert(validCustomCallOps.begin(),
                                 validCustomCallOps.end());

    func::FuncOp funcOp = getOperation();
    MLIRContext *context = &getContext();

    RewritePatternSet patterns(context);
    patterns.add<TorchGeluErfPattern>(context);
    patterns.add<TorchGeluTanhPattern>(context);
    patterns.add<TorchLayerNormPattern>(context);
    if (validCustomCallOpsSet.contains("byteir.l2_norm")) {
      patterns.add<TorchL2NormPattern>(context);
      patterns.add<TorchL2NormPattern1>(context);
    }

    LogicalResult result =
        applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
    if (failed(result)) {
      signalPassFailure();
    }
  }

  llvm::StringSet<> validCustomCallOpsSet;
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createFuseOpOnTorch(ArrayRef<std::string> validCustomCallOps) {
  return std::make_unique<FuseOpOnTorchPass>(validCustomCallOps);
}
