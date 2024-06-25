//===- OFInsertNecessaryCast.cpp
//-------------------------------------------===//
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

#include "onnx-frontend/src/Conversion/OFInsertNecessaryCast.hpp"

#include "third_party/onnx-mlir/src/Dialect/ONNX/DialectBuilder.hpp"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {
//===----------------------------------------------------------------------===//
// Concat
//===----------------------------------------------------------------------===//
// This pattern insert cast op to force convert incompatible operand.
// clang-format off
// from:
// %val = onnx.Op(%val, %axes) : tensor<?xTy0> -> tensor<?xTy0>
// onnx.Concat(..., %val, ...) -> tensor<?xTy1>
// to:
// %val = onnx.Op(%val, %axes) : tensor<?xTy0> -> tensor<?xTy0>
// %cast = onnx.Cast(%val) : tensor<?xTy0> -> tensor<?xTy1>
// onnx.Concat(..., %cast, ...) -> tensor<?xTy1>
// clang-format on
struct CheckConcatOp : public OpRewritePattern<mlir::ONNXConcatOp> {
  using OpRewritePattern<mlir::ONNXConcatOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::ONNXConcatOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto out = op.getConcatResult();
    auto outType = out.getType().cast<ShapedType>();
    auto outElementTy = outType.getElementType();

    auto _check = [&outElementTy](Value val) {
      return val.getType().cast<ShapedType>().getElementType() == outElementTy;
    };

    onnx_mlir::OnnxBuilder onnxBuilder(rewriter, loc);
    rewriter.setInsertionPoint(op);

    bool modify = false;
    SmallVector<Value, 4> inputs;
    for (auto inp : op.getInputs()) {
      if (_check(inp)) {
        inputs.push_back(inp);
        continue;
      }
      // insert cast
      auto castTy = RankedTensorType::get(
          inp.getType().cast<ShapedType>().getShape(), outElementTy);
      auto cast = onnxBuilder.cast(inp, outElementTy);
      inputs.push_back(cast);
      modify |= true;
    }

    if (!modify)
      return failure();

    auto concatOp = onnxBuilder.concat(outType, inputs, op.getAxis());
    rewriter.replaceOp(op, concatOp);
    return success();
  }
};

struct OFInsertNecessaryCastPass
    : public onnx_frontend::OFInsertNecessaryCastBase<
          OFInsertNecessaryCastPass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OFInsertNecessaryCastPass);

  OFInsertNecessaryCastPass() = default;

  void runOnOperation() final {
    func::FuncOp func = getOperation();
    MLIRContext *context = &getContext();

    RewritePatternSet patterns(context);

    patterns.insert<CheckConcatOp>(context);

    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

namespace onnx_frontend {
std::unique_ptr<mlir::Pass> createOFInsertNecessaryCastPass() {
  return std::make_unique<OFInsertNecessaryCastPass>();
}
} // namespace onnx_frontend
