//===- fuse_tf_ops.cc -----------------------------------------*--- C++ -*-===//
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

#include "tf_mlir_ext/transforms/fuse_tf_ops.h"
#include "tf_mlir_ext/transforms/passes_detail.h"
#include "tf_mlir_ext/utils/utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/compiler/mlir/lite/utils/validators.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tf2xla/transforms/utils.h"

using namespace mlir;
using namespace mlir::tfext;
using namespace llvm;

namespace {

std::optional<ArrayAttr>
ExtractDilationsAttrFromBlockShape(Value stb_block_shape, Value bts_block_shape,
                                   PatternRewriter &rewriter) {
  DenseIntElementsAttr stb_bs_attr, bts_bs_attr;
  if (!matchPattern(stb_block_shape, m_Constant(&stb_bs_attr)) ||
      !matchPattern(bts_block_shape, m_Constant(&bts_bs_attr))) {
    return {};
  }
  if (stb_bs_attr.size() != bts_bs_attr.size() || stb_bs_attr.size() != 3) {
    return {};
  }
  llvm::SmallVector<int64_t> stb_bs;
  for (int64_t i = 0, end = stb_bs_attr.size(); i < end; i++) {
    if (stb_bs_attr.getValues<APInt>()[i].getSExtValue() !=
        bts_bs_attr.getValues<APInt>()[i].getSExtValue()) {
      return {};
    }
    stb_bs.push_back(stb_bs_attr.getValues<APInt>()[i].getSExtValue());
  }

  return rewriter.getI64ArrayAttr({1, stb_bs[0], stb_bs[1], stb_bs[2], 1});
}

struct FuseDilatedConv3DPattern : public OpRewritePattern<TF::Conv3DOp> {
  using OpRewritePattern<TF::Conv3DOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::Conv3DOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getResult().hasOneUse()) {
      return rewriter.notifyMatchFailure(
          op, "result for current op has more than 1 use");
    }
    // Make sure Conv3D has 'VALID' padding.
    if (op->template getAttrOfType<StringAttr>("padding").getValue() !=
        "VALID") {
      return rewriter.notifyMatchFailure(
          op, "Conv3D op doesn't have valid padding");
    }
    // Make sure dilations are all ones if set.
    const ArrayAttr &dilations =
        op->template getAttrOfType<ArrayAttr>("dilations");
    if (dilations && !TFL::TFIntListIsAllOnes(dilations)) {
      return rewriter.notifyMatchFailure(op, "dilations should be all 1");
    }

    if (!TFL::TFTypeIsFloat32Tensor(op.getInput()) &&
        !TFL::TFTypeIsBFloat16OrHalfTensor(op.getInput())) {
      return rewriter.notifyMatchFailure(
          op, "op's input is not float or half or bfloat16");
    }
    if (!TFL::TFDataFormatIsNDHWC(op)) {
      return rewriter.notifyMatchFailure(op, "op's data format isn't NDHWC");
    }
    // Allow dynamic batch, width and height dimensions only.
    auto result_ty = cast<TensorType>(op.getResult().getType());
    if (!result_ty.hasRank() || result_ty.getRank() != 5 ||
        result_ty.isDynamicDim(4)) {
      return rewriter.notifyMatchFailure(
          op, "only dynamic batch, width and height dimensions are allowed");
    }

    TF::SpaceToBatchNDOp stb_op =
        op.getInput().getDefiningOp<TF::SpaceToBatchNDOp>();
    if (!stb_op || !stb_op.getResult().hasOneUse()) {
      return failure();
    }
    if (!stb_op.getResult().hasOneUse()) {
      return rewriter.notifyMatchFailure(
          stb_op, "stb_op's result doesn't have one use");
    }

    Operation *consumer_op = op.getResult().getUses().begin()->getOwner();
    if (!consumer_op) {
      return rewriter.notifyMatchFailure(op, "op doesn't have consumer node");
    }

    TF::BatchToSpaceNDOp bts_op =
        llvm::dyn_cast<TF::BatchToSpaceNDOp>(consumer_op);
    if (!bts_op) {
      return rewriter.notifyMatchFailure(consumer_op,
                                         "consumer op isn't BatchToSpace Op");
    }

    DenseIntElementsAttr paddings, crops;
    if (!matchPattern(stb_op.getPaddings(), m_Constant(&paddings)) ||
        !matchPattern(bts_op.getCrops(), m_Constant(&crops))) {
      return rewriter.notifyMatchFailure(
          stb_op, "either SpaceToBatch or BatchToSpaceND doesn't have constant "
                  "paddings/crops value");
    }
    if (paddings.getType() != crops.getType()) {
      return rewriter.notifyMatchFailure(
          stb_op,
          "SpaceToBatchND op's padding doesn't have same shape/type with "
          "BatchToSpaceND op's crops");
    }
    std::optional<ArrayAttr> dilations_attr =
        ExtractDilationsAttrFromBlockShape(stb_op.getBlockShape(),
                                           bts_op.getBlockShape(), rewriter);
    if (!dilations_attr.has_value()) {
      return rewriter.notifyMatchFailure(stb_op,
                                         "failed to extract dilation rate");
    }

    // note: the padding setting from
    // tensorflow/tensorflow/mlir/lite/transforms/dilated_conv.h
    // FIXME(lyq): maybe could split to Pad + Conv3d ???
    int64_t m = paddings.getType().getDimSize(0);
    for (uint64_t i = 0; i < m; i++) {
      for (uint64_t j = 0; j < 2; j++) {
        if (paddings.getValues<APInt>()[{i, j}].getSExtValue() !=
            crops.getValues<APInt>()[i, j].getSExtValue()) {
          op->setAttr("padding", rewriter.getStringAttr("SAME"));
          break;
        }
      }
    }

    op->setAttr("dilations", dilations_attr.value());
    op.setOperand(0, stb_op.getInput());
    op.getResult().setType(bts_op.getResult().getType());
    bts_op.getResult().replaceAllUsesWith(bts_op.getInput());
    stb_op.getResult().dropAllUses();
    return success();
  }
};

Value replaceWhereStatic(PatternRewriter &rewriter, Location loc, Value input,
                         Value oneHotOutput) {
  auto inputType = input.getType().dyn_cast<RankedTensorType>();
  auto oneHotOutputType = oneHotOutput.getType().dyn_cast<RankedTensorType>();
  assert(inputType.getRank() >= oneHotOutputType.getRank());
  if (inputType.getRank() > oneHotOutputType.getRank()) {
    auto inputShape = inputType.getShape();
    SmallVector<int64_t> oneHotShape =
        llvm::to_vector(oneHotOutputType.getShape());
    for (int64_t i = 0; i < (inputShape.size() - oneHotShape.size()); ++i) {
      oneHotShape.push_back(1);
    }
    auto shapeType =
        RankedTensorType::get({inputShape.size()}, rewriter.getIntegerType(64));
    auto shapeAttr = DenseIntElementsAttr::get(shapeType, oneHotShape);
    Value shape = rewriter.create<TF::ConstOp>(loc, shapeAttr);
    oneHotOutputType = oneHotOutputType.clone(oneHotShape);
    oneHotOutput = rewriter.create<TF::ReshapeOp>(loc, oneHotOutputType,
                                                  oneHotOutput, shape);
  }

  Value mul = rewriter.create<TF::MulOp>(loc, inputType, input, oneHotOutput);
  Value reduceDim = rewriter.create<TF::ConstOp>(
      loc, mhlo::GetI64ElementsAttr({1}, &rewriter));
  Value sum = rewriter.create<TF::SumOp>(
      loc, mul, reduceDim, /*keep_dims*/ rewriter.getBoolAttr(false));
  return sum;
}

Value replaceWhereDynamic(PatternRewriter &rewriter, Location loc, Value input,
                          Value oneHotInput, Value depth, Value onValue,
                          Value offValue, Value gather_axis,
                          IntegerAttr oneHotAxis) {
  auto inputType = input.getType().dyn_cast<RankedTensorType>();
  auto oneHotInputType = oneHotInput.getType().dyn_cast<RankedTensorType>();
  assert(inputType.getRank() > oneHotInputType.getRank());
  auto zeroAttr = DenseIntElementsAttr::get(
      RankedTensorType::get({1}, oneHotInputType.getElementType()), 0);
  Value zero = rewriter.create<TF::ConstOp>(loc, zeroAttr);
  Value compare = rewriter.create<TF::GreaterEqualOp>(loc, oneHotInput, zero);

  auto whereType = RankedTensorType::get({ShapedType::kDynamic, 1},
                                         rewriter.getIntegerType(64));
  Value indices = rewriter.create<TF::WhereOp>(loc, whereType, compare);

  auto squeezeType = RankedTensorType::get({ShapedType::kDynamic},
                                           rewriter.getIntegerType(64));
  indices = rewriter.create<TF::SqueezeOp>(loc, squeezeType, indices,
                                           rewriter.getI64ArrayAttr({1}));

  llvm::SmallVector<int64_t> oneHotInputGatherShape =
      llvm::to_vector(oneHotInputType.getShape());
  oneHotInputGatherShape[0] = ShapedType::kDynamic;
  auto oneHotInputGatherType = oneHotInputType.clone(oneHotInputGatherShape);
  Value oneHotInputGather = rewriter.create<TF::GatherV2Op>(
      loc, oneHotInputGatherType, oneHotInput, indices, gather_axis);

  Value oneHotOutput = rewriter.create<TF::OneHotOp>(
      loc, oneHotInputGather, depth, onValue, offValue, oneHotAxis);

  llvm::SmallVector<int64_t> inputGatherShape =
      llvm::to_vector(inputType.getShape());
  inputGatherShape[0] = ShapedType::kDynamic;
  auto inputGatherType = inputType.clone(inputGatherShape);
  Value inputGather = rewriter.create<TF::GatherV2Op>(
      loc, inputGatherType, input, indices, gather_axis);

  return replaceWhereStatic(rewriter, loc, inputGather, oneHotOutput);
}

#include "tf_mlir_ext/transforms/fuse_tf_ops.inc"

struct FuseTFOpsPass : public FuseTFOpsBase<FuseTFOpsPass> {
  FuseTFOpsPass(bool replaceWhereToStatic) {
    this->replaceWhereToStatic = replaceWhereToStatic;
  }

  void runOnOperation() override final {
    MLIRContext *ctx = &getContext();
    func::FuncOp funcOp = getOperation();
    RewritePatternSet patterns(ctx);

    patterns.add(std::make_unique<FuseDilatedConv3DPattern>(ctx));
    patterns.add(std::make_unique<FuseSigmoid>(ctx));
    if (replaceWhereToStatic) {
      patterns.add(std::make_unique<ReplaceWhereStatic>(ctx));
    } else {
      patterns.add(std::make_unique<ReplaceWhereDynamic>(ctx));
    }

    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::tfext::createFuseTFOpsPass(bool replaceWhereToStatic) {
  return std::make_unique<FuseTFOpsPass>(replaceWhereToStatic);
}
