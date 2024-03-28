//===- LcclToByre.cpp -----------------------------------------*--- C++ -*-===//
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

#include "byteir/Conversion/LcclToByre/LcclToByre.h"
#include "byteir/Dialect/Byre/ByreDialect.h"
#include "byteir/Dialect/Byre/Common.h"
#include "byteir/Dialect/Lccl/LcclOps.h"
#include "byteir/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/DialectConversion.h"

#include "../PassDetail.h"

using namespace mlir;
using namespace llvm;

namespace {

template <typename T> const StringRef getByreOpName() {
  if (std::is_same_v<T, lccl::BroadcastOp>)
    return byre::ByreBroadcastName;
  else if (std::is_same_v<T, lccl::AllReduceOp>)
    return byre::ByreAllReduceName;
  else if (std::is_same_v<T, lccl::AllGatherOp>)
    return byre::ByreAllGatherName;
  return "";
}

template <typename T>
SmallVector<NamedAttribute> getOpAttrExceptEeplicaGroups(T op) {
  SmallVector<NamedAttribute> attrs;
  for (auto attr : op.getOperation()->getAttrDictionary()) {
    if (attr.getName() == op.getReplicaGroupsAttrName())
      continue;
    attrs.push_back(attr);
  }
  return attrs;
}

template <typename T>
struct ConvertLcclOpToByrePattern : public OpRewritePattern<T> {
  using OpRewritePattern<T>::OpRewritePattern;
  LogicalResult matchAndRewrite(T op,
                                PatternRewriter &rewriter) const override {
    if (op.getDynamicReplicaGroups()) {
      auto byreOp = rewriter.replaceOpWithNewOp<byre::ComputeOp>(
          op, TypeRange(), getByreOpName<T>(), op.getOperands(), ArrayAttr());
      auto &&attrs = getOpAttrExceptEeplicaGroups(op);
      attrs.emplace_back(op.getSynchronousAttrName(), op.getSynchronousAttr());
      addAttrs(byreOp.getOperation(), attrs);
    } else {
      auto replicaGroups = op.getReplicaGroupsAttr();
      for (Attribute replicaGroup : replicaGroups) {
        auto byreOp = rewriter.create<byre::ComputeOp>(
            op.getLoc(), TypeRange(), getByreOpName<T>(), op.getOperands(),
            ArrayAttr());
        auto &&attrs = getOpAttrExceptEeplicaGroups(op);
        attrs.emplace_back(rewriter.getStringAttr(byre::ReplicaGroupStr),
                           replicaGroup);
        addAttrs(byreOp.getOperation(), attrs);
      }
      rewriter.eraseOp(op);
    }
    return success();
  }
};

struct ConvertSendOpToByrePattern : public OpRewritePattern<lccl::SendOp> {
  using OpRewritePattern<lccl::SendOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(lccl::SendOp op,
                                PatternRewriter &rewriter) const override {
    auto byreOp = rewriter.replaceOpWithNewOp<byre::ComputeOp>(
        op, TypeRange(), byre::ByreSendName, op.getOperands(), ArrayAttr());
    SmallVector<NamedAttribute> attrs;
    if (op.getTargetIndex().has_value())
      attrs.emplace_back(rewriter.getStringAttr(byre::ByreRankStr),
                         op.getTargetIndexAttr());
    attrs.emplace_back(op.getSynchronousAttrName(), op.getSynchronousAttr());
    addAttrs(byreOp.getOperation(), attrs);
    return success();
  }
};

struct ConvertRecvOpToByrePattern : public OpRewritePattern<lccl::RecvOp> {
  using OpRewritePattern<lccl::RecvOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(lccl::RecvOp op,
                                PatternRewriter &rewriter) const override {
    auto byreOp = rewriter.replaceOpWithNewOp<byre::ComputeOp>(
        op, TypeRange(), byre::ByreRecvName, op.getOperands(), ArrayAttr());
    SmallVector<NamedAttribute> attrs;
    if (op.getSourceIndex().has_value())
      attrs.emplace_back(rewriter.getStringAttr(byre::ByreRankStr),
                         op.getSourceIndexAttr());
    attrs.emplace_back(op.getSynchronousAttrName(), op.getSynchronousAttr());
    addAttrs(byreOp.getOperation(), attrs);
    return success();
  }
};

struct ConvertLcclToByrePass
    : public ConvertLcclToByreBase<ConvertLcclToByrePass> {

public:
  void runOnOperation() override {
    MLIRContext &ctx = getContext();
    RewritePatternSet patterns(&ctx);
    ConversionTarget target(ctx);
    populateLcclToByrePattern(patterns);
    target.addIllegalDialect<lccl::LcclDialect>();
    target.addLegalOp<memref::AllocOp>();
    target.addLegalDialect<byre::ByreDialect>();
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
}; // namespace

void mlir::populateLcclToByrePattern(RewritePatternSet &patterns) {
  patterns.add<ConvertSendOpToByrePattern, ConvertRecvOpToByrePattern,
               ConvertLcclOpToByrePattern<lccl::BroadcastOp>,
               ConvertLcclOpToByrePattern<lccl::AllReduceOp>,
               ConvertLcclOpToByrePattern<lccl::AllGatherOp>>(
      patterns.getContext());
}

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createConvertLcclToByrePass() {
  return std::make_unique<ConvertLcclToByrePass>();
}
