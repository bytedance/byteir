//===- Transforms/SimplifyLinearizedIndex.cpp -----------------*--- C++ -*-===//
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

#include "byteir/Dialect/MemRef/Transforms/SimplifyLinearizedIndex.h"
#include "byteir/Dialect/MemRef/Utils/Layout.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "PassDetail.h"

using namespace llvm;
using namespace mlir;
using namespace mlir::memref;

namespace {
struct SimplifyLinearizedIndex : public OpRewritePattern<arith::AddIOp> {
  using OpRewritePattern<arith::AddIOp>::OpRewritePattern;

  Value simplify(PatternRewriter &rewriter, Location loc, Value lhs,
                 Value rhs) const {
    if (auto mulOp = lhs.getDefiningOp<arith::MulIOp>()) {
      if (auto remOp = rhs.getDefiningOp<arith::RemSIOp>()) {
        auto I = remOp.getLhs();
        auto S0 = remOp.getRhs();
        if (mulOp.getRhs() != S0)
          return nullptr;

        // Case0 : (I / S0 % S1) * S0 + I % S0 => I % (S0 * S1)
        if (auto rem2Op = mulOp.getLhs().getDefiningOp<arith::RemSIOp>()) {
          auto S1 = rem2Op.getRhs();
          if (auto divOp = rem2Op.getLhs().getDefiningOp<arith::DivSIOp>()) {
            if (divOp.getLhs() != I || divOp.getRhs() != S0) {
              return nullptr;
            }
            auto S = rewriter.create<arith::MulIOp>(loc, S0, S1);
            return rewriter.create<arith::RemSIOp>(loc, I, S);
          }
        }

        // Case1: (I / S0) * S0 + I % S0 => I
        if (auto divOp = mulOp.getLhs().getDefiningOp<arith::DivSIOp>()) {
          if (divOp.getLhs() != I || divOp.getRhs() != S0) {
            return nullptr;
          }

          return I;
        }
      }
    }
    return nullptr;
  }

  LogicalResult matchAndRewrite(arith::AddIOp addOp,
                                PatternRewriter &rewriter) const override {
    if (!addOp.getType().isIndex())
      return failure();

    auto loc = addOp->getLoc();
    auto lhs = addOp.getLhs(), rhs = addOp.getRhs();

    if (auto newValue = simplify(rewriter, loc, lhs, rhs)) {
      rewriter.replaceOp(addOp, newValue);
      return success();
    }

    if (auto addOp2 = lhs.getDefiningOp<arith::AddIOp>()) {
      if (auto newValue = simplify(rewriter, loc, addOp2.getRhs(), rhs)) {
        rewriter.replaceOpWithNewOp<arith::AddIOp>(addOp, addOp2.getLhs(),
                                                   newValue);
        return success();
      }
    }

    if (auto addOp2 = rhs.getDefiningOp<arith::AddIOp>()) {
      if (auto newValue = simplify(rewriter, loc, lhs, addOp2.getLhs())) {
        rewriter.replaceOpWithNewOp<arith::AddIOp>(addOp, newValue,
                                                   addOp2.getRhs());
        return success();
      }
    }

    return failure();
  }
};

// x / c0 / c1 -> x / (c0 * c1)
struct FoldConsecutiveDivI : public OpRewritePattern<arith::DivSIOp> {
  using OpRewritePattern<arith::DivSIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::DivSIOp divOp,
                                PatternRewriter &rewriter) const override {
    if (!divOp.getType().isIndex())
      return failure();

    if (auto div2Op = divOp.getLhs().getDefiningOp<arith::DivSIOp>()) {
      if (auto c1 = divOp.getRhs().getDefiningOp<arith::ConstantIndexOp>()) {
        if (auto c0 = div2Op.getRhs().getDefiningOp<arith::ConstantIndexOp>()) {
          auto constant =
              rewriter.create<arith::MulIOp>(divOp->getLoc(), c0, c1);
          rewriter.replaceOpWithNewOp<arith::DivSIOp>(divOp, div2Op.getLhs(),
                                                      constant);
          return success();
        }
      }
    }
    return failure();
  }
};

struct SimplifyLinearizedIndexPass
    : public SimplifyLinearizedIndexBase<SimplifyLinearizedIndexPass> {
  void runOnOperation() override {

    RewritePatternSet patterns(&getContext());
    patterns.add<SimplifyLinearizedIndex, FoldConsecutiveDivI>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::createSimplifyLinearizedIndexPass() {
  return std::make_unique<SimplifyLinearizedIndexPass>();
}