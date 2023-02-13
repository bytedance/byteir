//===- Utils.cpp --------------------------------------------------- C++ --===//
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

#include "byteir/Transforms/Utils.h"
#include "byteir/Utils/Hoist.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;

namespace {

class HoistUpInBlockPattern : public RewritePattern {
public:
  HoistUpInBlockPattern(MLIRContext *context, DominanceInfo &dom,
                        std::function<bool(Operation *)> fun,
                        PatternBenefit benefit = 1)
      : RewritePattern(MatchAnyOpTypeTag(), benefit, context),
        controlFn(std::move(fun)), domInfo(dom) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (!controlFn(op)) {
      return failure();
    }

    auto pos = findHoistUpInBlock(op, domInfo);
    if (op->getPrevNode() == pos) {
      return failure();
    }

    op->moveAfter(pos);
    return success();
  }

private:
  std::function<bool(Operation *)> controlFn;
  DominanceInfo &domInfo;
};

class HoistDownInBlockPattern : public RewritePattern {
public:
  HoistDownInBlockPattern(MLIRContext *context, PostDominanceInfo &post,
                          std::function<bool(Operation *)> fun,
                          PatternBenefit benefit = 1)
      : RewritePattern(MatchAnyOpTypeTag(), benefit, context),
        controlFn(std::move(fun)), postDomInfo(post) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (!controlFn(op)) {
      return failure();
    }

    auto pos = findHoistDownInBlock(op, postDomInfo);
    if (op->getNextNode() == pos) {
      return failure();
    }

    op->moveBefore(pos);
    return success();
  }

private:
  std::function<bool(Operation *)> controlFn;
  PostDominanceInfo &postDomInfo;
};

class RemoveAttrPattern : public RewritePattern {
public:
  RemoveAttrPattern(MLIRContext *context, llvm::StringRef attr,
                    PatternBenefit benefit = 1)
      : RewritePattern(MatchAnyOpTypeTag(), benefit, context), attrName(attr) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (!op->hasAttr(attrName)) {
      return failure();
    }
    op->removeAttr(attrName);
    return success();
  }

private:
  std::string attrName;
};

} // namespace

void mlir::populateHoistUpInBlockPatterns(
    RewritePatternSet &patterns, DominanceInfo &domInfo,
    const std::function<bool(Operation *)> &checkFunc) {
  auto *context = patterns.getContext();
  patterns.add<HoistUpInBlockPattern>(context, domInfo, checkFunc);
}

void mlir::populateHoistDownInBlockPatterns(
    RewritePatternSet &patterns, PostDominanceInfo &postDomInfo,
    const std::function<bool(Operation *)> &checkFunc) {
  auto *context = patterns.getContext();
  patterns.add<HoistDownInBlockPattern>(context, postDomInfo, checkFunc);
}

void mlir::populateRemoveAttrPatterns(RewritePatternSet &patterns,
                                      llvm::StringRef attrName) {
  auto *context = patterns.getContext();
  patterns.add<RemoveAttrPattern>(context, attrName);
}
