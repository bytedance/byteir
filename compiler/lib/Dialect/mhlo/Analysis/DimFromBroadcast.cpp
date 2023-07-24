//===- DimFromBroadcast.cpp -----------------------------------------------===//
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

#include "byteir/Dialect/mhlo/Analysis/DimFromBroadcast.h"
#include "mhlo/IR/hlo_ops.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace llvm;
using namespace mlir;
using namespace byteir;

namespace {

SmallVector<unsigned> getGreaterThanOneIdx(ArrayRef<int64_t> array) {
  SmallVector<unsigned> indices;
  for (unsigned i = 0; i < array.size(); ++i) {
    if (array[i] > 1) {
      indices.push_back(i);
    }
  }
  return indices;
}

SmallVector<bool> broadcastInDimHandleFlag(mhlo::BroadcastInDimOp op,
                                           int64_t rank) {
  auto denseAttr = op.getBroadcastDimensions();
  SmallVector<bool> res(rank, true);
  for (int64_t i = 0, e = denseAttr.getNumElements(); i < e; ++i)
    res[denseAttr.getValues<APInt>()[i].getSExtValue()] = false;
  return res;
}

SmallVector<bool> reshapeHandleFlag(mhlo::ReshapeOp op, int64_t rank,
                                    ArrayRef<int64_t> oupShape,
                                    DimFlagAnalysis *analysis) {
  SmallVector<bool> res(rank, false);
  Value inp = op.getOperand();
  auto inpShapedType = inp.getType().dyn_cast<ShapedType>();
  if (!inpShapedType || !inpShapedType.hasRank()) {
    return res;
  }
  // TODO: This is only a conservative check currently. Will not
  // check pattern like X[a*b, c] = mhlo.reshape(Y[a, b, c])
  ArrayRef<int64_t> inpShape = inpShapedType.getShape();
  SmallVector<unsigned> oupGreaterThanOneIdx = getGreaterThanOneIdx(oupShape);
  SmallVector<unsigned> inpGreaterThanOneIdx = getGreaterThanOneIdx(inpShape);
  if (oupGreaterThanOneIdx.size() != inpGreaterThanOneIdx.size()) {
    return res;
  }
  for (unsigned i = 0; i < oupGreaterThanOneIdx.size(); ++i) {
    if (oupShape[oupGreaterThanOneIdx[i]] !=
        inpShape[inpGreaterThanOneIdx[i]]) {
      return res;
    }
  }
  SmallVector<bool> inpRes = analysis->getDimFlag(inp);
  for (unsigned i = 0; i < oupGreaterThanOneIdx.size(); ++i) {
    res[oupGreaterThanOneIdx[i]] = inpRes[inpGreaterThanOneIdx[i]];
  }
  return res;
}

SmallVector<bool> unaryElementwiseHandleFlag(Operation *op,
                                             DimFlagAnalysis *analysis) {
  return analysis->getDimFlag(op->getOperand(0));
}

SmallVector<bool> binaryElementwiseHandleFlag(Operation *op,
                                              DimFlagAnalysis *analysis) {
  SmallVector<bool> left = analysis->getDimFlag(op->getOperand(0));
  SmallVector<bool> right = analysis->getDimFlag(op->getOperand(1));
  if (left.size() != right.size()) {
    return SmallVector<bool>();
  }
  SmallVector<bool> res(false, left.size());
  for (size_t i = 0; i < left.size(); ++i) {
    res[i] = left[i] && right[i];
  }
  return res;
}
} // namespace

SmallVector<bool> DimFromBroadcast::compute(Value v) {
  auto shapedType = v.getType().dyn_cast<ShapedType>();
  if (!shapedType || !shapedType.hasRank()) {
    return SmallVector<bool>();
  }
  int64_t curRank = shapedType.getRank();
  ArrayRef<int64_t> curShape = shapedType.getShape();

  Operation *defOp = v.getDefiningOp();
  SmallVector<bool> dimFlag =
      llvm::TypeSwitch<Operation *, SmallVector<bool>>(defOp)
          .Case<mhlo::BroadcastInDimOp>(
              [&](auto op) { return broadcastInDimHandleFlag(op, curRank); })
          .Case<mhlo::ReshapeOp>([&](auto op) {
            return reshapeHandleFlag(op, curRank, curShape, analysis);
          })
          .Case<mhlo::AbsOp, mhlo::CbrtOp, mhlo::CeilOp, mhlo::ConvertOp,
                mhlo::ClzOp, mhlo::CosineOp, mhlo::ExpOp, mhlo::Expm1Op,
                mhlo::FloorOp, mhlo::ImagOp, mhlo::IsFiniteOp, mhlo::LogOp,
                mhlo::Log1pOp, mhlo::LogisticOp, mhlo::NotOp, mhlo::NegOp,
                mhlo::PopulationCountOp, mhlo::RealOp, mhlo::RoundOp,
                mhlo::RsqrtOp, mhlo::SignOp, mhlo::SineOp, mhlo::SqrtOp,
                mhlo::TanhOp>(
              [&](auto op) { return unaryElementwiseHandleFlag(op, analysis); })
          .Case<mhlo::AddOp, mhlo::Atan2Op, mhlo::ComplexOp, mhlo::DivOp,
                mhlo::MaxOp, mhlo::MinOp, mhlo::MulOp, mhlo::PowOp, mhlo::RemOp,
                mhlo::ShiftLeftOp, mhlo::ShiftRightArithmeticOp,
                mhlo::ShiftRightLogicalOp, mhlo::SubtractOp, mhlo::AndOp,
                mhlo::OrOp, mhlo::XorOp>([&](auto op) {
            return binaryElementwiseHandleFlag(op, analysis);
          })
          // TODO: Handle more operation types here.
          .Default(
              [&](Operation *) { return SmallVector<bool>(curRank, false); });
  return dimFlag;
}
