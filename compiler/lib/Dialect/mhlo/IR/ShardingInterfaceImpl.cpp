//===- ShardingInterfaceImpl.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/mhlo/IR/ShardingInterfaceImpl.h"
#include "byteir/Dialect/Mesh/IR/MeshOps.h"
#include "byteir/Dialect/Mesh/Interfaces/ShardingInterface.h"
#include "byteir/Utils/AffineUtils.h"
#include "byteir/Utils/AttrUtils.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/DialectRegistry.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "mhlo-sharding-impl"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE << "]: ")

using namespace mlir;
using namespace mlir::mhlo;
using namespace mlir::mesh;

namespace {

template <typename ElemwiseOp>
struct ElemwiseSharding
    : public ShardingInterface::ExternalModel<ElemwiseSharding<ElemwiseOp>,
                                              ElemwiseOp> {
  SmallVector<ShardingIteratorType> getLoopIteratorTypes(Operation *op) const {
    Value val = op->getOperand(0);
    auto type = val.getType().cast<RankedTensorType>();
    SmallVector<ShardingIteratorType> types(type.getRank(),
                                            ShardingIteratorType::parallel);
    return types;
  }

  SmallVector<AffineMap> getIndexingMaps(Operation *op) const {
    MLIRContext *ctx = op->getContext();
    Value val = op->getOperand(0);
    auto type = val.getType().cast<RankedTensorType>();
    int64_t rank = type.getRank();
    int64_t num = op->getNumOperands() + op->getNumResults();
    SmallVector<AffineMap> maps(num,
                                AffineMap::getMultiDimIdentityMap(rank, ctx));
    return maps;
  }
};

struct ConstantOpSharding
    : public ShardingInterface::ExternalModel<ConstantOpSharding, ConstantOp> {
  SmallVector<ShardingIteratorType> getLoopIteratorTypes(Operation *op) const {
    Value val = op->getResult(0);
    RankedTensorType tensorType = val.getType().cast<RankedTensorType>();
    int64_t rank = tensorType.getRank();
    SmallVector<ShardingIteratorType> types(rank,
                                            ShardingIteratorType::parallel);
    return types;
  }

  SmallVector<AffineMap> getIndexingMaps(Operation *op) const {
    MLIRContext *ctx = op->getContext();
    Value val = op->getResult(0);
    RankedTensorType tensorType = val.getType().cast<RankedTensorType>();
    int64_t rank = tensorType.getRank();
    SmallVector<AffineMap> maps;
    maps.push_back(AffineMap::getMultiDimIdentityMap(rank, ctx));
    return maps;
  }

  FailureOr<ShardingOption> getShardingOption(Operation *op,
                                              OpBuilder &b) const {
    ShardingOption shardingOption;
    return shardingOption;
  }

  LogicalResult setShardingAnnotations(Operation *op, OpBuilder &b) const {
    bool isAnnotated = llvm::any_of(op->getUsers(), [&](Operation *user) {
      auto annotateOp = llvm::dyn_cast<mesh::AnnotateOp>(user);
      if (!annotateOp)
        return false;
      return annotateOp.getAsResult();
    });
    if (isAnnotated)
      return success();

    ShardingOption shardingOption;
    ArrayAttr shardingAnnotationAttr =
        convertArrayOfI64ArrayToAttr(b, shardingOption);
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointAfter(op);
    auto annotateOp = b.create<mesh::AnnotateOp>(
        op->getLoc(), op->getResult(0).getType(), op->getResult(0),
        shardingAnnotationAttr, /*required*/ false, /*as_result*/ true);
    op->getResult(0).replaceAllUsesExcept(annotateOp, annotateOp);
    return success();
  }
};

// m:0, n:1, k:2
// A : (m, k) or (0, 2)
// B : (k, n) or (2, 1)
// C : (m, n) or (0, 1)
struct DotOpSharding
    : public ShardingInterface::ExternalModel<DotOpSharding, DotOp> {
  SmallVector<ShardingIteratorType> getLoopIteratorTypes(Operation *op) const {
    SmallVector<ShardingIteratorType> types(3, ShardingIteratorType::parallel);
    types[2] = ShardingIteratorType::reduction_sum;
    return types;
  }

  SmallVector<AffineMap> getIndexingMaps(Operation *op) const {
    MLIRContext *ctx = op->getContext();
    SmallVector<AffineMap> maps;
    maps.push_back(getMultiDimIdentityMapWithTargets(3, {0, 2}, ctx));
    maps.push_back(getMultiDimIdentityMapWithTargets(3, {2, 1}, ctx));
    maps.push_back(getMultiDimIdentityMapWithTargets(3, {0, 1}, ctx));
    return maps;
  }
};

struct DotGeneralOpSharding
    : public ShardingInterface::ExternalModel<DotGeneralOpSharding,
                                              DotGeneralOp> {
  // The order of loop iterator types will be
  // batch_dims, other_lhs_parallel_dims, other_rhs_parallel_dims, contract_dims
  SmallVector<ShardingIteratorType> getLoopIteratorTypes(Operation *op) const {
    DotGeneralOp dotOp = llvm::cast<DotGeneralOp>(op);
    Value lhs = dotOp.getLhs();
    Value rhs = dotOp.getRhs();
    RankedTensorType lhsType = lhs.getType().cast<RankedTensorType>();
    RankedTensorType rhsType = rhs.getType().cast<RankedTensorType>();

    DotDimensionNumbersAttr dimNumsAttr = dotOp.getDotDimensionNumbers();
    ArrayRef<int64_t> lhsBatchDims = dimNumsAttr.getLhsBatchingDimensions();
    ArrayRef<int64_t> lhsContDims = dimNumsAttr.getLhsContractingDimensions();
    int64_t batchAndContRank = lhsBatchDims.size() + lhsContDims.size();
    int64_t loopNum = lhsType.getRank() + rhsType.getRank() - batchAndContRank;

    SmallVector<ShardingIteratorType> types(loopNum,
                                            ShardingIteratorType::parallel);
    for (int64_t i = loopNum - lhsContDims.size(); i < loopNum; ++i)
      types[i] = ShardingIteratorType::reduction_sum;
    return types;
  }

  SmallVector<AffineMap> getIndexingMaps(Operation *op) const {
    DotGeneralOp dotOp = llvm::cast<DotGeneralOp>(op);
    MLIRContext *ctx = op->getContext();
    SmallVector<AffineMap> maps;

    Value lhs = dotOp.getLhs();
    Value rhs = dotOp.getRhs();
    RankedTensorType lhsType = lhs.getType().cast<RankedTensorType>();
    RankedTensorType rhsType = rhs.getType().cast<RankedTensorType>();

    DotDimensionNumbersAttr dimNumsAttr = dotOp.getDotDimensionNumbers();
    ArrayRef<int64_t> lhsBatchDims = dimNumsAttr.getLhsBatchingDimensions();
    ArrayRef<int64_t> lhsContDims = dimNumsAttr.getLhsContractingDimensions();
    ArrayRef<int64_t> rhsBatchDims = dimNumsAttr.getRhsBatchingDimensions();
    ArrayRef<int64_t> rhsContDims = dimNumsAttr.getRhsContractingDimensions();
    int64_t batchDimsNum = lhsBatchDims.size();
    int64_t contDimsNum = lhsContDims.size();
    int64_t batchAndContRank = batchDimsNum + contDimsNum;
    int64_t loopNum = lhsType.getRank() + rhsType.getRank() - batchAndContRank;

    // indexing map for lhs and rhs
    SmallVector<int64_t> lhsResultPositions(lhsType.getRank(), -1);
    SmallVector<int64_t> rhsResultPositions(rhsType.getRank(), -1);
    for (auto it : llvm::enumerate(llvm::zip(lhsBatchDims, rhsBatchDims))) {
      int64_t idx = it.index();
      int64_t lhsDim = std::get<0>(it.value());
      int64_t rhsDim = std::get<1>(it.value());
      lhsResultPositions[lhsDim] = idx;
      rhsResultPositions[rhsDim] = idx;
    }
    for (auto it : llvm::enumerate(llvm::zip(lhsContDims, rhsContDims))) {
      int64_t idx = it.index();
      int64_t lhsDim = std::get<0>(it.value());
      int64_t rhsDim = std::get<1>(it.value());
      lhsResultPositions[lhsDim] = idx + loopNum - contDimsNum;
      rhsResultPositions[rhsDim] = idx + loopNum - contDimsNum;
    }
    for (size_t i = 0, curPos = batchDimsNum; i < lhsResultPositions.size();
         ++i) {
      if (lhsResultPositions[i] != -1)
        continue;
      lhsResultPositions[i] = curPos++;
    }
    for (size_t i = 0, curPos = lhsType.getRank() - contDimsNum;
         i < rhsResultPositions.size(); ++i) {
      if (rhsResultPositions[i] != -1)
        continue;
      rhsResultPositions[i] = curPos++;
    }
    maps.push_back(
        getMultiDimIdentityMapWithTargets(loopNum, lhsResultPositions, ctx));
    maps.push_back(
        getMultiDimIdentityMapWithTargets(loopNum, rhsResultPositions, ctx));

    // indexing map for output
    SmallVector<int64_t> outputResultPostions;
    for (int64_t i = 0; i < loopNum - contDimsNum; ++i)
      outputResultPostions.push_back(i);
    maps.push_back(
        getMultiDimIdentityMapWithTargets(loopNum, outputResultPostions, ctx));
    return maps;
  }
};

} // namespace

namespace {

template <typename OpType> static void registerElemwiseOne(MLIRContext *ctx) {
  OpType::template attachInterface<ElemwiseSharding<OpType>>(*ctx);
}

/// Variadic helper function.
template <typename... OpTypes>
static void registerElemwiseAll(MLIRContext *ctx) {
  (registerElemwiseOne<OpTypes>(ctx), ...);
}

} // namespace

void mlir::mhlo::registerShardingInterfaceExternalModels(
    DialectRegistry &registry) {

  registry.addExtension(+[](MLIRContext *ctx, MhloDialect *dialect) {
    ConstantOp::attachInterface<ConstantOpSharding>(*ctx);
    DotOp::attachInterface<DotOpSharding>(*ctx);
    DotGeneralOp::attachInterface<DotGeneralOpSharding>(*ctx);

    // unary element wise ops
    registerElemwiseAll<AbsOp, CbrtOp, CeilOp, ConvertOp, ClzOp, CosineOp,
                        ExpOp, Expm1Op, FloorOp, ImagOp, IsFiniteOp, LogOp,
                        Log1pOp, LogisticOp, NotOp, NegOp, PopulationCountOp,
                        RealOp, RoundOp, RoundNearestEvenOp, RsqrtOp, SignOp,
                        SineOp, TanOp, SqrtOp, TanhOp>(ctx);

    // binary element wise ops
    registerElemwiseAll<AddOp, Atan2Op, ComplexOp, DivOp, MaxOp, MinOp, MulOp,
                        PowOp, RemOp, ShiftLeftOp, ShiftRightArithmeticOp,
                        ShiftRightLogicalOp, SubtractOp, AndOp, OrOp, XorOp>(
        ctx);
  });
}
