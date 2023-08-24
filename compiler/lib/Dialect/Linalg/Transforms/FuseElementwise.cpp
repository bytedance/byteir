//===- FuseElementwise.cpp ------------------------------------*--- C++ -*-===//
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
// Some code from ElementwiseOpFusion.cpp of LLVM
// Original license:
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/Linalg/Transforms/FuseElementwise.h"

#include "byteir/Dialect/Linalg/IR/LinalgExtOps.h"
#include "byteir/Dialect/Linalg/Transforms/Transforms.h"
#include "byteir/Utils/AffineUtils.h"
#include "byteir/Utils/Hoist.h"
#include "byteir/Utils/TypeUtils.h"
#include "byteir/Utils/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#include "PassDetail.h"

#include <iostream>

using namespace llvm;
using namespace mlir;
using namespace mlir::linalg;
using namespace mlir::linalg_ext;

#define DEBUG_TYPE "linalg-fuse-elementwise-ext"

namespace {

// an internal attr to help identify aliasGeneric
constexpr StringRef getInternalAliasAttrName() { return "__internal_alias__"; }

static FailureOr<linalg::GenericOp> getAliasGeneric(OpBuilder &b, Value val) {
  auto tensorTy = val.getType().dyn_cast<TensorType>();
  if (!tensorTy) {
    return failure();
  }

  auto loc = val.getLoc();
  SmallVector<Type> resultTypes(1, tensorTy);
  SmallVector<Value> inputOperands(1, val);
  SmallVector<OpFoldResult> sizes;
  sizes.reserve(tensorTy.getRank());
  for (int64_t i = 0; i < tensorTy.getRank(); ++i) {
    sizes.push_back(getDim(b, loc, val, i));
  }

  auto empty = b.create<tensor::EmptyOp>(loc, sizes, tensorTy.getElementType());
  SmallVector<Value> outputOperands(1, empty);
  SmallVector<AffineMap> indexMaps(
      2, b.getMultiDimIdentityMap(tensorTy.getRank()));
  SmallVector<utils::IteratorType> iteratorTys(tensorTy.getRank(),
                                               utils::IteratorType::parallel);

  auto genericOp = b.create<GenericOp>(val.getLoc(), resultTypes, inputOperands,
                                       outputOperands, indexMaps, iteratorTys);

  // generate region
  Block *block = new Block();
  genericOp.getRegion().push_back(block);
  auto ip = b.saveInsertionPoint();
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointToStart(block);
  auto bbArgInput =
      block->addArgument(tensorTy.getElementType(), genericOp.getLoc());
  block->addArgument(tensorTy.getElementType(), genericOp.getLoc());
  auto alias = b.create<linalg_ext::AliasOp>(genericOp.getLoc(), bbArgInput);
  SmallVector<Value> yieldValues(1, alias.getResult());
  b.create<linalg::YieldOp>(genericOp.getLoc(), yieldValues);

  // insert getInternalAliasAttrName attr
  genericOp->setAttr(getInternalAliasAttrName(), b.getUnitAttr());

  b.restoreInsertionPoint(ip);
  return genericOp;
}

// TODO: move to public
static bool checkDominateAllUsers(Operation *replaceOp, Value orig,
                                  DominanceInfo &domInfo) {
  for (auto user : orig.getUsers()) {
    if (!domInfo.properlyDominates(replaceOp, user)) {
      return false;
    }
  }
  return true;
}

static bool isBroadcast(GenericOp op) {
  mlir::Block &b = op.getRegion().front();
  // this is a trick, using 1 as yieldOp only
  if (b.getOperations().size() == 1) {
    return true;
  }
  return false;
}

static bool areShapeEqual(GenericOp consumer, OpOperand *producer) {
  // consumer
  Value consumerOutput = consumer.getOutputs()[0];
  auto consumerShapeTy = consumerOutput.getType().dyn_cast<ShapedType>();
  if (!consumerShapeTy)
    return false;

  auto producerOp = producer->get().getDefiningOp<GenericOp>();
  for (auto output : producerOp.getOutputs()) {
    auto outputShapeTy = output.getType().dyn_cast<ShapedType>();
    if (!outputShapeTy)
      return false;
    if (!areSameShape(consumerShapeTy, outputShapeTy)) {
      return false;
    }
  }

  return true;
}

static bool isUsedByReturn(OpOperand *producer) {
  for (Operation *useOp : producer->get().getUsers()) {
    if (isa<func::ReturnOp>(useOp))
      return true;
  }

  return false;
}

/// Append to `fusedOpIndexingMapAttrs` the indexing maps for the operands of
/// the `producer` to use in the fused operation given the indexing map of the
/// result of the producer in the consumer.
static AffineMap getIndexingMapOfProducerOperandsInCoordinatesOfFusedOp(
    OpOperand *producerOpOperand, AffineMap producerResultIndexMap,
    AffineMap fusedConsumerArgIndexMap) {
  // The indexing map in the consumer op (fusedConsumerArgIndexMap) is a map
  // from consumer loop -> consumer arg tensor index/producer result tensor
  // index. The fused loop is same as the consumer loop. For each producer arg
  // the indexing map to be computed is a map from consumer loop -> producer
  // arg tensor index.
  // producerResultIndexMap is a map from producer loop -> tensor index.
  // Compute the inverse to get map from tensor index -> producer loop.
  // The inverse is a map from producer result tensor index -> producer loop.
  AffineMap invProducerResultIndexMap =
      inversePermutation(producerResultIndexMap);
  assert(invProducerResultIndexMap &&
         "expected producer result indexing map to be invertible");

  LinalgOp producer = cast<LinalgOp>(producerOpOperand->getOwner());
  // argMap is a map from producer loop -> producer arg tensor index.
  AffineMap argMap = producer.getMatchingIndexingMap(producerOpOperand);

  // Compose argMap with invProducerResultIndexMap to get a map from
  // producer result tensor index -> producer arg tensor index.
  AffineMap t1 = argMap.compose(invProducerResultIndexMap);

  // Compose t1 with fusedConsumerArgIndexMap gives an indexing map from
  // consumer loop/ fused loop -> producer arg tensor index.
  return t1.compose(fusedConsumerArgIndexMap);
}

static AffineMap
composeProduceResultAndConsumerOperand(AffineMap producerResultIndexMap,
                                       AffineMap fusedConsumerArgIndexMap) {
  AffineMap invProducerResultIndexMap =
      inversePermutation(producerResultIndexMap);
  assert(invProducerResultIndexMap &&
         "expected producer result indexing map to be invertible");
  return invProducerResultIndexMap.compose(fusedConsumerArgIndexMap);
}

/// Generate the region of the fused tensor operation. The region of the fused
/// op must be empty.
///
/// This is an variant of upstream's version to support multiple fusedOperands
static void generateFusedElementwiseOpRegion(
    RewriterBase &rewriter, GenericOp fusedOp,
    AffineMap consumerToProducerLoopsMap,
    llvm::SetVector<OpOperand *> &fusedOperands, unsigned nloops,
    llvm::SmallDenseSet<int> &preservedProducerResults) {
  assert(fusedOperands.size() > 0);
  auto producer = cast<GenericOp>(fusedOperands[0]->get().getDefiningOp());
  auto consumer = cast<GenericOp>(fusedOperands[0]->getOwner());
  // Build the region of the fused op.
  Block &producerBlock = producer->getRegion(0).front();
  Block &consumerBlock = consumer->getRegion(0).front();
  Block *fusedBlock = new Block();
  fusedOp.getRegion().push_back(fusedBlock);
  IRMapping mapper;
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(fusedBlock);

  // 2. Add an index operation for every fused loop dimension and use the
  // `consumerToProducerLoopsMap` to map the producer indices.
  if (producer.hasIndexSemantics()) {
    // Add an index operation for every fused loop dimension.
    unsigned numFusedOpLoops =
        std::max(producer.getNumLoops(), consumer.getNumLoops());
    SmallVector<Value> fusedIndices;
    fusedIndices.reserve(numFusedOpLoops);
    llvm::transform(llvm::seq<uint64_t>(0, numFusedOpLoops),
                    std::back_inserter(fusedIndices), [&](uint64_t dim) {
                      return rewriter.create<IndexOp>(producer.getLoc(), dim);
                    });
    for (IndexOp indexOp :
         llvm::make_early_inc_range(producerBlock.getOps<IndexOp>())) {
      Value newIndex = rewriter.create<affine::AffineApplyOp>(
          producer.getLoc(),
          consumerToProducerLoopsMap.getSubMap(indexOp.getDim()), fusedIndices);
      mapper.map(indexOp.getResult(), newIndex);
    }
  }
  // TODO: allow fusing the producer of an output operand.
  for (OpOperand *fusedOperand : fusedOperands)
    assert(consumer.isDpsInput(fusedOperand) &&
           "expected producer of input operand");

  // Replacing consumerIdx requires getting the cloned, yielded, value from
  // the (cloned) producer block. This happens in step 9.

  // 3. All of the producer's input operands
  for (BlockArgument bbArg :
       producerBlock.getArguments().take_front(producer.getNumDpsInputs()))
    mapper.map(bbArg, fusedBlock->addArgument(bbArg.getType(), bbArg.getLoc()));

  // 4. All of the consumer's input operands
  for (auto it : llvm::zip(consumer.getDpsInputOperands(),
                           consumerBlock.getArguments())) {
    OpOperand *opOperand = std::get<0>(it);
    BlockArgument bbArg = std::get<1>(it);
    if (!fusedOperands.contains(opOperand))
      mapper.map(bbArg,
                 fusedBlock->addArgument(bbArg.getType(), bbArg.getLoc()));
  }

  // 5. All of the producer's output operands
  for (const auto &bbArg : llvm::enumerate(
           producerBlock.getArguments().take_back(producer.getNumDpsInits()))) {
    if (!preservedProducerResults.count(bbArg.index()))
      continue;
    mapper.map(bbArg.value(), fusedBlock->addArgument(bbArg.value().getType(),
                                                      bbArg.value().getLoc()));
  }

  // 6. All of consumer's output operands.
  for (BlockArgument bbArg :
       consumerBlock.getArguments().take_back(consumer.getNumDpsInits()))
    mapper.map(bbArg, fusedBlock->addArgument(bbArg.getType(), bbArg.getLoc()));

  // 7. Clone all producer operations except for the yield and index operations
  // to the fused operation.
  for (auto &op : producerBlock.without_terminator()) {
    if (!isa<IndexOp>(op))
      rewriter.clone(op, mapper);
  }
  // 8. Now we can map the consumerBlock's `consumerIdx` block argument. Just
  // forward the yield operand.
  auto producerYieldOp = cast<linalg::YieldOp>(producerBlock.getTerminator());
  for (OpOperand *fusedOperand : fusedOperands) {
    unsigned producerResultNumber =
        cast<OpResult>(fusedOperand->get()).getResultNumber();
    Value replacement = mapper.lookupOrDefault(
        producerYieldOp.getOperand(producerResultNumber));
    // Sanity checks, if replacement is not already in the mapper then it must
    // be produced outside.
    if (replacement == producerYieldOp.getOperand(producerResultNumber)) {
      if (auto bb = dyn_cast<BlockArgument>(replacement))
        assert(bb.getOwner() != &producerBlock &&
               "yielded block argument must have been mapped");
      else
        assert(!producer->isAncestor(replacement.getDefiningOp()) &&
               "yielded value must have been mapped");
    }
    mapper.map(consumerBlock.getArgument(fusedOperand->getOperandNumber()),
               replacement);
  }

  // 9. Clone operations from the consumer to the fused op.
  for (auto &op : consumerBlock.without_terminator())
    rewriter.clone(op, mapper);

  // 10. Include the final yield (which is the remapped values for all the
  // yield)
  auto consumerYieldOp = cast<linalg::YieldOp>(consumerBlock.getTerminator());
  SmallVector<Value> fusedYieldValues;
  fusedYieldValues.reserve(producerYieldOp.getNumOperands() +
                           consumerYieldOp.getNumOperands());
  for (const auto &producerYieldVal :
       llvm::enumerate(producerYieldOp.getOperands())) {
    if (preservedProducerResults.count(producerYieldVal.index()))
      fusedYieldValues.push_back(
          mapper.lookupOrDefault(producerYieldVal.value()));
  }
  for (auto consumerYieldVal : consumerYieldOp.getOperands())
    fusedYieldValues.push_back(mapper.lookupOrDefault(consumerYieldVal));
  rewriter.create<linalg::YieldOp>(fusedOp.getLoc(), fusedYieldValues);

  // Sanity checks.
  assert(fusedBlock->getNumArguments() == fusedOp.getNumOperands() &&
         "Ill-formed GenericOp region");
}

/// Fuse two `linalg.generic` operations that have a producer-consumer
/// relationship captured through `fusedOperand`. The method expects
/// that `areElementwiseOpsFusable` returns true for the given `fusedOperand`.
///
/// This is an enhancement of mlir upstream's version to support cases where the
/// consumer uses multiple results of the producer
static FailureOr<mlir::linalg::ElementwiseOpFusionResult>
fuseElementwiseOpsExt(RewriterBase &rewriter, OpOperand *fusedOperand) {
  assert(areElementwiseOpsFusable(fusedOperand) &&
         "expected elementwise operation pre-conditions to pass");
  auto producerResult = cast<OpResult>(fusedOperand->get());
  auto producer = cast<GenericOp>(producerResult.getOwner());
  auto consumer = cast<GenericOp>(fusedOperand->getOwner());
  // TODO: allow fusing the producer of an output operand.
  assert(consumer.isDpsInput(fusedOperand) &&
         "expected producer of input operand");
  /// Find the results of the producer that have uses outside of the consumer.
  llvm::SmallDenseSet<int> preservedProducerResults;
  for (const auto &producerResult : llvm::enumerate(producer->getResults())) {
    auto *outputOperand = producer.getDpsInitOperand(producerResult.index());
    if (producer.payloadUsesValueFromOperand(outputOperand) ||
        !producer.canOpOperandsBeDropped(outputOperand) ||
        llvm::any_of(producerResult.value().getUsers(), [&](Operation *user) {
          return user != consumer.getOperation();
        })) {
      preservedProducerResults.insert(producerResult.index());
    }
  }

  // Compute the fused operands list and indexing maps.
  SmallVector<Value> fusedInputOperands, fusedOutputOperands;
  SmallVector<Type> fusedResultTypes;
  SmallVector<AffineMap> fusedIndexMaps;
  // In the following, numbering matches that of `generateFusedTensorOpRegion`.
  // 3. Consumer input operands/maps up to consumerIdx (exclusive).
  auto consumerInputs = consumer.getDpsInputOperands();
  SmallVector<OpOperand *> operandsFromProducer = llvm::to_vector(
      llvm::make_filter_range(consumerInputs, [&](OpOperand *operand) {
        auto opResult = operand->get().dyn_cast<OpResult>();
        return opResult && opResult.getOwner() == producer.getOperation();
      }));
  llvm::SetVector<OpOperand *> operandsFromProducerSet(
      operandsFromProducer.begin(), operandsFromProducer.end());

  // 4. Check all the consumer-producer OpOperands' indexing map the same
  AffineMap producerResultIndexMap =
      producer.getIndexingMapMatchingResult(producerResult);
  AffineMap toCheckAffineMap = composeProduceResultAndConsumerOperand(
      producerResultIndexMap, consumer.getMatchingIndexingMap(fusedOperand));
  for (OpOperand *consumerOperand : consumerInputs) {
    if (consumerOperand != fusedOperand &&
        operandsFromProducerSet.contains(consumerOperand)) {
      auto curProducerResult = consumerOperand->get().cast<OpResult>();
      AffineMap curProducerResultIndexMap =
          producer.getIndexingMapMatchingResult(curProducerResult);
      AffineMap toCheckWithAffineMap = composeProduceResultAndConsumerOperand(
          curProducerResultIndexMap,
          consumer.getMatchingIndexingMap(consumerOperand));
      if (toCheckAffineMap != toCheckWithAffineMap)
        return failure();
    }
  }

  // 5. Collect all of the producer inputs
  for (OpOperand *opOperand : producer.getDpsInputOperands()) {
    fusedInputOperands.push_back(opOperand->get());
    AffineMap map = getIndexingMapOfProducerOperandsInCoordinatesOfFusedOp(
        opOperand, producerResultIndexMap,
        consumer.getMatchingIndexingMap(fusedOperand));
    fusedIndexMaps.push_back(map);
  }
  for (OpOperand *opOperand : consumerInputs) {
    if (!operandsFromProducerSet.contains(opOperand)) {
      fusedInputOperands.push_back(opOperand->get());
      fusedIndexMaps.push_back(consumer.getMatchingIndexingMap(opOperand));
    }
  }

  // 6. Collect all of the producer outputs.
  for (const auto &opOperand : llvm::enumerate(producer.getDpsInitOperands())) {
    if (!preservedProducerResults.count(opOperand.index()))
      continue;

    fusedOutputOperands.push_back(opOperand.value()->get());
    AffineMap map = getIndexingMapOfProducerOperandsInCoordinatesOfFusedOp(
        opOperand.value(), producerResultIndexMap,
        consumer.getMatchingIndexingMap(fusedOperand));
    fusedIndexMaps.push_back(map);
    fusedResultTypes.push_back(opOperand.value()->get().getType());
  }

  // 7. All of consumer's output operands (skip operands: added by the builder).
  for (OpOperand *opOperand : consumer.getDpsInitOperands()) {
    fusedOutputOperands.push_back(opOperand->get());
    fusedIndexMaps.push_back(consumer.getMatchingIndexingMap(opOperand));
    Type resultType = opOperand->get().getType();
    if (!resultType.isa<MemRefType>())
      fusedResultTypes.push_back(resultType);
  }

  // Generate the fused op.
  auto fusedOp = rewriter.create<GenericOp>(
      consumer.getLoc(), fusedResultTypes, fusedInputOperands,
      fusedOutputOperands, rewriter.getAffineMapArrayAttr(fusedIndexMaps),
      consumer.getIteratorTypes(),
      /*doc=*/nullptr,
      /*library_call=*/nullptr);
  if (!fusedOp.getShapesToLoopsMap()) {
    // Fused op has invalid indexing maps. Typically this means something is off
    // in the input, but going ahead here would result in verification errors.
    // So cleanup and abort.
    rewriter.eraseOp(fusedOp);
    return rewriter.notifyMatchFailure(
        fusedOp, "fused op failed loop bound computation check");
  }

  // Construct an AffineMap from consumer loops to producer loops.
  // consumer loop -> tensor index
  AffineMap consumerResultIndexMap =
      consumer.getMatchingIndexingMap(fusedOperand);
  // tensor index -> producer loop
  AffineMap invProducerResultIndexMap =
      inversePermutation(producerResultIndexMap);
  assert(invProducerResultIndexMap &&
         "expected producer result indexig map to be invertible");
  // consumer loop -> producer loop
  AffineMap consumerToProducerLoopsMap =
      invProducerResultIndexMap.compose(consumerResultIndexMap);

  generateFusedElementwiseOpRegion(
      rewriter, fusedOp, consumerToProducerLoopsMap, operandsFromProducerSet,
      consumer.getNumLoops(), preservedProducerResults);
  ElementwiseOpFusionResult result;
  result.fusedOp = fusedOp;
  int resultNum = 0;
  for (auto [index, producerResult] : llvm::enumerate(producer->getResults()))
    if (preservedProducerResults.count(index))
      result.replacements[producerResult] = fusedOp->getResult(resultNum++);
  for (auto consumerResult : consumer->getResults())
    result.replacements[consumerResult] = fusedOp->getResult(resultNum++);
  return result;
}

//===---------------------------------------------------------------------===//
// Methods and patterns that fuse reshape ops with elementwise operations by
// expanding the dimensionality of the elementwise operations.
//===---------------------------------------------------------------------===//

/// Conditions for folding a generic operation with a reshape op by expanding
/// the iteration space dimensionality for tensor operations. These are
/// preconditions assumed by `foldReshapeByDimExpansion` which implements the
/// following fusion pattern.
///
///  Consider
///
///  %c = linalg.generic ins(%a, %b : memref<?x?x?xf32>, memref<?x?xf32>)
///         indexing_maps = [affine_map<(d0, d1, d2) -> (d1, d0, d2)>,
///                          affine_map<(d0, d1, d2) -> (d1, d2)>,
///                          affine_map<(d0, d1, d2) -> (d0, d2, d1)>]
///  %d = tensor.expand_shape %c [[0, 1], [2], [3, 4, 5]]
///       : tensor<?x?x?xf32> into tensor<?x?x?x?x?x?xf32>
///
///  The reshape can be folded into the `genericOp` if its loop dimensionality
///  is increased to match the result (operand) of the tensor.expand_shape.
///  The indexing_map of the fused tensor in the `genericOp` and the
///  reassociation map helps compute the indexing maps of the modified op.
///  For the above example, based on the reassociation map it
///  can be concluded that
///
///  - The loop used to access the first dimension of the fused tensor is split
///    into two.
///  - The loop used to access the second dimension of the fused tensor is kept
///    as is.
///  - The loop used to access the third dimension of the fused tensor is split
///    into three.
///
///  i.e. (e0, e1, e2, e3, e4) is the domain of the indexing map of the modified
///  op, then
///
///   d0 -> e0, e1
///   d1 -> e2, e3, e4
///   d2 -> e5
///
///  substituting this, the generic op can be rewritten as
///
///  %d = linalg.generic ins(%0, %1 : )
///        indexing_maps =
///         [affine_map<(e0, e1, e2, e3, e4, e5) -> (e2, e3, e4, e0, e1, e5)>,
///          affine_map<(e0, e1, e2, e3, e4, e5) -> (e2, e3, e4, e5)>,
///          affine_map<(e0, e1, e2, e3, e4, e5) -> (e0, e1, e5, e2, e3, e4)>]
///
///  Since operands to the linalg generic are now 5D, reshapes can be introduced
///  to make it consistent
///
///  %0 = tensor.expand_shape %a [[0, 1, 2], [3, 4], [5]]
///       : tensor<?x?x?xf32> into tensor<?x?x?x?x?x?xf32>
///  %1 = tensor.expand_shape %b [[0, 1, 2], [3]]
///       : tensor<?x?x?xf32> into tensor<?x?x?x?xf32>
///
///  The added reshapes are again expanding patterns, so they will get fused
///  with its producers if possible.
///
/// This is a modification of upstream's version to allow const in affine map
static bool isFusableWithReshapeByDimExpansion(GenericOp genericOp,
                                               OpOperand *fusableOpOperand) {
  // Is fusable only if:
  // - All the indexing maps for operands and results are projected
  //   permutations.
  // - The fused tensor is not a scalar.
  // - All the loops are parallel loops.
  return genericOp.hasTensorSemantics() &&
         llvm::all_of(llvm::enumerate(genericOp.getIndexingMaps().getValue()),
                      [&](auto it) {
                        Attribute attr = it.value();
                        int64_t idx = it.index();
                        AffineMap map = cast<AffineMapAttr>(attr).getValue();
                        // TODO: enhance ExpansionInfo::compute to allow const
                        // in fusableOpOperand's affine map
                        if (idx == fusableOpOperand->getOperandNumber())
                          return map.isProjectedPermutation();
                        else
                          return isProjectedPermutationAndAllowConst(map);
                      }) &&
         genericOp.getMatchingIndexingMap(fusableOpOperand).getNumResults() >
             0 &&
         llvm::all_of(genericOp.getIteratorTypesArray(), isParallelIterator);
}

class ExpansionInfo {
public:
  // Computes the mapping from original dimensions of the op to the dimensions
  // of the expanded op given the `indexingMap` of the fused operand/result of
  // the generic op, the `reassocationMaps` of the reshape op and the shape of
  // the expanded op.
  LogicalResult compute(LinalgOp linalgOp, OpOperand *fusableOpOperand,
                        ArrayRef<AffineMap> reassociationMaps,
                        ArrayRef<int64_t> expandedShape,
                        ArrayRef<int64_t> collapsedShape,
                        PatternRewriter &rewriter);
  unsigned getOrigOpNumDims() const { return reassociation.size(); }
  unsigned getExpandedOpNumDims() const { return expandedOpNumDims; }
  ReassociationIndicesRef getExpandedDims(unsigned i) const {
    return reassociation[i];
  }
  ArrayRef<int64_t> getExpandedShapeOfDim(unsigned i) const {
    return expandedShapeMap[i];
  }
  ArrayRef<int64_t> getOriginalShape() const { return originalLoopExtent; }

private:
  /// Reassociation from the dimensions in the original operation to the
  /// dimension of the expanded operation.
  SmallVector<ReassociationIndices> reassociation;
  /// Mapping from extent of loops in the original operation, to the extent of
  /// loops in the expanded operation.
  SmallVector<SmallVector<int64_t>> expandedShapeMap;
  /// Extent of the loop in the original operation.
  SmallVector<int64_t> originalLoopExtent;
  unsigned expandedOpNumDims;
};

LogicalResult ExpansionInfo::compute(LinalgOp linalgOp,
                                     OpOperand *fusableOpOperand,
                                     ArrayRef<AffineMap> reassociationMaps,
                                     ArrayRef<int64_t> expandedShape,
                                     ArrayRef<int64_t> collapsedShape,
                                     PatternRewriter &rewriter) {
  if (reassociationMaps.empty())
    return failure();
  AffineMap fusedIndexMap = linalgOp.getMatchingIndexingMap(fusableOpOperand);

  SmallVector<int64_t, 4> originalLoopRange = linalgOp.getStaticLoopRanges();
  originalLoopExtent.assign(originalLoopRange.begin(), originalLoopRange.end());

  reassociation.clear();
  expandedShapeMap.clear();
  // Compute the number of dimension in the expanded op that correspond to each
  // dimension of the original op.
  SmallVector<unsigned> numExpandedDims(fusedIndexMap.getNumDims(), 1);
  expandedShapeMap.resize(fusedIndexMap.getNumDims());
  for (const auto &resultExpr : llvm::enumerate(fusedIndexMap.getResults())) {
    unsigned pos = resultExpr.value().cast<AffineDimExpr>().getPosition();
    AffineMap foldedDims = reassociationMaps[resultExpr.index()];
    numExpandedDims[pos] = foldedDims.getNumResults();
    ArrayRef<int64_t> shape =
        expandedShape.slice(foldedDims.getDimPosition(0), numExpandedDims[pos]);
    expandedShapeMap[pos].assign(shape.begin(), shape.end());
  }
  // The remaining dimensions remain the same.
  for (unsigned i : llvm::seq<unsigned>(0, fusedIndexMap.getNumDims()))
    if (expandedShapeMap[i].empty())
      expandedShapeMap[i] = {originalLoopExtent[i]};

  // Compute reassociation map from the original op to the expanded op.
  unsigned sum = 0;
  reassociation.reserve(fusedIndexMap.getNumDims());
  for (const auto &numFoldedDim : llvm::enumerate(numExpandedDims)) {
    auto seq = llvm::seq<int64_t>(sum, sum + numFoldedDim.value());
    reassociation.emplace_back(seq.begin(), seq.end());
    sum += numFoldedDim.value();
  }
  expandedOpNumDims = sum;
  return success();
}

/// Epanding the body of a linalg operation requires adaptations of the accessed
/// loop indices. Specifically, access of indices in the original operation need
/// to be replaced with linearizations of indices in the expanded op. That
/// requires the shape of the expanded dimensions to be static (at least all but
/// the most significant). For now check that these are all statically sized.
/// Note that this could be extended to handle dynamic case, but the
/// implementation below uses `affine.apply` which seems to have issues when the
/// shapes are not static.
static LogicalResult isGenericOpExpandable(GenericOp genericOp,
                                           const ExpansionInfo &expansionInfo,
                                           PatternRewriter &rewriter) {
  if (!genericOp.hasIndexSemantics())
    return success();
  for (unsigned i : llvm::seq<unsigned>(0, expansionInfo.getOrigOpNumDims())) {
    ArrayRef<int64_t> expandedShape = expansionInfo.getExpandedShapeOfDim(i);
    if (expandedShape.size() == 1)
      continue;
    for (int64_t shape : expandedShape.drop_front()) {
      if (ShapedType::isDynamic(shape)) {
        return rewriter.notifyMatchFailure(
            genericOp, "cannot expand due to index semantics and dynamic dims");
      }
    }
  }
  return success();
}

/// Return the indexing map to use in the expanded op for a given the
/// `indexingMap` of the original operation.
///
/// This is a modification of upstream's version to allow const in affine map
static AffineMap
getIndexingMapInExpandedOp(OpBuilder &builder, AffineMap indexingMap,
                           const ExpansionInfo &expansionInfo) {
  SmallVector<AffineExpr> newExprs;
  for (AffineExpr expr : indexingMap.getResults()) {
    if (expr.isa<AffineConstantExpr>()) {
      newExprs.push_back(expr);
      continue;
    }
    unsigned pos = expr.cast<AffineDimExpr>().getPosition();
    SmallVector<AffineExpr, 4> expandedExprs = llvm::to_vector<4>(
        llvm::map_range(expansionInfo.getExpandedDims(pos), [&](int64_t v) {
          return builder.getAffineDimExpr(static_cast<unsigned>(v));
        }));
    newExprs.append(expandedExprs.begin(), expandedExprs.end());
  }
  return AffineMap::get(expansionInfo.getExpandedOpNumDims(),
                        indexingMap.getNumSymbols(), newExprs,
                        builder.getContext());
}

/// Return the type of the operand/result to use in the expanded op given the
/// type in the original op.
///
/// This is a modification of upstream's version to allow const in affine map
static RankedTensorType getExpandedType(RankedTensorType originalType,
                                        AffineMap indexingMap,
                                        const ExpansionInfo &expansionInfo) {
  SmallVector<int64_t> expandedShape;
  for (auto it : llvm::enumerate(indexingMap.getResults())) {
    AffineExpr expr = it.value();
    int64_t idx = it.index();
    if (expr.isa<AffineConstantExpr>()) {
      expandedShape.push_back(originalType.getDimSize(idx));
      continue;
    }
    unsigned dim = expr.cast<AffineDimExpr>().getPosition();
    auto dimExpansion = expansionInfo.getExpandedShapeOfDim(dim);
    expandedShape.append(dimExpansion.begin(), dimExpansion.end());
  }
  return RankedTensorType::get(expandedShape, originalType.getElementType());
}

/// Returns the reassociation maps to use in the `tensor.expand_shape`
/// operation to convert the operands of the original operation to operands of
/// the expanded operation. The same method is used to compute the
/// `tensor.collapse_shape` used to collapse the result of the expanded
/// op to get the value that can replace all uses of the results of the original
/// op.
///
/// This is a modification of upstream's version to allow const in affine map
static SmallVector<ReassociationIndices>
getReassociationForExpansion(AffineMap indexingMap,
                             const ExpansionInfo &expansionInfo) {
  SmallVector<ReassociationIndices> reassociation;
  unsigned numReshapeDims = 0;
  for (AffineExpr expr : indexingMap.getResults()) {
    if (expr.isa<AffineConstantExpr>()) {
      reassociation.push_back({numReshapeDims++});
      continue;
    }
    unsigned dim = expr.cast<AffineDimExpr>().getPosition();
    auto numExpandedDims = expansionInfo.getExpandedDims(dim).size();
    SmallVector<int64_t, 2> indices = llvm::to_vector<2>(
        llvm::seq<int64_t>(numReshapeDims, numReshapeDims + numExpandedDims));
    reassociation.emplace_back(std::move(indices));
    numReshapeDims += numExpandedDims;
  }
  return reassociation;
}

/// Update the body of an expanded linalg operation having index semantics. The
/// indices of the original operation need to be recovered by linearizing the
/// indices of the correspoding dimensions of the expanded operation. For now it
/// is assumed that the shapes of the expanded operation needed for
/// linearization are static.
static void updateExpandedGenericOpRegion(PatternRewriter &rewriter,
                                          Location loc, Region &fusedRegion,
                                          const ExpansionInfo &expansionInfo) {
  // Replace the original indices by the linearization of the expanded indices.
  for (IndexOp indexOp :
       llvm::make_early_inc_range(fusedRegion.front().getOps<IndexOp>())) {
    ArrayRef<int64_t> expandedDims =
        expansionInfo.getExpandedDims(indexOp.getDim());
    assert(!expandedDims.empty() && "expected valid expansion info");

    // Skip index operations that are not affected by the expansion.
    if (expandedDims.size() == 1 &&
        expandedDims.front() == (int64_t)indexOp.getDim())
      continue;

    // Linearize the expanded indices of the original index dimension.
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointAfter(indexOp);
    ArrayRef<int64_t> expandedDimsShape =
        expansionInfo.getExpandedShapeOfDim(indexOp.getDim()).drop_front();
    SmallVector<Value> expandedIndices;
    expandedIndices.reserve(expandedDims.size() - 1);
    llvm::transform(
        expandedDims.drop_front(), std::back_inserter(expandedIndices),
        [&](int64_t dim) { return rewriter.create<IndexOp>(loc, dim); });
    Value newIndex = rewriter.create<IndexOp>(loc, expandedDims.front());
    for (auto it : llvm::zip(expandedDimsShape, expandedIndices)) {
      assert(!ShapedType::isDynamic(std::get<0>(it)));
      AffineExpr idx, acc;
      bindDims(rewriter.getContext(), idx, acc);
      newIndex = rewriter.create<affine::AffineApplyOp>(
          indexOp.getLoc(), idx + acc * std::get<0>(it),
          ValueRange{std::get<1>(it), newIndex});
    }
    rewriter.replaceOp(indexOp, newIndex);
  }
}

/// Implements the fusion of a tensor.collapse_shape or a tensor.expand_shape op
/// and a generic op as explained in `isFusableWithReshapeByExpansion`. Assumes
/// that those conditions have been satisfied.
static std::optional<SmallVector<Value>>
fuseWithReshapeByExpansion(GenericOp genericOp, Operation *reshapeOp,
                           OpOperand *fusableOpOperand,
                           PatternRewriter &rewriter) {
  assert(isFusableWithReshapeByDimExpansion(genericOp, fusableOpOperand) &&
         "preconditions for fuse operation failed");
  // Check if reshape is expanding or collapsing.
  auto expandingReshapeOp = dyn_cast<tensor::ExpandShapeOp>(*reshapeOp);
  auto collapsingReshapeOp = dyn_cast<tensor::CollapseShapeOp>(*reshapeOp);
  bool isExpanding = (expandingReshapeOp != nullptr);
  RankedTensorType expandedType = isExpanding
                                      ? expandingReshapeOp.getResultType()
                                      : collapsingReshapeOp.getSrcType();
  RankedTensorType collapsedType = isExpanding
                                       ? expandingReshapeOp.getSrcType()
                                       : collapsingReshapeOp.getResultType();

  ExpansionInfo expansionInfo;
  if (failed(expansionInfo.compute(
          genericOp, fusableOpOperand,
          isExpanding ? expandingReshapeOp.getReassociationMaps()
                      : collapsingReshapeOp.getReassociationMaps(),
          expandedType.getShape(), collapsedType.getShape(), rewriter)))
    return std::nullopt;

  if (failed(isGenericOpExpandable(genericOp, expansionInfo, rewriter)))
    return std::nullopt;

  SmallVector<AffineMap, 4> expandedOpIndexingMaps = llvm::to_vector<4>(
      llvm::map_range(genericOp.getIndexingMapsArray(), [&](AffineMap m) {
        return getIndexingMapInExpandedOp(rewriter, m, expansionInfo);
      }));

  // Set insertion point to the generic op.
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(genericOp);

  SmallVector<Value> expandedOpOperands;
  expandedOpOperands.reserve(genericOp.getNumDpsInputs());
  for (OpOperand *opOperand : genericOp.getDpsInputOperands()) {
    if (opOperand == fusableOpOperand) {
      expandedOpOperands.push_back(isExpanding ? expandingReshapeOp.getSrc()
                                               : collapsingReshapeOp.getSrc());
      continue;
    }
    if (auto opOperandType =
            dyn_cast<RankedTensorType>(opOperand->get().getType())) {
      AffineMap indexingMap = genericOp.getMatchingIndexingMap(opOperand);
      RankedTensorType expandedOperandType =
          getExpandedType(opOperandType, indexingMap, expansionInfo);
      if (expandedOperandType != opOperand->get().getType()) {
        // Reshape the operand to get the right type.
        SmallVector<ReassociationIndices> reassociation =
            getReassociationForExpansion(indexingMap, expansionInfo);
        if (failed(reshapeLikeShapesAreCompatible(
                [&](const Twine &msg) {
                  return rewriter.notifyMatchFailure(genericOp, msg);
                },
                opOperandType.getShape(), expandedOperandType.getShape(),
                reassociation,
                /*isExpandingReshape=*/true)))
          return std::nullopt;
        expandedOpOperands.push_back(rewriter.create<tensor::ExpandShapeOp>(
            genericOp.getLoc(), expandedOperandType, opOperand->get(),
            reassociation));
        continue;
      }
    }
    expandedOpOperands.push_back(opOperand->get());
  }

  Location loc = genericOp.getLoc();
  SmallVector<Value> outputs;
  for (OpOperand *opOperand : genericOp.getDpsInitOperands()) {
    AffineMap indexingMap = genericOp.getMatchingIndexingMap(opOperand);
    auto opOperandType = cast<RankedTensorType>(opOperand->get().getType());
    RankedTensorType expandedOutputType =
        getExpandedType(opOperandType, indexingMap, expansionInfo);
    if (expandedOutputType != opOperand->get().getType()) {
      SmallVector<ReassociationIndices> reassociation =
          getReassociationForExpansion(indexingMap, expansionInfo);
      if (failed(reshapeLikeShapesAreCompatible(
              [&](const Twine &msg) {
                return rewriter.notifyMatchFailure(genericOp, msg);
              },
              opOperandType.getShape(), expandedOutputType.getShape(),
              reassociation,
              /*isExpandingReshape=*/true)))
        return std::nullopt;
      outputs.push_back(rewriter.create<tensor::ExpandShapeOp>(
          genericOp.getLoc(), expandedOutputType, opOperand->get(),
          reassociation));
    } else {
      outputs.push_back(opOperand->get());
    }
  }

  // The iterator types of the expanded op are all parallel.
  SmallVector<utils::IteratorType> iteratorTypes(
      expansionInfo.getExpandedOpNumDims(), utils::IteratorType::parallel);

  TypeRange resultTypes = ValueRange(outputs).getTypes();
  auto fusedOp =
      rewriter.create<GenericOp>(genericOp.getLoc(), resultTypes,
                                 /*inputs=*/expandedOpOperands, outputs,
                                 expandedOpIndexingMaps, iteratorTypes);
  Region &fusedRegion = fusedOp->getRegion(0);
  Region &originalRegion = genericOp->getRegion(0);
  rewriter.cloneRegionBefore(originalRegion, fusedRegion, fusedRegion.begin());

  // Update the index accesses after the expansion.
  updateExpandedGenericOpRegion(rewriter, loc, fusedRegion, expansionInfo);

  // Reshape the result values to their original shape if this is a collapsing
  // reshape folded into its consumer.
  SmallVector<Value> resultVals;
  for (OpResult opResult : genericOp->getOpResults()) {
    int64_t resultNumber = opResult.getResultNumber();
    if (resultTypes[resultNumber] != opResult.getType()) {
      SmallVector<ReassociationIndices> reassociation =
          getReassociationForExpansion(
              genericOp.getMatchingIndexingMap(
                  genericOp.getDpsInitOperand(resultNumber)),
              expansionInfo);
      resultVals.push_back(rewriter.create<tensor::CollapseShapeOp>(
          genericOp.getLoc(), opResult.getType(),
          fusedOp->getResult(resultNumber), reassociation));
    } else {
      resultVals.push_back(fusedOp->getResult(resultNumber));
    }
  }
  // Assuming a single result.
  return resultVals;
}

/// Pattern to fuse a tensor.collapse_shape op with its consumer generic op,
/// when the reshape op is collapsing dimensions. The dimensionality of the loop
/// in the consumer is expanded.
class FoldWithProducerReshapeOpByExpansionExt
    : public OpRewritePattern<GenericOp> {
public:
  FoldWithProducerReshapeOpByExpansionExt(MLIRContext *context,
                                          ControlFusionFn foldReshapes,
                                          PatternBenefit benefit = 1)
      : OpRewritePattern<GenericOp>(context, benefit),
        controlFoldingReshapes(std::move(foldReshapes)) {}

  LogicalResult matchAndRewrite(GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    for (OpOperand *opOperand : genericOp.getDpsInputOperands()) {
      tensor::CollapseShapeOp reshapeOp =
          opOperand->get().getDefiningOp<tensor::CollapseShapeOp>();
      if (!reshapeOp)
        continue;
      // Fold only if
      // - The tensor reshape op is folding.
      // - All constraints of fusing with reshape by expansion are met.
      if (!isFusableWithReshapeByDimExpansion(genericOp, opOperand) ||
          (!controlFoldingReshapes(opOperand))) {
        continue;
      }

      std::optional<SmallVector<Value>> replacementValues =
          fuseWithReshapeByExpansion(genericOp, reshapeOp, opOperand, rewriter);
      if (!replacementValues) {
        return failure();
      }
      rewriter.replaceOp(genericOp, *replacementValues);
      return success();
    }
    return failure();
  }

private:
  ControlFusionFn controlFoldingReshapes;
};

/// Patterns to fuse a generic op, with the producer of its operands.
class FuseElementwiseProducerConsumerExt : public OpRewritePattern<GenericOp> {
public:
  FuseElementwiseProducerConsumerExt(MLIRContext *context, bool diffShapes,
                                     ControlFusionFn fun, DominanceInfo &dom,
                                     PostDominanceInfo &post,
                                     PatternBenefit benefit = 1)
      : OpRewritePattern<GenericOp>(context, benefit),
        enableDiffShapes(diffShapes), controlFn(std::move(fun)), domInfo(dom),
        postDomInfo(post) {}

  LogicalResult matchAndRewrite(GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    if (isBroadcast(genericOp)) {
      return failure();
    }

    // Find the first operand that is defined by another generic op on tensors.
    for (OpOperand &opOperand : genericOp->getOpOperands()) {
      if (!isProducerElementwiseOpFusable(&opOperand)) {
        continue;
      }

      if (!controlFn(&opOperand)) {
        continue;
      }

      if (!enableDiffShapes && !areShapeEqual(genericOp, &opOperand) &&
          isUsedByReturn(&opOperand)) {
        continue;
      }

      FailureOr<ElementwiseOpFusionResult> fusionResult =
          fuseElementwiseOpsExt(rewriter, &opOperand);

      if (failed(fusionResult)) {
        continue;
      }

      if (succeeded(fusionResult)) {
        // FIXME(lyq): applyPatternsAndFoldGreedily only trigger dce between
        // patterns, so producer doesn't be erased when genericOp is replaced.
        // If applyPatternsAndFoldGreedily's behavior changed, we should fix
        // this.
        auto producer = opOperand.get().getDefiningOp<GenericOp>();

        Operation *fusedOp = fusionResult->fusedOp;
        auto replacements =
            fusedOp->getResults().take_back(genericOp.getNumResults());
        rewriter.replaceOp(genericOp, replacements);

        // Try to replace the producer with fusedOp
        // if fusedOp dominates all of the producer's users
        unsigned idx = 0;
        for (auto &&res : producer.getResults()) {
          // only check used producer results
          if (res.use_empty()) {
            continue;
          }

          hoistDownDescendantUsers(res, postDomInfo, /*checkOperand=*/false);

          if (!checkDominateAllUsers(fusedOp, res, domInfo)) {
            idx++;
            continue;
          }
          res.replaceAllUsesWith(fusedOp->getResult(idx++));
        }
        return success();
      }
    }
    return failure();
  }

private:
  bool enableDiffShapes;
  ControlFusionFn controlFn;
  DominanceInfo &domInfo;
  PostDominanceInfo &postDomInfo;
};

class InsertLinalgExtAlias : public OpRewritePattern<GenericOp> {
public:
  InsertLinalgExtAlias(MLIRContext *context, DominanceInfo &dom,
                       PatternBenefit benefit = 1)
      : OpRewritePattern<GenericOp>(context, benefit), domInfo(dom) {}

  LogicalResult matchAndRewrite(GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    // Find the first operand that is used by another generic op on tensors.
    // Note this will be triggered only when producer and consumer fusion
    // is not found

    if (genericOp->hasAttr(getInternalAliasAttrName()))
      return failure();

    for (OpOperand *opOperand : genericOp.getDpsInputOperands()) {
      auto val = opOperand->get();
      // skip producer consumer
      if (nullptr != val.getDefiningOp<GenericOp>())
        continue;

      SmallVector<Operation *> users;
      SmallPtrSet<Operation *, 8> userSet;

      for (auto &use : val.getUses()) {
        auto consumerGenericOp = dyn_cast<GenericOp>(use.getOwner());
        if (!consumerGenericOp)
          continue;

        if (consumerGenericOp.isDpsInput(&use) &&
            !userSet.contains(use.getOwner())) {
          users.push_back(use.getOwner());
          userSet.insert(use.getOwner());
        }
      }

      // users.size == 1
      // meaning only found genericOp itself
      if (users.size() == 1) {
        continue;
      }

      // find hoist location
      auto firstOp = leastProperlyDominantOp(users, domInfo);
      rewriter.setInsertionPoint(firstOp);
      auto aliasGeneric = getAliasGeneric(rewriter, val);

      if (failed(aliasGeneric)) {
        continue;
      }

      // replace all generic users except the aliasGeneric itself
      val.replaceUsesWithIf(aliasGeneric->getResult(0),
                            [&](OpOperand &opOperand) {
                              return isa<GenericOp>(opOperand.getOwner()) &&
                                     opOperand.getOwner() != *aliasGeneric;
                            });

      return success();
    }

    return failure();
  }

private:
  DominanceInfo &domInfo;
};

class RemoveLinalgExtAlias : public OpRewritePattern<linalg_ext::AliasOp> {
public:
  RemoveLinalgExtAlias(MLIRContext *context, PatternBenefit benefit = 1)
      : OpRewritePattern<linalg_ext::AliasOp>(context, benefit) {}

  LogicalResult matchAndRewrite(linalg_ext::AliasOp aliasOp,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOp(aliasOp, aliasOp.getOperand());
    return success();
  }
};

} // namespace

// this is simply a renamed function from areElementwiseOpsFusable
// to isProducerElementwiseOpFusable for more readibility
bool mlir::linalg::isProducerElementwiseOpFusable(OpOperand *fusedOperand) {
  return areElementwiseOpsFusable(fusedOperand);
}

void mlir::linalg::populateElementwiseOpsProducerConsumerFusionPatterns(
    RewritePatternSet &patterns, bool diffShapes,
    const ControlFusionFn &controlElementwiseOpsFusion, DominanceInfo &dom,
    PostDominanceInfo &postDomInfo) {
  auto *context = patterns.getContext();
  patterns.add<FuseElementwiseProducerConsumerExt>(
      context, diffShapes, controlElementwiseOpsFusion, dom, postDomInfo);

  // include the default populateElementwiseOpsFusionPatterns
  // but disable FuseElementwiseOps by passing a always-false ControlFusionFn
  // meaning the default FuseElementwiseOps won't be used
  populateElementwiseOpsFusionPatterns(
      patterns, [](OpOperand *fusedOperand) { return false; });
}

void mlir::linalg_ext::populateInsertLinalgExtAliasForSharedInputFusionPatterns(
    RewritePatternSet &patterns, DominanceInfo &dom) {
  auto *context = patterns.getContext();
  patterns.add<InsertLinalgExtAlias>(context, dom);

  // include the default populateElementwiseOpsFusionPatterns
  // but disable FuseElementwiseOps by passing a always-false ControlFusionFn
  // meaning the default FuseElementwiseOps won't be used
  populateElementwiseOpsFusionPatterns(
      patterns, [](OpOperand *fusedOperand) { return false; });
}

void mlir::linalg_ext::populateRemoveLinalgExtAliasPattern(
    RewritePatternSet &patterns) {
  auto *context = patterns.getContext();
  patterns.add<RemoveLinalgExtAlias>(context);
}

namespace {
static void populateCommonPatterns(RewritePatternSet &patterns,
                                   const ControlFusionFn &controlFn) {
  MLIRContext *context = patterns.getContext();
  populateFoldReshapeOpsByExpansionPatterns(patterns, controlFn);
  patterns.add<FoldWithProducerReshapeOpByExpansionExt>(patterns.getContext(),
                                                        controlFn);

  // convert a MapOp-to-GenericOp pattern to extend fusion possibility
  populateMapOpToGenericPattern(patterns);

  // General canonicalization patterns.
  affine::AffineApplyOp::getCanonicalizationPatterns(patterns, context);
  GenericOp::getCanonicalizationPatterns(patterns, context);
  tensor::ExpandShapeOp::getCanonicalizationPatterns(patterns, context);
  tensor::CollapseShapeOp::getCanonicalizationPatterns(patterns, context);
  context->getLoadedDialect<LinalgDialect>()->getCanonicalizationPatterns(
      patterns);

  // Add constant folding patterns.
  populateConstantFoldLinalgOperations(patterns, controlFn);
}

struct LinalgElementwiseFusionExtPass
    : public LinalgElementwiseFusionExtBase<LinalgElementwiseFusionExtPass> {
  LinalgElementwiseFusionExtPass(bool sharedInput, bool diffShapes)
      : LinalgElementwiseFusionExtBase() {
    this->enableSharedInput = sharedInput;
    this->enableDiffShapes = diffShapes;
    controlFn = [](OpOperand *fusedOperand) {
      Operation *producer = fusedOperand->get().getDefiningOp();
      return producer != nullptr;
    };
  }

  LinalgElementwiseFusionExtPass(linalg::ControlFusionFn controlFunc,
                                 bool sharedInput, bool diffShapes) {
    this->enableSharedInput = sharedInput;
    this->enableDiffShapes = diffShapes;
    controlFn = std::move(controlFunc);
  }

  void runOnOperation() override {
    Operation *op = getOperation();
    MLIRContext *context = op->getContext();

    auto &domInfo = getAnalysis<DominanceInfo>();
    auto &postDomInfo = getAnalysis<PostDominanceInfo>();

    // simplify Tensor DimOp first
    // simplifyTensorDimOpUsedInLinalgWithinOp(*op);

    // do producer-consumer fusion first
    {
      RewritePatternSet patterns(context);
      // Add elementwise op fusion patterns.
      populateElementwiseOpsProducerConsumerFusionPatterns(
          patterns, enableDiffShapes, controlFn, domInfo, postDomInfo);

      populateCommonPatterns(patterns, controlFn);

      // Use TopDownTraversal for compile time reasons
      GreedyRewriteConfig grc;
      grc.useTopDownTraversal = true;
      (void)applyPatternsAndFoldGreedily(op, std::move(patterns), grc);
    }

    // do shared-input fusion
    if (enableSharedInput) {
      RewritePatternSet patterns(context);
      // Add elementwise op fusion patterns.
      populateInsertLinalgExtAliasForSharedInputFusionPatterns(patterns,
                                                               domInfo);

      // Use TopDownTraversal for compile time reasons
      GreedyRewriteConfig grc;
      grc.useTopDownTraversal = true;
      (void)applyPatternsAndFoldGreedily(op, std::move(patterns), grc);
    }

    // do producer-consumer fusion again
    if (enableSharedInput) {
      RewritePatternSet patterns(context);
      auto alwayTrueControlFn = [](OpOperand *fusedOperand) { return true; };
      // Add elementwise op fusion patterns.
      populateElementwiseOpsProducerConsumerFusionPatterns(
          patterns, enableDiffShapes, alwayTrueControlFn, domInfo, postDomInfo);
      populateCommonPatterns(patterns, alwayTrueControlFn);

      // Use TopDownTraversal for compile time reasons
      GreedyRewriteConfig grc;
      grc.useTopDownTraversal = true;
      (void)applyPatternsAndFoldGreedily(op, std::move(patterns), grc);
    }

    // clean up is only sharedInput
    if (enableSharedInput) {
      RewritePatternSet patterns(context);
      populateRemoveLinalgExtAliasPattern(patterns);
      GreedyRewriteConfig grc;
      grc.useTopDownTraversal = true;
      (void)applyPatternsAndFoldGreedily(op, std::move(patterns), grc);
    }

    // simplify Tensor DimOp in the end too
    simplifyTensorDimOpUsedInLinalgWithinOp(*op);
  }

  ControlFusionFn controlFn;
};
} // namespace

std::unique_ptr<Pass>
mlir::createLinalgElementwiseFusionExtPass(bool enableSharedInput,
                                           bool enableDiffShapes) {
  return std::make_unique<LinalgElementwiseFusionExtPass>(enableSharedInput,
                                                          enableDiffShapes);
}

std::unique_ptr<Pass> mlir::createLinalgElementwiseFusionExtPass(
    const linalg::ControlFusionFn &controlElementwiseOpFusion,
    bool enableSharedInput, bool enableDiffShapes) {
  return std::make_unique<LinalgElementwiseFusionExtPass>(
      controlElementwiseOpFusion, enableSharedInput, enableDiffShapes);
}
