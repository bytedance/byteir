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
#include "byteir/Utils/Hoist.h"
#include "byteir/Utils/TypeUtils.h"
#include "byteir/Utils/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

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
