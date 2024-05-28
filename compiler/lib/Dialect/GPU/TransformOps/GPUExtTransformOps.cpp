//===-- GPUExtTransformOps.cpp ------------------------------------------===//
//
// Copyright 2024 ByteDance Ltd. and/or its affiliates. All rights reserved.
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
// Some code comes from GPUTransformOps.cpp in LLVM project
// Original license:
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/GPU/TransformOps/GPUExtTransformOps.h"
#include "byteir/Dialect/GPU/TransformOps/Utils.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/TransformOps/Utils.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/DeviceMappingInterface.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include <type_traits>

using namespace mlir;
using namespace mlir::gpu;
using namespace mlir::transform;
using namespace mlir::transform::gpu_ext;

#define DEBUG_TYPE "gpu-transforms"
#define DEBUG_TYPE_ALIAS "gpu-transforms-alias"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")
#define DBGS_ALIAS() (llvm::dbgs() << '[' << DEBUG_TYPE_ALIAS << "] ")

//===----------------------------------------------------------------------===//
// Block and thread mapping utilities.
//===----------------------------------------------------------------------===//

namespace {
/// Local types used for mapping verification.
struct MappingKind {};
struct BlockMappingKind : MappingKind {};
struct ThreadMappingKind : MappingKind {};

ParseResult parseI64ArrayWithDynamic(OpAsmParser &parser,
                                     DenseI64ArrayAttr &integers) {
  SmallVector<int64_t, 4> integerVals;
  auto parseInteger = [&]() {
    int64_t integer;
    if (succeeded(parser.parseOptionalKeyword("kDynamic"))) {
      integer = ShapedType::kDynamic;
    } else {
      if (failed(parser.parseInteger<int64_t>(integer)))
        return failure();
    }
    integerVals.push_back(integer);
    return success();
  };

  if (parser.parseCommaSeparatedList(OpAsmParser::Delimiter::Square,
                                     parseInteger))
    return failure();
  integers = parser.getBuilder().getDenseI64ArrayAttr(integerVals);
  return success();
}

void printI64ArrayWithDynamic(OpAsmPrinter &printer,
                              ArrayRef<int64_t> integers) {
  printer << '[';
  if (integers.empty()) {
    printer << "]";
    return;
  }

  llvm::interleaveComma(integers, printer, [&](int64_t integer) {
    if (integer == ShapedType::kDynamic) {
      printer << "kDynamic";
    } else {
      printer << integer;
    }
  });
  printer << ']';
}

} // namespace

static DiagnosedSilenceableFailure
definiteFailureHelper(std::optional<TransformOpInterface> transformOp,
                      Operation *target, const Twine &message) {
  if (transformOp.has_value())
    return transformOp->emitDefiniteFailure() << message;
  return emitDefiniteFailure(target, message);
}

/// Check if given mapping attributes are one of the desired attributes
template <typename MappingKindType>
static DiagnosedSilenceableFailure
checkMappingAttributeTypes(std::optional<TransformOpInterface> transformOp,
                           scf::ForallOp forallOp) {
  if (!forallOp.getMapping().has_value()) {
    return definiteFailureHelper(transformOp, forallOp,
                                 "scf.forall op requires a mapping attribute");
  }

  bool hasBlockMapping =
      llvm::any_of(forallOp.getMapping().value(), [](Attribute attr) {
        return isa<GPUBlockMappingAttr>(attr);
      });
  bool hasWarpgroupMapping =
      llvm::any_of(forallOp.getMapping().value(), [](Attribute attr) {
        return isa<GPUWarpgroupMappingAttr>(attr);
      });
  bool hasWarpMapping =
      llvm::any_of(forallOp.getMapping().value(), [](Attribute attr) {
        return isa<GPUWarpMappingAttr>(attr);
      });
  bool hasThreadMapping =
      llvm::any_of(forallOp.getMapping().value(), [](Attribute attr) {
        return isa<GPUThreadMappingAttr>(attr);
      });
  int64_t countMappingTypes = 0;
  countMappingTypes += hasBlockMapping ? 1 : 0;
  countMappingTypes += hasWarpgroupMapping ? 1 : 0;
  countMappingTypes += hasWarpMapping ? 1 : 0;
  countMappingTypes += hasThreadMapping ? 1 : 0;
  if (countMappingTypes > 1) {
    return definiteFailureHelper(
        transformOp, forallOp,
        "cannot mix different mapping types, use nesting");
  }
  if (std::is_same<MappingKindType, BlockMappingKind>::value &&
      !hasBlockMapping) {
    return definiteFailureHelper(
        transformOp, forallOp,
        "scf.forall op requires a mapping attribute of kind 'block'");
  }
  if (std::is_same<MappingKindType, ThreadMappingKind>::value &&
      !hasThreadMapping && !hasWarpMapping && !hasWarpgroupMapping) {
    return definiteFailureHelper(transformOp, forallOp,
                                 "scf.forall op requires a mapping attribute "
                                 "of kind 'thread' or 'warp'");
  }

  DenseSet<Attribute> seen;
  for (Attribute map : forallOp.getMapping()->getValue()) {
    if (seen.contains(map)) {
      return definiteFailureHelper(
          transformOp, forallOp,
          "duplicate attribute, cannot map different loops "
          "to the same mapping id");
    }
    seen.insert(map);
  }

  auto isLinear = [](Attribute a) {
    return cast<DeviceMappingAttrInterface>(a).isLinearMapping();
  };
  if (llvm::any_of(forallOp.getMapping()->getValue(), isLinear) &&
      !llvm::all_of(forallOp.getMapping()->getValue(), isLinear)) {
    return definiteFailureHelper(
        transformOp, forallOp,
        "cannot mix linear and non-linear mapping modes");
  }

  return DiagnosedSilenceableFailure::success();
}

template <typename MappingKindType>
static DiagnosedSilenceableFailure
verifyGpuMapping(std::optional<TransformOpInterface> transformOp,
                 scf::ForallOp forallOp) {
  // Check the types of the mapping attributes match.
  DiagnosedSilenceableFailure typeRes =
      checkMappingAttributeTypes<MappingKindType>(transformOp, forallOp);
  if (!typeRes.succeeded())
    return typeRes;

  // Perform other non-types verifications.
  if (!forallOp.isNormalized())
    return definiteFailureHelper(transformOp, forallOp,
                                 "unsupported non-normalized loops");
  if (forallOp.getNumResults() > 0)
    return definiteFailureHelper(transformOp, forallOp,
                                 "only bufferized scf.forall can be mapped");
  bool useLinearMapping = cast<DeviceMappingAttrInterface>(
                              forallOp.getMapping()->getValue().front())
                              .isLinearMapping();
  // TODO: This would be more natural with support for Optional<EnumParameter>
  // in GPUDeviceMappingAttr.
  int64_t maxNumMappingsSupported =
      useLinearMapping ? (getMaxEnumValForMappingId() -
                          static_cast<uint64_t>(MappingId::DimZ))
                       : 3;
  if (forallOp.getRank() > maxNumMappingsSupported) {
    return definiteFailureHelper(transformOp, forallOp,
                                 "scf.forall with rank > ")
           << maxNumMappingsSupported
           << " does not lower for the specified mapping attribute type";
  }
  return DiagnosedSilenceableFailure::success();
}

/// Struct to return the result of the rewrite of a forall operation.
struct ForallRewriteResultExt {
  SmallVector<Value> mappingSizes;
  SmallVector<Value> mappingIds;
  OpFoldResult totalMappingSize;
};

/// Helper to replace ids of dimensions known to be 1 by 0 to simplify the IR.
template <typename OpTy, typename OperationOrBlock>
static void
replaceUnitMappingIdsHelper(RewriterBase &rewriter, Location loc,
                            OperationOrBlock *parent, Value replacement,
                            ArrayRef<int64_t> availableMappingSizes) {
  parent->walk([&](OpTy idOp) {
    if (availableMappingSizes[static_cast<int64_t>(idOp.getDimension())] == 1)
      rewriter.replaceAllUsesWith(idOp.getResult(), replacement);
  });
}

static std::optional<int64_t> getAsIndex(Value val) {
  if (!val) {
    return std::nullopt;
  }

  OpFoldResult ofr = getAsOpFoldResult(val);
  return getConstantIntValue(ofr);
}

// get mapping sizes of forallOp.
// if trip count is dynamic, the corresponding
// value is calculated by (upperbound - lowerbound).celldiv(step)
static SmallVector<OpFoldResult> getMappingSizes(RewriterBase &rewriter,
                                                 scf::ForallOp forallOp) {
  auto upperBoundArr = forallOp.getMixedUpperBound();
  auto LowerBoundArr = forallOp.getMixedLowerBound();
  auto stepArr = forallOp.getMixedStep();

  SmallVector<OpFoldResult> ret;
  int64_t idx = 0;
  for (auto [lb, ub, step] :
       llvm::zip_equal(LowerBoundArr, upperBoundArr, stepArr)) {
    auto mayCstLb = getConstantIntValue(lb);
    auto mayCstUb = getConstantIntValue(ub);
    auto mayCstStep = getConstantIntValue(step);
    if (!mayCstLb.has_value() || !mayCstUb.has_value() ||
        !mayCstStep.has_value()) {
      Value span = rewriter.create<arith::SubIOp>(
          forallOp.getLoc(), forallOp.getUpperBound(rewriter)[idx],
          forallOp.getLowerBound(rewriter)[idx]);
      Value tripCnt = rewriter.create<arith::CeilDivSIOp>(
          forallOp.getLoc(), span, forallOp.getStep(rewriter)[idx]);
      ret.emplace_back(tripCnt);
    } else {
      int64_t loopSpan = mayCstUb.value() - mayCstLb.value();
      int64_t tripCnt =
          (loopSpan + mayCstStep.value() - 1) / mayCstStep.value();
      ret.emplace_back(getAsIndexOpFoldResult(rewriter.getContext(), tripCnt));
    }
    idx += 1;
  }
  return ret;
}

static Value getDimensionSizeOfId(RewriterBase &rewriter, Value id) {
  if (auto threadId = id.getDefiningOp<ThreadIdOp>()) {
    return rewriter.create<BlockDimOp>(
        threadId.getLoc(), rewriter.getIndexType(), threadId.getDimension());
  } else if (auto blockId = id.getDefiningOp<BlockIdOp>()) {
    return rewriter.create<GridDimOp>(blockId.getLoc(), rewriter.getIndexType(),
                                      blockId.getDimension());
  }
  return Value();
}

static DiagnosedSilenceableFailure rewriteOneForallCommonImpl(
    RewriterBase &rewriter, std::optional<TransformOpInterface> transformOp,
    scf::ForallOp forallOp, ArrayRef<OpFoldResult> availableMappingSizes,
    ForallRewriteResultExt &result, const GpuIdBuilderExt &gpuIdBuilder) {
  LDBG("--start rewriteOneForallCommonImpl");

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(forallOp);

  // Step 1. Complete the mapping to a full mapping (with 1s) if necessary.
  auto numParallelIterations =
      getConstantIntValues(forallOp.getMixedUpperBound());

  assert(forallOp.isNormalized() && "requires normalized forall op");

  SmallVector<OpFoldResult> tmpMappingSizes =
      getMappingSizes(rewriter, forallOp);
  SetVector<Attribute> forallMappingAttrs;
  forallMappingAttrs.insert(forallOp.getMapping()->getValue().begin(),
                            forallOp.getMapping()->getValue().end());
  auto comparator = [](Attribute a, Attribute b) -> bool {
    return cast<DeviceMappingAttrInterface>(a).getMappingId() <
           cast<DeviceMappingAttrInterface>(b).getMappingId();
  };

  // Step 1.b. In the linear case, compute the max mapping to avoid needlessly
  // mapping all dimensions. In the 3-D mapping case we need to map all
  // dimensions.
  DeviceMappingAttrInterface maxMapping =
      cast<DeviceMappingAttrInterface>(*std::max_element(
          forallMappingAttrs.begin(), forallMappingAttrs.end(), comparator));
  DeviceMappingAttrInterface maxLinearMapping;
  if (maxMapping.isLinearMapping())
    maxLinearMapping = maxMapping;
  for (auto attr : gpuIdBuilder.mappingAttributes) {
    // If attr overflows, just skip.
    if (maxLinearMapping && comparator(maxLinearMapping, attr))
      continue;
    // Try to insert. If element was already present, just continue.
    if (!forallMappingAttrs.insert(attr))
      continue;
    // Otherwise, we have a new insertion without a size -> use size 1.
    tmpMappingSizes.push_back(getAsIndexOpFoldResult(rewriter.getContext(), 1));
  }

  // Step 2. sort the values by the corresponding DeviceMappingAttrInterface.
  SmallVector<OpFoldResult> forallMappingSizes = getValuesSortedByKey(
      forallMappingAttrs.getArrayRef(), tmpMappingSizes, comparator);

  // Step 3. Generate the mappingIdOps using the provided generator.
  Location loc = forallOp.getLoc();
  SmallVector<OpFoldResult> originalBasis(availableMappingSizes);
  bool originalBasisWasProvided = !originalBasis.empty();
  if (!originalBasisWasProvided) {
    originalBasis = forallMappingSizes;
    while (originalBasis.size() < 3)
      originalBasis.push_back(getAsIndexOpFoldResult(rewriter.getContext(), 1));
  }

  IdBuilderResultExt builderResult =
      gpuIdBuilder.idBuilder(rewriter, loc, forallMappingSizes, originalBasis);

  // Step 4. Map the induction variables to the mappingIdOps, this may involve
  // a permutation.
  SmallVector<Value> mappingIdOps = builderResult.mappingIdOps;
  IRMapping bvm;
  for (auto [iv, dim] : llvm::zip_equal(
           forallOp.getInductionVars(),
           forallMappingAttrs.getArrayRef().take_front(forallOp.getRank()))) {
    auto mappingAttr = cast<DeviceMappingAttrInterface>(dim);
    Value peIdOp = mappingIdOps[mappingAttr.getRelativeIndex()];
    bvm.map(iv, peIdOp);
  }

  // Step 5. If the originalBasis is already known, create conditionals to
  // predicate the region. Otherwise, the current forall determines the
  // originalBasis and no predication occurs.
  Value predicate;

  // insertionPoint and  targetBlock for forall's body
  Block *targetBlock = forallOp->getBlock();
  Block::iterator insertionPoint = rewriter.getInsertionPoint();

  rewriter.setInsertionPoint(forallOp);

  if (originalBasisWasProvided) {
    SmallVector<mlir::Value> activeMappingSizes =
        builderResult.activeMappingSizes;
    SmallVector<mlir::Value> availableMappingSizes =
        builderResult.availableMappingSizes;
    SmallVector<Value> activeIdOps = builderResult.activeIdOps;

    for (auto [activeId, activeMappingSize, availableMappingSize] :
         llvm::zip_equal(activeIdOps, activeMappingSizes,
                         availableMappingSizes)) {
      auto maybeCstActiveMappingSize = getAsIndex(activeMappingSize);
      auto maybeCstAvailableMappingSize = getAsIndex(availableMappingSize);

      if (maybeCstActiveMappingSize.has_value() &&
          maybeCstAvailableMappingSize.has_value() &&
          maybeCstActiveMappingSize.value() ==
              maybeCstAvailableMappingSize.value())
        continue;

      if (maybeCstActiveMappingSize.has_value() &&
          maybeCstAvailableMappingSize.has_value() &&
          maybeCstActiveMappingSize.value() >
              maybeCstAvailableMappingSize.value()) {
        return definiteFailureHelper(
            transformOp, forallOp,
            "Trying to map to fewer GPU threads than loop iterations but "
            "overprovisioning is not yet supported. "
            "Try additional tiling of the before mapping or map to more "
            "threads.");
      }

      // activeMappingSize always small than availableMappingSizes
      Value tmpPredicate = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::ult, activeId, activeMappingSize);
      LDBG("----predicate: " << tmpPredicate);
      predicate = predicate ? rewriter.create<arith::AndIOp>(loc, predicate,
                                                             tmpPredicate)
                            : tmpPredicate;
    }
  }

  // Step 6. Move the body of forallOp.
  // Erase the terminator first, it will not be used.
  rewriter.eraseOp(forallOp.getTerminator());
  if (predicate) {
    // Step 6.a. If predicated, move at the beginning.
    auto ifOp = rewriter.create<scf::IfOp>(loc, predicate,
                                           /*withElseRegion=*/false);
    targetBlock = ifOp.thenBlock();
    insertionPoint = ifOp.thenBlock()->begin();
  }

  Block &sourceBlock = forallOp.getRegion().front();
  targetBlock->getOperations().splice(insertionPoint,
                                      sourceBlock.getOperations());

  // Step 7. RAUW indices.
  for (Value loopIndex : forallOp.getInductionVars()) {
    Value threadIdx = bvm.lookup(loopIndex);
    rewriter.replaceAllUsesWith(loopIndex, threadIdx);
  }

  rewriter.setInsertionPoint(forallOp);
  auto forallMappingSizesValue =
      getValueOrCreateConstantIndexOp(rewriter, loc, forallMappingSizes);

  auto totalMappingSize =
      getIndexProduct(rewriter, loc, forallMappingSizesValue).value();

  result = ForallRewriteResultExt{forallMappingSizesValue, mappingIdOps,
                                  totalMappingSize};

  // Step 8. Erase old op.
  rewriter.eraseOp(forallOp);
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// MapNestedForallToThreadsExtOp
//===----------------------------------------------------------------------===//

static DiagnosedSilenceableFailure checkMappingSpec(
    std::optional<TransformOpInterface> transformOp, scf::ForallOp forallOp,
    ArrayRef<int64_t> numParallelIterations, ArrayRef<int64_t> blockOrGridSizes,
    int factor, bool useLinearMapping = false) {
  if (!useLinearMapping && blockOrGridSizes.front() % factor != 0) {
    auto diag = definiteFailureHelper(
        transformOp, forallOp,
        Twine("3-D mapping: size of threadIdx.x must be a multiple of ") +
            std::to_string(factor));
    return diag;
  }
  // skip this check.
  if (computeProduct(numParallelIterations) * factor >
      computeProduct(blockOrGridSizes)) {
    auto diag = definiteFailureHelper(
        transformOp, forallOp,
        Twine("the number of required parallel resources (blocks or "
              "threads) ") +
            std::to_string(computeProduct(numParallelIterations) * factor) +
            std::string(" overflows the number of available resources ") +
            std::to_string(computeProduct(blockOrGridSizes)));
    return diag;
  }
  return DiagnosedSilenceableFailure::success();
}

static DiagnosedSilenceableFailure
getThreadIdBuilder(std::optional<TransformOpInterface> transformOp,
                   scf::ForallOp forallOp, ArrayRef<OpFoldResult> blockSizesOfr,
                   int64_t warpSize, GpuIdBuilderExt &gpuIdBuilder) {
  auto mappingAttr = cast<DeviceMappingAttrInterface>(
      forallOp.getMapping()->getValue().front());
  bool useLinearMapping = mappingAttr.isLinearMapping();

  // Sanity checks that may result in runtime verification errors.
  auto numParallelIterations =
      getConstantIntValues((forallOp.getMixedUpperBound()));
  auto blockSizes = getConstantIntValues(blockSizesOfr);
  // skip it for support dynamic
  if (!forallOp.isNormalized()) {
    return definiteFailureHelper(transformOp, forallOp,
                                 "requires normalized forall op");
  }

  int64_t factor = 1;
  if (isa<GPUWarpgroupMappingAttr>(mappingAttr)) {
    factor = mlir::transform::gpu::GpuWarpgroupIdBuilder::kNumWarpsPerGroup *
             warpSize;
  } else if (isa<GPUWarpMappingAttr>(mappingAttr)) {
    factor = warpSize;
  }

  // if forall mapping size or blockSizes are dynamic,
  // assuming blockSizes always larger that forall mapping size.
  if (forallOp.isNormalized() && numParallelIterations.has_value() &&
      blockSizes.has_value()) {
    DiagnosedSilenceableFailure diag =
        checkMappingSpec(transformOp, forallOp, numParallelIterations.value(),
                         blockSizes.value(), factor, useLinearMapping);
    if (!diag.succeeded())
      return diag;
  }

  // Start mapping.
  MLIRContext *ctx = forallOp.getContext();
  gpuIdBuilder =
      TypeSwitch<DeviceMappingAttrInterface, GpuIdBuilderExt>(mappingAttr)
          .Case([&](GPUWarpgroupMappingAttr) {
            return GpuWarpgroupIdBuilderExt(ctx, warpSize, useLinearMapping);
          })
          .Case([&](GPUWarpMappingAttr) {
            return GpuWarpIdBuilderExt(ctx, warpSize, useLinearMapping);
          })
          .Case([&](GPUThreadMappingAttr) {
            return GpuThreadIdBuilderExt(ctx, useLinearMapping);
          })
          .Default([&](DeviceMappingAttrInterface) -> GpuIdBuilderExt {
            llvm_unreachable("unknown mapping attribute");
          });
  return DiagnosedSilenceableFailure::success();
}

static DiagnosedSilenceableFailure mapOneForallToThreadsExtImpl(
    RewriterBase &rewriter, std::optional<TransformOpInterface> transformOp,
    scf::ForallOp forallOp, ArrayRef<int64_t> blockSizes, int64_t warpSize,
    bool syncAfterDistribute) {

  {
    // GPU-specific verifications. There is no better place to anchor
    // those right now: the ForallOp is target-independent and the transform
    // op does not apply to individual ForallOp.
    DiagnosedSilenceableFailure diag =
        verifyGpuMapping<ThreadMappingKind>(transformOp, forallOp);
    if (!diag.succeeded())
      return diag;
  }
  auto blockSizeOfr = getAsIndexOpFoldResult(rewriter.getContext(), blockSizes);
  GpuIdBuilderExt gpuIdBuilder;
  {
    // Try to construct the id builder, if it fails, return.
    DiagnosedSilenceableFailure diag = getThreadIdBuilder(
        transformOp, forallOp, blockSizeOfr, warpSize, gpuIdBuilder);
    if (!diag.succeeded())
      return diag;
  }

  Location loc = forallOp.getLoc();
  OpBuilder::InsertionGuard g(rewriter);
  // Insert after to allow for syncthreads after `forall` is erased.
  rewriter.setInsertionPointAfter(forallOp);
  ForallRewriteResultExt rewriteResult;

  DiagnosedSilenceableFailure diag =
      rewriteOneForallCommonImpl(rewriter, transformOp, forallOp, blockSizeOfr,
                                 rewriteResult, gpuIdBuilder);
  if (!diag.succeeded())
    return diag;
  // Add a syncthreads if needed. TODO: warpsync
  if (syncAfterDistribute)
    rewriter.create<BarrierOp>(loc);

  return DiagnosedSilenceableFailure::success();
}

static DiagnosedSilenceableFailure
mapNestedForallToThreadsExtImpl(RewriterBase &rewriter,
                                std::optional<TransformOpInterface> transformOp,
                                Operation *target, ArrayRef<int64_t> blockDims,
                                int64_t warpSize, bool syncAfterDistribute) {
  LDBG("Start mapNestedForallToThreadsExtImpl");
  if (blockDims.size() != 3) {
    return definiteFailureHelper(transformOp, target,
                                 "requires size-3 thread mapping");
  }

  // Create an early zero index value for replacements.
  Location loc = target->getLoc();
  Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  DiagnosedSilenceableFailure diag = DiagnosedSilenceableFailure::success();
  WalkResult walkResult = target->walk([&](scf::ForallOp forallOp) {
    diag =
        mapOneForallToThreadsExtImpl(rewriter, transformOp, forallOp, blockDims,
                                     warpSize, syncAfterDistribute);
    if (diag.isDefiniteFailure())
      return WalkResult::interrupt();
    if (diag.succeeded())
      return WalkResult::skip();
    return WalkResult::advance();
  });
  if (walkResult.wasInterrupted())
    return diag;

  // Replace ids of dimensions known to be 1 by 0 to simplify the IR.
  // Here, the result of mapping determines the available mapping sizes.
  replaceUnitMappingIdsHelper<ThreadIdOp>(rewriter, loc, target, zero,
                                          blockDims);

  return DiagnosedSilenceableFailure::success();
}

DiagnosedSilenceableFailure
transform::MapNestedForallToThreadsExtOp::applyToOne(
    transform::TransformRewriter &rewriter, Operation *target,
    ApplyToEachResultList &results, TransformState &state) {
  LaunchOp gpuLaunch = dyn_cast<LaunchOp>(target);
  auto transformOp = cast<TransformOpInterface>(getOperation());

  // Basic high-level verifications.
  if (!gpuLaunch)
    return emitSilenceableError() << "Given target is not a gpu.launch";

  // Mapping to block ids.
  SmallVector<int64_t> blockDims{getBlockDims()};
  DiagnosedSilenceableFailure diag = mlir::transform::gpu::checkGpuLimits(
      transformOp, std::nullopt, std::nullopt, std::nullopt, blockDims[0],
      blockDims[1], blockDims[2]);
  if (diag.isSilenceableFailure()) {
    diag.attachNote(getLoc()) << getBlockDimsAttrName() << " is too large";
    return diag;
  }

  // Set the GPU launch configuration for the block dims early, this is not
  // subject to IR inspection.
  diag = mlir::transform::gpu::alterGpuLaunch(
      rewriter, gpuLaunch, transformOp, std::nullopt, std::nullopt,
      std::nullopt, blockDims[0], blockDims[1], blockDims[2]);

  rewriter.setInsertionPointToStart(&gpuLaunch.getBody().front());
  diag = mapNestedForallToThreadsExtImpl(rewriter, transformOp, gpuLaunch,
                                         blockDims, getWarpSize(),
                                         getSyncAfterDistribute());

  results.push_back(gpuLaunch.getOperation());
  return diag;
}

LogicalResult transform::MapNestedForallToThreadsExtOp::verify() {
  if (getBlockDims().size() != 3) {
    return emitOpError() << "transform requires size-3 block_dims";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// MapForallToBlocksExt
//===----------------------------------------------------------------------===//
static DiagnosedSilenceableFailure
inferGridDim(RewriterBase &rewriter, SmallVectorImpl<OpFoldResult> &gridDims,
             ForallRewriteResultExt result, bool useLinearMapping) {

  auto isDynamic = [](OpFoldResult &val) -> bool {
    auto maybeInt = getConstantIntValue(val);
    if (maybeInt.has_value() && maybeInt.value() != ShapedType::kDynamic) {
      return false;
    }
    return true;
  };

  bool hasDynamic = llvm::any_of(gridDims, isDynamic);

  if (useLinearMapping) {
    if (hasDynamic || gridDims.empty())
      gridDims = SmallVector<OpFoldResult>{result.totalMappingSize};
  } else {
    if (gridDims.empty()) {
      gridDims = getAsOpFoldResult(result.mappingSizes);
    } else {
      for (auto en : llvm::enumerate(gridDims)) {
        if (isDynamic(en.value())) {
          assert(en.index() < result.mappingSizes.size() &&
                 "Can't infer dynamic grid dim.");
          gridDims[en.index()] = result.mappingSizes[en.index()];
        }
      }
    }
  }

  while (gridDims.size() < 3) {
    gridDims.emplace_back(getAsIndexOpFoldResult(rewriter.getContext(), 1));
  }

  return DiagnosedSilenceableFailure::success();
}

static DiagnosedSilenceableFailure mapForallToBlocksExtOpImpl(
    RewriterBase &rewriter, TransformOpInterface transformOp,
    scf::ForallOp forallOp, SmallVectorImpl<OpFoldResult> &gridDims,
    const GpuIdBuilderExt &gpuIdBuilder) {

  {
    // GPU-specific verifications. There is no better place to anchor
    // those right now: the ForallOp is target-independent and the transform
    // op does not apply to individual ForallOp.
    DiagnosedSilenceableFailure diag =
        verifyGpuMapping<BlockMappingKind>(transformOp, forallOp);
    if (!diag.succeeded())
      return diag;
  }

  Location loc = forallOp.getLoc();
  Block *parentBlock = forallOp->getBlock();
  Value zero;
  {
    // Create an early zero index value for replacements and immediately reset
    // the insertion point.
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(parentBlock);
    zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  }

  ForallRewriteResultExt rewriteResult;
  DiagnosedSilenceableFailure diag = rewriteOneForallCommonImpl(
      rewriter, transformOp, forallOp,
      /*availableMappingSizes=*/gridDims, rewriteResult, gpuIdBuilder);

  // Return if anything goes wrong, use silenceable failure as a match
  // failure.
  if (!diag.succeeded())
    return diag;

  bool useLinearMapping = false;
  if (forallOp.getMapping()) {
    auto mappingAttr = cast<DeviceMappingAttrInterface>(
        forallOp.getMapping()->getValue().front());
    useLinearMapping = mappingAttr.isLinearMapping();
  }

  // If gridDims is dynamic, infer it from the mappingSizes.
  {
    diag = inferGridDim(rewriter, gridDims, rewriteResult, useLinearMapping);
    if (!diag.succeeded())
      return diag;
  }

  assert(gridDims.size() == 3 && "Need 3-D gridDims");

  // Replace ids of dimensions known to be 1 by 0 to simplify the IR.
  // Here, the result of mapping determines the available mapping sizes.
  SmallVector<int64_t> staticMappingSizes;
  for (auto sz : rewriteResult.mappingSizes) {
    auto maybeIndex = getAsIndex(sz);
    if (maybeIndex.has_value()) {
      staticMappingSizes.emplace_back(maybeIndex.value());
    } else {
      staticMappingSizes.emplace_back(ShapedType::kDynamic);
    }
  }
  while (staticMappingSizes.size() < 3)
    staticMappingSizes.emplace_back(1);

  replaceUnitMappingIdsHelper<BlockDimOp>(rewriter, loc, parentBlock, zero,
                                          staticMappingSizes);

  return DiagnosedSilenceableFailure::success();
}

static DiagnosedSilenceableFailure
setGridDimsForGpuLaunch(RewriterBase &rewriter, LaunchOp gpuLaunch,
                        TransformOpInterface transformOp, Value gridDimX,
                        Value gridDimY, Value gridDimZ) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(gpuLaunch);
  std::optional<int64_t> maybeCstGridDimX = getAsIndex(gridDimX);
  std::optional<int64_t> maybeCstGridDimY = getAsIndex(gridDimY);
  std::optional<int64_t> maybeCstGridDimZ = getAsIndex(gridDimZ);

  DiagnosedSilenceableFailure diag = mlir::transform::gpu::checkGpuLimits(
      transformOp, maybeCstGridDimX, maybeCstGridDimY, maybeCstGridDimZ,
      std::nullopt, std::nullopt, std::nullopt);
  if (!diag.succeeded())
    return diag;

  auto copyDimComputation = [&](Value dim) -> Value {
    llvm::SetVector<Operation *> backwardSlice;
    getBackwardSlice(dim, &backwardSlice);
    IRMapping bvm;
    for (auto &&op : backwardSlice) {
      Operation *newOp = rewriter.clone(*op, bvm);
    }
    if (auto defOp = dim.getDefiningOp()) {
      SmallPtrSet<Operation *, 4> excepts;
      Operation *newDefOp = rewriter.clone(*defOp, bvm);
      return newDefOp->getResult(0);
    } else {
      return dim;
    }
  };

  gpuLaunch.getGridSizeXMutable().assign(copyDimComputation(gridDimX));
  gpuLaunch.getGridSizeYMutable().assign(copyDimComputation(gridDimY));
  gpuLaunch.getGridSizeZMutable().assign(copyDimComputation(gridDimZ));
  return DiagnosedSilenceableFailure::success();
}

DiagnosedSilenceableFailure mlir::transform::MapForallToBlocksExtOp::applyToOne(
    transform::TransformRewriter &rewriter, Operation *target,
    ApplyToEachResultList &results, TransformState &state) {
  LaunchOp gpuLaunch = dyn_cast<LaunchOp>(target);
  auto transformOp = cast<TransformOpInterface>(getOperation());

  if (!getGenerateGpuLaunch() && !gpuLaunch) {
    DiagnosedSilenceableFailure diag =
        emitSilenceableError()
        << "Given target is not gpu.launch, set `generate_gpu_launch` "
           "attribute";
    diag.attachNote(target->getLoc()) << "when applied to this payload op";
    return diag;
  }

  scf::ForallOp topLevelForallOp;
  DiagnosedSilenceableFailure diag = mlir::transform::gpu::findTopLevelForallOp(
      target, topLevelForallOp, transformOp);
  if (!diag.succeeded()) {
    diag.attachNote(target->getLoc()) << "when applied to this payload op";
    return diag;
  }
  assert(topLevelForallOp && "expect an scf.forall");

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(topLevelForallOp);

  // Generate gpu launch here and move the forall inside
  if (getGenerateGpuLaunch()) {
    DiagnosedSilenceableFailure diag = mlir::transform::gpu::createGpuLaunch(
        rewriter, target->getLoc(), transformOp, gpuLaunch);
    if (!diag.succeeded())
      return diag;

    rewriter.setInsertionPointToStart(&gpuLaunch.getBody().front());
    Operation *newForallOp = rewriter.clone(*topLevelForallOp);
    rewriter.eraseOp(topLevelForallOp);
    topLevelForallOp = cast<scf::ForallOp>(newForallOp);
  }

  // The BlockIdBuilder adapts to whatever is thrown at it.
  bool useLinearMapping = false;
  if (topLevelForallOp.getMapping()) {
    auto mappingAttr = cast<DeviceMappingAttrInterface>(
        topLevelForallOp.getMapping()->getValue().front());
    useLinearMapping = mappingAttr.isLinearMapping();
  }

  SmallVector<OpFoldResult> gridDims;
  int64_t dynamicDimsCount = 0;
  for (auto en : llvm::enumerate(getGridDims())) {
    if (en.value() == ShapedType::kDynamic) {
      dynamicDimsCount += 1;
      auto dim = rewriter.create<BlockDimOp>(
          target->getLoc(), rewriter.getIndexType(),
          static_cast<mlir::gpu::Dimension>(en.index()));
      gridDims.emplace_back(dim);
    } else {
      gridDims.emplace_back(
          getAsIndexOpFoldResult(rewriter.getContext(), en.value()));
    }
  }

  if (useLinearMapping && dynamicDimsCount > 0 && dynamicDimsCount < 3) {
    DiagnosedSilenceableFailure diag = emitSilenceableError()
                                       << "if forall use linear mapping under "
                                          "dynamic, grid dims must be set to "
                                          "empty or all dynamic";
    return diag;
  }

  // use mapping size of forall as gridDims.
  if (dynamicDimsCount == 3) {
    gridDims.clear();
  }

  GpuBlockIdBuilderExt gpuBlockIdBuilder(getContext(), useLinearMapping);

  diag = mapForallToBlocksExtOpImpl(rewriter, transformOp, topLevelForallOp,
                                    gridDims, gpuBlockIdBuilder);
  if (!diag.succeeded())
    return diag;

  rewriter.setInsertionPoint(gpuLaunch);

  auto gridDimsValue =
      getValueOrCreateConstantIndexOp(rewriter, gpuLaunch->getLoc(), gridDims);

  // Set the GPU launch configuration for the grid dims late, this is
  // subject to IR inspection.
  diag = setGridDimsForGpuLaunch(
      rewriter, gpuLaunch, cast<TransformOpInterface>(getOperation()),
      gridDimsValue[0], gridDimsValue[1], gridDimsValue[2]);

  results.push_back(gpuLaunch);
  return diag;
}

LogicalResult transform::MapForallToBlocksExtOp::verify() {
  if (!getGridDims().empty() && getGridDims().size() != 3) {
    return emitOpError() << "transform requires empty or size-3 grid_dims";
  }
  return success();
}

ParseResult MapForallToBlocksExtOp::parse(OpAsmParser &parser,
                                          OperationState &result) {
  OpAsmParser::UnresolvedOperand targetRawOperands[1];
  llvm::ArrayRef<OpAsmParser::UnresolvedOperand> targetOperands(
      targetRawOperands);
  llvm::SMLoc targetOperandsLoc;
  (void)targetOperandsLoc;
  DenseI64ArrayAttr grid_dimsAttr;
  llvm::ArrayRef<Type> targetTypes;
  llvm::ArrayRef<Type> resultTypes;
  DenseI64ArrayAttr gridDims;

  targetOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(targetRawOperands[0]))
    return failure();

  if (succeeded(parser.parseOptionalKeyword("generate_gpu_launch"))) {
    result.getOrAddProperties<MapForallToBlocksExtOp::Properties>()
        .generate_gpu_launch = parser.getBuilder().getUnitAttr();
  }

  if (succeeded(parser.parseOptionalKeyword("grid_dims"))) {
    if (parser.parseEqual())
      return failure();
    parseI64ArrayWithDynamic(parser, gridDims);
    result.addAttribute(getGridDimsAttrName(result.name), gridDims);
  }

  {
    auto loc = parser.getCurrentLocation();
    (void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
          return parser.emitError(loc)
                 << "'" << result.name.getStringRef() << "' op ";
        })))
      return failure();
  }

  if (parser.parseColon())
    return failure();

  FunctionType functionType;
  if (parser.parseType(functionType))
    return failure();
  targetTypes = functionType.getInputs();
  resultTypes = functionType.getResults();
  result.addTypes(resultTypes);
  if (parser.resolveOperands(targetOperands, targetTypes, targetOperandsLoc,
                             result.operands))
    return failure();
  return success();
}

void MapForallToBlocksExtOp::print(OpAsmPrinter &printer) {
  printer << ' ';
  printer << getTarget();

  if (getGenerateGpuLaunchAttr()) {
    printer << ' ' << "generate_gpu_launch";
  }

  if (getGridDimsAttr()) {
    printer << ' ' << "grid_dims";
    printer << ' ' << "=";
    printer << ' ';
    printI64ArrayWithDynamic(printer, getGridDims());
  }

  llvm::SmallVector<llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("generate_gpu_launch");
  elidedAttrs.push_back("grid_dims");

  {
    Builder odsBuilder(getContext());
    Attribute attr = getGridDimsAttr();
    if (attr && (attr == odsBuilder.getDenseI64ArrayAttr({})))
      elidedAttrs.push_back("grid_dims");
  }

  {
    Builder odsBuilder(getContext());
    Attribute attr = getGenerateGpuLaunchAttr();
    if (attr && (attr == ((false) ? odsBuilder.getUnitAttr() : nullptr)))
      elidedAttrs.push_back("generate_gpu_launch");
  }

  printer.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  printer << ' ' << ":";
  printer << ' ';
  printer.printFunctionalType(llvm::ArrayRef<Type>(getTarget().getType()),
                              llvm::ArrayRef<Type>(getResult().getType()));
}

//===----------------------------------------------------------------------===//
// Transform op registration
//===----------------------------------------------------------------------===//

namespace {
/// Registers new ops and declares PDL as dependent dialect since the
/// additional ops are using PDL types for operands and results.
class GPUExtTransformDialectExtension
    : public transform::TransformDialectExtension<
          GPUExtTransformDialectExtension> {
public:
  GPUExtTransformDialectExtension() {
    declareGeneratedDialect<scf::SCFDialect>();
    declareGeneratedDialect<arith::ArithDialect>();
    declareGeneratedDialect<GPUDialect>();
    registerTransformOps<
#define GET_OP_LIST
#include "byteir/Dialect/GPU/TransformOps/GPUExtTransformOps.cpp.inc"
        >();
  }
};
} // namespace

#define GET_OP_CLASSES
#include "byteir/Dialect/GPU/TransformOps/GPUExtTransformOps.cpp.inc"

void mlir::gpu_ext::registerTransformDialectExtension(
    DialectRegistry &registry) {
  registry.addExtensions<GPUExtTransformDialectExtension>();
}
