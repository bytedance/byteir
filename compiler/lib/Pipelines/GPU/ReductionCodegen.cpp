//===- ReductionCodegen.cpp ---------------------------------*--- C++ -*-===//
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

#include "byteir/Pipelines/GPU/ReductionCodegen.h"

#include "byteir/Conversion/ToGPU/ToGPU.h"
#include "byteir/Conversion/ToLLVM/ToLLVM.h"
#include "byteir/Dialect/Linalg/TransformOps/LinalgExtTransformOps.h"
#include "byteir/Dialect/Transform/IR/TransformExtOps.h"
#include "byteir/Dialect/Transform/Transforms/TransformInsertion.h"
#include "byteir/Pipelines/Common/Utils.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/TransformOps/BufferizationTransformOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/TransformOps/SCFTransformOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SmallSet.h"
#include <byteir/Dialect/Vector/TransformOps/VectorExtTransformOps.h>
#include <mlir/Dialect/PDL/IR/PDLTypes.h>
#include <mlir/Dialect/Vector/TransformOps/VectorTransformOps.h>
#include <numeric>

using namespace mlir;

namespace {
//----------------------------------------------------------------------------//
// common helpers
//----------------------------------------------------------------------------//
// TODO: move to common header

constexpr bool isPowerOf2(int64_t n) { return (!(n & (n - 1))); }

constexpr int64_t nextPowerOf2(int64_t n) {
  return (n <= 1) ? 1 : (isPowerOf2(n) ? n : (2 * nextPowerOf2((n + 1) / 2)));
}

bool isMappedToGPUBlocks(scf::ForOp forOp) {
  if (auto loopToSIMTAttr =
          forOp->getAttrOfType<StringAttr>(getLoopToSIMTAttrName())) {
    auto mappingTo = loopToSIMTAttr.getValue();
    if (mappingTo == getBlockIdXName() || mappingTo == getBlockIdYName() ||
        mappingTo == getBlockIdZName()) {
      return true;
    }
  }
  return false;
}

bool isMappedToGPUBlocks(scf::ForallOp forallOp) {
  if (auto mapping = forallOp.getMappingAttr()) {
    if (llvm::any_of(mapping.getValue(), [](Attribute attr) {
          return isa<gpu::GPUBlockMappingAttr>(attr);
        })) {
      return true;
    }
  }

  return false;
}

bool isMappedToGPUBlocks(Operation *op) {
  if (auto forOp = llvm::dyn_cast_or_null<scf::ForOp>(op)) {
    return isMappedToGPUBlocks(forOp);
  }
  if (auto forallOp = llvm::dyn_cast_or_null<scf::ForallOp>(op)) {
    return isMappedToGPUBlocks(forallOp);
  }
  return false;
}

bool isMappedToGPUThreads(scf::ForOp forOp) {
  if (auto loopToSIMTAttr =
          forOp->getAttrOfType<StringAttr>(getLoopToSIMTAttrName())) {
    auto mappingTo = loopToSIMTAttr.getValue();
    if (mappingTo == getThreadIdXName() || mappingTo == getThreadIdYName() ||
        mappingTo == getThreadIdZName()) {
      return true;
    }
  }
  return false;
}

bool isMappedToGPUThreads(scf::ForallOp forallOp) {
  if (auto mapping = forallOp.getMappingAttr()) {
    if (llvm::any_of(mapping.getValue(), [](Attribute attr) {
          return isa<gpu::GPUThreadMappingAttr>(attr);
        })) {
      return true;
    }
  }

  return false;
}

bool isMappedToGPUThreads(Operation *op) {
  if (auto forOp = llvm::dyn_cast_or_null<scf::ForOp>(op)) {
    return isMappedToGPUThreads(forOp);
  }
  if (auto forallOp = llvm::dyn_cast_or_null<scf::ForallOp>(op)) {
    return isMappedToGPUThreads(forallOp);
  }
  return false;
}

uint64_t getNumTiledLoops(ArrayRef<int64_t> tileSizes) {
  return llvm::count_if(tileSizes,
                        [](int64_t tileSize) { return tileSize > 0; });
}

std::optional<int64_t> getReductionDim(linalg::GenericOp genericOp) {
  SmallVector<unsigned> reductionDims;
  genericOp.getReductionDims(reductionDims);
  if (reductionDims.size() == 1) {
    return reductionDims[0];
  }
  return std::nullopt;
}

std::optional<int64_t> getOperandReductionDim(OpOperand &operand) {
  auto genericOp = llvm::dyn_cast<linalg::GenericOp>(operand.getOwner());
  if (!genericOp)
    return std::nullopt;

  auto dim = getReductionDim(genericOp);
  if (!dim.has_value())
    return std::nullopt;

  auto affineMap = genericOp.getIndexingMapsArray()[operand.getOperandNumber()];
  if (!affineMap || !affineMap.isProjectedPermutation())
    return std::nullopt;

  for (auto &&en : llvm::enumerate(affineMap.getResults())) {
    if (auto dimExpr = en.value().dyn_cast<AffineDimExpr>()) {
      if (dimExpr.getPosition() == *dim) {
        return en.index();
      }
    }
  }

  return std::nullopt;
}

SmallVector<int64_t> getDynamicDims(linalg::GenericOp genericOp) {
  auto staticLoopRanges = genericOp.getStaticLoopRanges();
  SmallVector<int64_t> ret;
  for (int64_t i = 0; i < staticLoopRanges.size(); ++i) {
    if (ShapedType::isDynamic(staticLoopRanges[i])) {
      ret.push_back(i);
    }
  }
  return ret;
}

//----------------------------------------------------------------------------//
// configuration structs
//----------------------------------------------------------------------------//

static constexpr StringLiteral kGridReduction = "__grid_reduction__";
static constexpr StringLiteral kBlockReduction = "__block_reduction__";
static constexpr StringLiteral kWarpReduction = "__warp_reduction__";
static constexpr StringLiteral kThreadReduction = "__thread_reduction__";

struct ProducerSelector {
  uint64_t operandNumber;
  llvm::StringRef opName;
  std::vector<ProducerSelector> producerSelectors;

  ProducerSelector(uint64_t operandNumber, llvm::StringRef opName)
      : operandNumber(operandNumber), opName(opName) {}

  static bool detectFillOperand(OpOperand *opOperand,
                                std::vector<ProducerSelector> &selectors) {
    if (opOperand->get().getDefiningOp<linalg::FillOp>()) {
      selectors.emplace_back(opOperand->getOperandNumber(),
                             linalg::FillOp::getOperationName());
      return true;
    }
    return false;
  }

  static bool detectPadOperand(OpOperand *opOperand,
                               std::vector<ProducerSelector> &selectors) {
    Operation *definingOp = opOperand->get().getDefiningOp();
    if (!definingOp)
      return false;

    if (llvm::isa<tensor::ExpandShapeOp, tensor::CollapseShapeOp>(definingOp)) {
      ProducerSelector selector(opOperand->getOperandNumber(),
                                definingOp->getName().getStringRef());
      if (detectPadOperand(&definingOp->getOpOperand(0),
                           selector.producerSelectors)) {
        selectors.emplace_back(std::move(selector));
        return true;
      }
    } else if (llvm::isa<tensor::PadOp>(definingOp)) {
      selectors.emplace_back(opOperand->getOperandNumber(),
                             tensor::PadOp::getOperationName());
      return true;
    }
    return false;
  }
};

struct GridSplitConfig {
  int64_t splitFactor;
  int64_t dimension;

  void apply(ImplicitLocOpBuilder &b, Value pdlV);
};

struct GridTileConfig {
  SmallVector<int64_t> tileSizes;
  SmallVector<gpu::MappingId> mapping;
  std::vector<ProducerSelector> fuseCandidates;

  void apply(ImplicitLocOpBuilder &b, Value pdlV, bool usingForall);
};

struct BlockSplitConfig {
  SmallVector<int64_t> splitFactors;
  SmallVector<int64_t> dimensions;
  SmallVector<int64_t> padDims;
  SmallVector<Attribute> padValues;

  void apply(ImplicitLocOpBuilder &b, Value pdlV);
};

struct BlockTileConfig {
  SmallVector<int64_t> tileSizes;
  SmallVector<gpu::MappingId> mapping;
  std::vector<ProducerSelector> fuseCandidates;
  int64_t reductionSize;
  int64_t remainBlockSize;
  int64_t warpSize;
  bool mapToWarp;

  void apply(ImplicitLocOpBuilder &b, Value pdlV, bool usingForall);
};

struct ThreadTileConfig {
  SmallVector<int64_t> parallelTileSizes;
  SmallVector<int64_t> reductionTileSizes;
  SmallVector<int64_t> unrollFactors;
  std::vector<ProducerSelector> initOperands;

  void apply(ImplicitLocOpBuilder &b, Value pdlV);
};

void processProducerSelectors(
    ImplicitLocOpBuilder &b,
    const std::vector<ProducerSelector> &producerSelectors, Value fuseInto,
    SmallVector<Value> &selected, Type producerType = nullptr) {
  for (auto selector : producerSelectors) {
    auto producer = b.create<transform::GetProducerOfOperand>(
        /* producer type */ producerType
            ? producerType
            : transform::OperationType::get(b.getContext(), selector.opName),
        /* target */ fuseInto,
        /* operand number */ selector.operandNumber);
    selected.push_back(producer.getProducer());
    processProducerSelectors(b, selector.producerSelectors, selected.back(),
                             selected);
  }
}

transform::TileUsingForallOp
tileToForallAndFuseImpl(ImplicitLocOpBuilder &b, Value toTile,
                        const SmallVector<int64_t> &tileSizes,
                        const SmallVector<Attribute> &mapping,
                        const std::vector<ProducerSelector> &fuseCandidates,
                        const int64_t reductionSize = 0,
                        const int64_t warpSize = 32,
                        const int64_t remainBlockSize = 512) {
  SmallVector<Value> toBeFused;
  processProducerSelectors(b, fuseCandidates, toTile, toBeFused);

  auto tileOp = b.create<transform::TileUsingForallOp>(
      /* target */ toTile,
      /* staticTileSizes */ tileSizes,
      /* ctor tag */ transform::TileSizesSpec(),
      /* mapping */ b.getArrayAttr(mapping));
  for (auto &&producerOp : toBeFused) {
    b.create<transform::FuseIntoContainingOp>(
        /* producerOp */ producerOp,
        /* containingOp */ tileOp.getForallOp());
  }
  // if (reductionSize == warpSize && remainBlockSize >= warpSize) {
  //   auto warpMap = b.getArrayAttr(
  //       {gpu::GPUWarpMappingAttr::get(b.getContext(), gpu::Warps::DimX)});
  //   auto tileSize = ArrayRef<int64_t>{0, 0};
  //   tileOp = b.create<transform::TileToForallOp>(
  //       /* target */ tileOp.getTiledOp(),
  //       /* staticTileSizes */ tileSize,
  //       /* ctor tag */ transform::TileSizesSpec(),
  //       /* mapping */
  //       warpMap);
  // }
  return tileOp;
}

void tileToSCFForAndFuseImpl(ImplicitLocOpBuilder &b, Value toTile,
                             const SmallVector<int64_t> &tileSizes,
                             const SmallVector<Attribute> &mapping) {
  auto pdlType = pdl::OperationType::get(b.getContext());
  auto fuseOp = b.create<transform::FuseOp>(
      /* transformed */ pdlType,
      /* loops */
      SmallVector<Type>(getNumTiledLoops(tileSizes), pdlType),
      /* target */ toTile,
      /* tile_sizes */ b.getI64ArrayAttr(tileSizes),
      /* tile_interchange */ ArrayAttr());
  for (auto &&[loop, mapTo] : llvm::zip(fuseOp.getLoops(), mapping)) {
    Value paramV = b.create<transform::ParamConstantOp>(
        /* type */ pdl::AttributeType::get(b.getContext()),
        /* value */ mapTo);
    b.create<transform::AnnotateOp>(
        /* target */ loop,
        /* name */ getLoopToSIMTAttrName(),
        /* param */ paramV);
  }
}

void GridSplitConfig::apply(ImplicitLocOpBuilder &b, Value pdlV) {
  if (splitFactor) {
    auto splitted = b.create<transform::SplitReductionOp>(
        /* target */ pdlV,
        /* splitFactor */ splitFactor,
        /* insertSplitDimension */ dimension,
        /* innerParallel */ false,
        /* useScalingAlgorithm */ false,
        /* useAlloc */ false);
    b.create<transform::AnnotateOp>(
        /* target */ splitted.getSplitLinalgOp(),
        /* name */ kGridReduction,
        /* param */ Value());
    b.create<transform::AnnotateOp>(
        /* target */ splitted.getCombiningLinalgOp(),
        /* name */ kGridReduction,
        /* param */ Value());
  } else {
    b.create<transform::AnnotateOp>(
        /* target */ pdlV,
        /* name */ kGridReduction,
        /* param */ Value());
  }
}

void GridTileConfig::apply(ImplicitLocOpBuilder &b, Value pdlV,
                           bool usingForall) {
  if (usingForall) {
    auto mappingAttrs = llvm::to_vector(
        llvm::map_range(mapping, [&](gpu::MappingId dim) -> Attribute {
          return gpu::GPUBlockMappingAttr::get(b.getContext(), dim);
        }));
    tileToForallAndFuseImpl(b, pdlV, tileSizes, mappingAttrs, fuseCandidates);
  } else {
    static constexpr std::array<StringRef, 3> mappings{
        getBlockIdXName(), getBlockIdYName(), getBlockIdZName()};
    auto mappingAttrs = llvm::to_vector(
        llvm::map_range(mapping, [&](gpu::MappingId dim) -> Attribute {
          return b.getStringAttr(mappings[static_cast<int64_t>(dim)]);
        }));
    tileToSCFForAndFuseImpl(b, pdlV, tileSizes, mappingAttrs);
  }
}

void BlockSplitConfig::apply(ImplicitLocOpBuilder &b, Value pdlV) {
  if (!padDims.empty()) {
    auto padOp = b.create<transform::PadOp>(
        TypeRange{pdlV.getType(), pdlV.getType(), pdlV.getType()}, pdlV,
        /*padding_values=*/b.getArrayAttr(padValues),
        /*padding_dimensions=*/
        b.getI64ArrayAttr(padDims),
        /*padToMultipleOf=*/ArrayAttr{},
        /*pack_paddings=*/ArrayAttr{},
        /*transpose_paddings=*/ArrayAttr{},
        /*copyBack=*/transform::PadOp::kCopyOpNone);
    pdlV = padOp.getPadded();
  }
  if (!splitFactors.empty()) {
    Value toSplit = pdlV;
    for (auto &&[splitFactor, redDim] : llvm::zip(splitFactors, dimensions)) {
      auto splitted = b.create<transform::SplitReductionOp>(
          /* target */ toSplit,
          /* splitFactor */ splitFactor,
          /* insertSplitDimension */ redDim,
          /* innerParallel */ false,
          /* useScalingAlgorithm */ false,
          /* useAlloc */ false);
      b.create<transform::AnnotateOp>(
          /* target */ splitted.getInitOrAllocOp(),
          /* name */ kBlockReduction,
          /* param */ Value());
      b.create<transform::AnnotateOp>(
          /* target */ splitted.getCombiningLinalgOp(),
          /* name */ kBlockReduction,
          /* param */ Value());
      toSplit = splitted.getCombiningLinalgOp();
    }
    pdlV = toSplit;
  } else {
    b.create<transform::AnnotateOp>(
        /* target */ pdlV,
        /* name */ kBlockReduction,
        /* param */ Value());
  }
  auto func = b.create<transform::GetParentOp>(
      pdlV.getType(), pdlV,
      /* isolated_from_above */ true,
      /* allow_empty_results */ false,
      /* op_name */ b.getStringAttr(func::FuncOp::getOperationName()),
      /* deduplicate */ false,
      /* nth_parent */ 1);
  b.create<transform::ApplyPatternsOp>(func, [](OpBuilder &b, Location loc) {
    b.create<transform::ApplyCanonicalizationPatternsOp>(loc);
  });
  auto forall = b.create<transform::GetParentOp>(
      pdlV.getType(), pdlV,
      /* isolated_from_above */ false,
      /* allow_empty_results */ false,
      /* op_name */ b.getStringAttr(scf::ForallOp::getOperationName()),
      /* deduplicate */ false,
      /* nth_parent */ 1);
  if (!padDims.empty()) {
    auto parallelInsertSliceType = transform::OperationType::get(
        b.getContext(), tensor::ParallelInsertSliceOp::getOperationName());
    auto parallelInsertSlice = b.create<transform::MatchOp>(
        parallelInsertSliceType, forall,
        tensor::ParallelInsertSliceOp::getOperationName());
    b.create<transform::InsertSliceToCopyExtOp>(pdlV.getType(),
                                                parallelInsertSlice);
  }
  auto emptyTensorType = transform::OperationType::get(
      b.getContext(), tensor::EmptyOp::getOperationName());
  auto emptyTensor = b.create<transform::MatchOp>(
      emptyTensorType, forall, tensor::EmptyOp::getOperationName());
  auto allocTensorType = transform::OperationType::get(
      b.getContext(), bufferization::AllocTensorOp::getOperationName());
  auto allocTensor = b.create<transform::EmptyTensorToAllocTensorOp>(
      allocTensorType, emptyTensor);
  auto memorySpaceAttrName =
      bufferization::AllocTensorOp::getMemorySpaceAttrName(OperationName(
          bufferization::AllocTensorOp::getOperationName(), b.getContext()));
  auto workgroupMemoryAddressSpace = gpu::AddressSpaceAttr::get(
      b.getContext(), gpu::GPUDialect::getWorkgroupAddressSpace());
  Value paramV = b.create<transform::ParamConstantOp>(
      /* type */ pdl::AttributeType::get(b.getContext()),
      /* value */ workgroupMemoryAddressSpace);
  b.create<transform::AnnotateOp>(
      /* target */ allocTensor,
      /* name */ memorySpaceAttrName,
      /* param */ paramV);
}

void BlockTileConfig::apply(ImplicitLocOpBuilder &b, Value pdlV,
                            bool usingForall) {
  static int32_t reduction_kernel_cnt = 0;
  if (usingForall) {
    auto mappingAttrs = llvm::to_vector(
        llvm::map_range(mapping, [&](gpu::MappingId dim) -> Attribute {
          if (mapToWarp == true) {
            return gpu::GPUWarpMappingAttr::get(b.getContext(), dim);
          }
          return gpu::GPUThreadMappingAttr::get(b.getContext(), dim);
        }));
    transform::TileUsingForallOp tileOp = tileToForallAndFuseImpl(
        b, pdlV, tileSizes, mappingAttrs, fuseCandidates, reductionSize,
        warpSize, remainBlockSize);
    if (mapToWarp) {
      auto pdlType = pdl::OperationType::get(b.getContext());
      std::string func_name =
          "redunction_kernel_" + std::to_string(reduction_kernel_cnt++);
      transform::LoopOutlineOp outlineOp = b.create<transform::LoopOutlineOp>(
          pdlType, pdlType, tileOp.getTiledOp(), func_name);
      auto vecChildOp =
          b.create<transform::VectorizeChildrenAndApplyPatternsOp>(
              outlineOp.getFunction());
      b.create<transform::ApplyPatternsOp>(
          vecChildOp, [](OpBuilder &b, Location loc) {
            b.create<transform::ApplyLowerMultiReductionPatternsOp>(
                loc, vector::VectorMultiReductionLowering::InnerReduction);
          });
      auto shuffleOp =
          b.create<transform::ConvertReductionToGPUShuffleOp>(vecChildOp);
      b.create<transform::InlineOp>(outlineOp.getCall());
    }
  } else {
    static constexpr std::array<StringRef, 3> mappings{
        getThreadIdXName(), getThreadIdYName(), getThreadIdZName()};
    auto mappingAttrs = llvm::to_vector(
        llvm::map_range(mapping, [&](gpu::MappingId dim) -> Attribute {
          return b.getStringAttr(mappings[static_cast<int64_t>(dim)]);
        }));
    tileToSCFForAndFuseImpl(b, pdlV, tileSizes, mappingAttrs);
  }
}

void ThreadTileConfig::apply(ImplicitLocOpBuilder &b, Value pdlV) {
  auto pdlType = pdl::OperationType::get(b.getContext());
  auto numTiledParallelLoops = getNumTiledLoops(parallelTileSizes);
  SmallVector<Value> loops;
  if (numTiledParallelLoops > 0) {
    auto fuseOp = b.create<transform::FuseOp>(
        /* transformed */ pdlType,
        /* loops */
        SmallVector<Type>(getNumTiledLoops(parallelTileSizes), pdlType),
        /* target */ pdlV,
        /* tile_sizes */ b.getI64ArrayAttr(parallelTileSizes),
        /* tile_interchange */ ArrayAttr());
    loops = fuseOp.getLoops();
    pdlV = fuseOp.getTransformed();
  }

  auto tileOp = b.create<transform::TileUsingForOp>(
      /* target */ pdlV,
      /* tillSizes */ reductionTileSizes);
  loops.push_back(tileOp.getLoops()[0]);
  for (auto &&[loop, factor] : llvm::reverse(llvm::zip(loops, unrollFactors))) {
    b.create<transform::LoopUnrollOp>(loop, factor);
  }
}

//----------------------------------------------------------------------------//
// codegen strategies
//----------------------------------------------------------------------------//

bool isReductionOp(linalg::GenericOp genericOp) {
  if (genericOp.getNumReductionLoops() != 1)
    return false;

  if (!llvm::all_of(genericOp.getIndexingMapsArray(), [](AffineMap affineMap) {
        return affineMap.isProjectedPermutation(/* allowZeroInResults */ false);
      }))
    return false;

  return true;
}

bool isGridReductionOp(linalg::GenericOp genericOp) {
  if (!isReductionOp(genericOp))
    return false;

  // early return for manual tag
  if (genericOp->hasAttr(kGridReduction))
    return true;

  // top level generic op in function
  if (genericOp->getParentOfType<func::FuncOp>())
    return true;

  return false;
}

bool isBlockReductionOp(linalg::GenericOp genericOp) {
  if (!isReductionOp(genericOp))
    return false;

  // early return for manual tag
  if (genericOp->hasAttr(kBlockReduction))
    return true;

  // nested in op which is mapped to GPU blocks
  if (isMappedToGPUBlocks(genericOp->getParentOp()))
    return true;

  return false;
}

bool isThreadReductionOp(linalg::GenericOp genericOp) {
  if (!isReductionOp(genericOp))
    return false;

  // early return for manual tag
  if (genericOp->hasAttr(kThreadReduction))
    return true;

  // nested in op which is mapped to GPU threads
  if (isMappedToGPUThreads(genericOp->getParentOp()))
    return true;

  return false;
}

std::optional<GridSplitConfig> getGridSplitConfig(linalg::GenericOp genericOp,
                                                  int64_t splitFactor) {
  if (!isGridReductionOp(genericOp))
    return std::nullopt;

  auto redDim = *getReductionDim(genericOp);
  auto staticLoopRanges = genericOp.getStaticLoopRanges();
  if (ShapedType::isDynamic(staticLoopRanges[redDim]) ||
      staticLoopRanges[redDim] % splitFactor != 0 ||
      staticLoopRanges[redDim] <= 1024)
    return std::nullopt;

  return GridSplitConfig{splitFactor, redDim ? redDim - 1 : redDim};
}

std::optional<GridTileConfig> getGridTileConfig(linalg::GenericOp genericOp,
                                                int64_t warpSize,
                                                int64_t blockSize) {
  if (!isGridReductionOp(genericOp))
    return std::nullopt;

  int64_t numLoops = genericOp.getNumLoops();
  SmallVector<int64_t> tileSizes(numLoops, 1);
  auto loopSizes =
      cast<linalg::LinalgOp>(genericOp.getOperation()).computeStaticLoopSizes();
  for (auto &&affineMap : genericOp.getIndexingMapsArray()) {
    if (affineMap.isPermutation()) {
      auto dim = affineMap.getDimPosition(numLoops - 1);
      if (loopSizes[dim] > warpSize) { // TODO: padding
        tileSizes[dim] *= warpSize;
        break;
      }
    }
  }

  auto redDim = getReductionDim(genericOp).value();
  tileSizes[redDim] = 0;

  std::vector<ProducerSelector> fuseCandidates;
  for (OpOperand &opOperand : genericOp.getDpsInitsMutable()) {
    ProducerSelector::detectFillOperand(&opOperand, fuseCandidates);
  }

  auto numTiledLoops = getNumTiledLoops(tileSizes);
  if (!numTiledLoops) {
    tileSizes[redDim] = loopSizes[redDim];
    numTiledLoops = 1;
  }
  if (numTiledLoops >= 1 && numTiledLoops <= 3) {
    SmallVector<int64_t> mapping(numLoops, -1);
    int64_t dimMapping = static_cast<int64_t>(gpu::MappingId::DimX);
    for (auto &&affineMap : genericOp.getIndexingMapsArray()) {
      if (affineMap.isPermutation()) {
        for (int64_t i = numLoops - 1; i >= 0; i--) {
          auto dim = affineMap.getDimPosition(i);
          if (tileSizes[dim] > 0) {
            mapping[dim] = dimMapping++;
          }
        }
        break;
      }
    }
    mapping.erase(std::remove(mapping.begin(), mapping.end(), -1),
                  mapping.end());
    if (mapping.size() != numTiledLoops)
      return std::nullopt;

    return GridTileConfig{
        tileSizes,
        llvm::to_vector(llvm::map_range(
            mapping, [](int64_t i) { return static_cast<gpu::MappingId>(i); })),
        fuseCandidates};
  }
  return std::nullopt;
}

std::optional<BlockSplitConfig> getBlockSplitConfig(linalg::GenericOp genericOp,
                                                    int64_t splitFactor,
                                                    int64_t warpSize) {
  if (!isBlockReductionOp(genericOp))
    return std::nullopt;

  SmallVector<int64_t> padDims = getDynamicDims(genericOp);
  SmallVector<Attribute> padValues;

  SmallVector<int64_t> splitFactors;
  SmallVector<int64_t> dimensions;
  auto redDim = *getReductionDim(genericOp);
  auto staticLoopRanges = genericOp.getStaticLoopRanges();
  if (ShapedType::isDynamic(staticLoopRanges[redDim]))
    return std::nullopt;

  if (auto redPos = getOperandReductionDim(*genericOp.getDpsInputOperand(0))) {
    if (redPos.value() == genericOp.getNumLoops() - 1) {
      auto newSplitFactor = splitFactor * 2;
      while (staticLoopRanges[redDim] % newSplitFactor == 0 &&
             newSplitFactor <= splitFactor * warpSize) {
        newSplitFactor *= 2;
      }
      splitFactor = newSplitFactor / 2;
    }
  }
  if (staticLoopRanges[redDim] < splitFactor) {
    splitFactor = staticLoopRanges[redDim];
  } else {
    if (staticLoopRanges[redDim] % splitFactor != 0)
      return std::nullopt;

    splitFactors.push_back(splitFactor);
    dimensions.push_back(redDim ? redDim - 1 : redDim);
  }

  mlir::Builder b(genericOp.getContext());
  for (auto &&operand : genericOp->getOperands()) {
    if (auto shapedType = llvm::dyn_cast<ShapedType>(operand.getType())) {
      padValues.push_back(b.getZeroAttr(shapedType.getElementType()));
    } else {
      return std::nullopt;
    }
  }

  // for (; splitFactor > 2; splitFactor >>= 1) {
  //   splitFactors.push_back(splitFactor / 2);
  //   dimensions.push_back(redDim ? redDim - 1 : redDim);
  // }
  // haven't consider padding
  if (splitFactor > warpSize) {
    splitFactors.push_back(splitFactor / 32);
    dimensions.push_back(redDim ? redDim - 1 : redDim);
  }

  return BlockSplitConfig{splitFactors, dimensions, padDims, padValues};
}

std::optional<BlockTileConfig> getBlockTileConfig(linalg::GenericOp genericOp,
                                                  int64_t warpSize,
                                                  int64_t blockSize) {
  if (!isBlockReductionOp(genericOp))
    return std::nullopt;

  int64_t numLoops = genericOp.getNumLoops();
  SmallVector<int64_t> tileSizes(numLoops, 0);
  auto loopSizes =
      cast<linalg::LinalgOp>(genericOp.getOperation()).computeStaticLoopSizes();

  int64_t remainBlockSize = blockSize;
  auto redDim = getReductionDim(genericOp).value();
  bool mapToWarp = false;
  for (int64_t idx = 0; idx < numLoops && remainBlockSize > 1; ++idx) {
    if (idx == redDim) {
      if ((loopSizes[idx] <= warpSize) &&
          (getOperandReductionDim(*genericOp.getDpsInputOperand(0)).value() ==
           numLoops - 1) &&
          (remainBlockSize >= loopSizes[idx])) {
        tileSizes[idx] = loopSizes[idx];
        remainBlockSize /= loopSizes[idx];
        mapToWarp = true;
      }
    } else {
      int64_t curLoopSize2 = nextPowerOf2(loopSizes[idx]);
      int64_t curBlockSize = std::min(curLoopSize2, remainBlockSize);
      tileSizes[idx] = curLoopSize2 / curBlockSize;
      remainBlockSize /= curBlockSize;
    }
  }

  if ((remainBlockSize == blockSize) ||
      (loopSizes[redDim] == warpSize && remainBlockSize >= warpSize)) {
    tileSizes[redDim] = loopSizes[redDim];
  }

  std::vector<ProducerSelector> fuseCandidates;
  for (OpOperand *opOperand : genericOp.getDpsInputOperands()) {
    ProducerSelector::detectPadOperand(opOperand, fuseCandidates);
  }
  for (OpOperand &opOperand : genericOp.getDpsInitsMutable()) {
    ProducerSelector::detectFillOperand(&opOperand, fuseCandidates);
  }

  auto numTiledLoops = getNumTiledLoops(tileSizes);
  if (numTiledLoops >= 1 && numTiledLoops <= 3) {
    SmallVector<int64_t> mapping(numLoops, -1);
    int64_t dimMapping = static_cast<int64_t>(gpu::MappingId::LinearDim0);
    for (auto &&affineMap : genericOp.getIndexingMapsArray()) {
      if (affineMap.isPermutation()) {
        for (int64_t i = numLoops - 1; i >= 0; i--) {
          auto dim = affineMap.getDimPosition(i);
          if (tileSizes[dim] > 0) {
            mapping[dim] = dimMapping++;
          }
        }
        break;
      }
    }
    mapping.erase(std::remove(mapping.begin(), mapping.end(), -1),
                  mapping.end());
    if (mapping.size() != numTiledLoops)
      return std::nullopt;

    return BlockTileConfig{
        tileSizes,
        llvm::to_vector(llvm::map_range(
            mapping, [](int64_t i) { return static_cast<gpu::MappingId>(i); })),
        fuseCandidates,
        loopSizes[redDim],
        remainBlockSize,
        warpSize,
        mapToWarp};
  }
  return std::nullopt;
}

std::optional<ThreadTileConfig>
getThreadTileConfig(linalg::GenericOp genericOp) {
  if (!isThreadReductionOp(genericOp))
    return std::nullopt;

  int64_t numLoops = genericOp.getNumLoops();
  SmallVector<int64_t> parallelTileSizes(numLoops, 1);
  SmallVector<int64_t> reductionTileSizes(numLoops, 0);
  auto reductionDim = *getReductionDim(genericOp);

  parallelTileSizes[reductionDim] = 0;
  reductionTileSizes[reductionDim] = 1;

  SmallVector<int64_t> unrollFactors =
      cast<linalg::LinalgOp>(genericOp.getOperation()).computeStaticLoopSizes();

  std::vector<ProducerSelector> initOperands;
  for (OpOperand &opOperand : genericOp.getDpsInitsMutable()) {
    ProducerSelector::detectFillOperand(&opOperand, initOperands);
  }

  return ThreadTileConfig{parallelTileSizes, reductionTileSizes, unrollFactors,
                          initOperands};
}

//----------------------------------------------------------------------------//
// transform insertion impl
//----------------------------------------------------------------------------//

void createGPUSplitGridReductionTransformImpl(OpPassManager &pm,
                                              const std::string &anchor,
                                              const std::string &prefix,
                                              int64_t splitFactor) {
  TransformInsertionConfig config;
  config.funcAnchor = anchor;
  config.matchPrefix = prefix;
  config.opFilter = [=](Operation *op) {
    if (auto genericOp = llvm::dyn_cast_or_null<linalg::GenericOp>(op)) {
      return getGridSplitConfig(genericOp, splitFactor).has_value();
    }
    return false;
  };

  config.transformBuilder = [=](ImplicitLocOpBuilder &b, Operation *op,
                                Value pdlV) {
    auto splitConfig =
        getGridSplitConfig(llvm::cast<linalg::GenericOp>(op), splitFactor)
            .value();
    splitConfig.apply(b, pdlV);
  };

  pm.addPass(createGenericTransformInsertionPass(config));
}

void createGPUTileGridReductionTransformImpl(
    OpPassManager &pm, const std::string &anchor, const std::string &prefix,
    int64_t warpSize, int64_t blockSize, bool usingForall) {
  TransformInsertionConfig config;
  config.funcAnchor = anchor;
  config.matchPrefix = prefix;
  config.opFilter = [=](Operation *op) {
    if (auto genericOp = llvm::dyn_cast_or_null<linalg::GenericOp>(op)) {
      return getGridTileConfig(genericOp, warpSize, blockSize).has_value();
    }
    return false;
  };

  config.transformBuilder = [=](ImplicitLocOpBuilder &b, Operation *op,
                                Value pdlV) {
    auto tileConfig = getGridTileConfig(llvm::cast<linalg::GenericOp>(op),
                                        warpSize, blockSize)
                          .value();
    tileConfig.apply(b, pdlV, usingForall);
  };

  pm.addPass(createGenericTransformInsertionPass(config));
}

void createGPUSplitBlockReductionTransformImpl(OpPassManager &pm,
                                               const std::string &anchor,
                                               const std::string &prefix,
                                               int64_t splitFactor,
                                               int64_t warpSize) {
  TransformInsertionConfig config;
  config.funcAnchor = anchor;
  config.matchPrefix = prefix;
  config.opFilter = [=](Operation *op) {
    if (auto genericOp = llvm::dyn_cast_or_null<linalg::GenericOp>(op)) {
      return getBlockSplitConfig(genericOp, splitFactor, warpSize).has_value();
    }
    return false;
  };

  config.transformBuilder = [=](ImplicitLocOpBuilder &b, Operation *op,
                                Value pdlV) {
    auto splitConfig = getBlockSplitConfig(llvm::cast<linalg::GenericOp>(op),
                                           splitFactor, warpSize)
                           .value();
    splitConfig.apply(b, pdlV);
  };

  pm.addPass(createGenericTransformInsertionPass(config));
}

void createGPUTileBlockReductionTransformImpl(
    OpPassManager &pm, const std::string &anchor, const std::string &prefix,
    int64_t warpSize, int64_t blockSize, bool usingForall) {
  TransformInsertionConfig config;
  config.funcAnchor = anchor;
  config.matchPrefix = prefix;
  config.opFilter = [=](Operation *op) {
    if (auto genericOp = llvm::dyn_cast_or_null<linalg::GenericOp>(op)) {
      return getBlockTileConfig(genericOp, warpSize, blockSize).has_value();
    } else if (auto copyOp = llvm::dyn_cast_or_null<linalg::CopyOp>(op)) {
      return copyOp.getNumLoops() == 1;
    }
    return false;
  };

  config.transformBuilder = [=](ImplicitLocOpBuilder &b, Operation *op,
                                Value pdlV) {
    if (auto genericOp = llvm::dyn_cast_or_null<linalg::GenericOp>(op)) {
      auto tileConfig = getBlockTileConfig(llvm::cast<linalg::GenericOp>(op),
                                           warpSize, blockSize)
                            .value();
      tileConfig.apply(b, pdlV, usingForall);
    } else if (auto copyOp = llvm::dyn_cast_or_null<linalg::CopyOp>(op)) {
      auto tileOp = b.create<transform::TileUsingForallOp>(
          /* target */ pdlV,
          /* staticTileSizes */ SmallVector<int64_t>(1, blockSize),
          /* ctor tag */ transform::NumThreadsSpec(),
          /* mapping */
          b.getArrayAttr(gpu::GPUThreadMappingAttr::get(
              b.getContext(), gpu::MappingId::LinearDim0)));
    }
  };

  pm.addPass(createGenericTransformInsertionPass(config));
}

void createGPUTileThreadReductionTransformImpl(OpPassManager &pm,
                                               const std::string &anchor,
                                               const std::string &prefix) {
  TransformInsertionConfig config;
  config.funcAnchor = anchor;
  config.matchPrefix = prefix;
  config.opFilter = [=](Operation *op) {
    if (auto genericOp = llvm::dyn_cast_or_null<linalg::GenericOp>(op)) {
      return getThreadTileConfig(genericOp).has_value();
    }
    return false;
  };

  config.transformBuilder = [=](ImplicitLocOpBuilder &b, Operation *op,
                                Value pdlV) {
    auto tileConfig =
        getThreadTileConfig(llvm::cast<linalg::GenericOp>(op)).value();
    tileConfig.apply(b, pdlV);
  };

  pm.addPass(createGenericTransformInsertionPass(config));
}
} // namespace

void mlir::createGPUSplitGridReductionTransform(
    OpPassManager &pm, const GPUSplitGridReductionOptions &options) {
  invokeOpPassPipelineBuilder(createGPUSplitGridReductionTransformImpl, pm,
                              options.funcAnchor, options.annotatePrefix,
                              options.splitFactor);
}

void mlir::createGPUTileGridReductionTransform(
    OpPassManager &pm, const GPUTileGridReductionOptions &options) {
  invokeOpPassPipelineBuilder(createGPUTileGridReductionTransformImpl, pm,
                              options.funcAnchor, options.annotatePrefix,
                              options.warpSize, options.blockSize,
                              options.usingForall);
}

void mlir::createGPUSplitBlockReductionTransform(
    OpPassManager &pm, const GPUSplitBlockReductionOptions &options) {
  invokeOpPassPipelineBuilder(createGPUSplitBlockReductionTransformImpl, pm,
                              options.funcAnchor, options.annotatePrefix,
                              options.splitFactor, options.warpSize);
}

void mlir::createGPUTileBlockReductionTransform(
    OpPassManager &pm, const GPUTileBlockReductionOptions &options) {
  invokeOpPassPipelineBuilder(createGPUTileBlockReductionTransformImpl, pm,
                              options.funcAnchor, options.annotatePrefix,
                              options.warpSize, options.blockSize,
                              options.usingForall);
}

void mlir::createGPUTileThreadReductionTransform(
    OpPassManager &pm, const GPUTileThreadReductionOptions &options) {
  invokeOpPassPipelineBuilder(createGPUTileThreadReductionTransformImpl, pm,
                              options.funcAnchor, options.annotatePrefix);
}
