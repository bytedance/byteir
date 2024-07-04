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
#include "byteir/Dialect/GPU/Transforms/Utils.h"
#include "byteir/Dialect/Linalg/TransformOps/LinalgExtTransformOps.h"
#include "byteir/Dialect/Transform/IR/TransformExtOps.h"
#include "byteir/Dialect/Transform/Transforms/TransformInsertion.h"
#include "byteir/Dialect/Vector/Transforms/MoveForallRegionIntoWarpOp.h"
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
#include "mlir/Dialect/Vector/TransformOps/VectorTransformOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SmallSet.h"

#include <optional>

using namespace mlir;

namespace {
//----------------------------------------------------------------------------//
// common helpers
//----------------------------------------------------------------------------//
// TODO: move to common header

static constexpr int64_t kGridSplitThreshold = 4096;
static constexpr int64_t kGridTileNumThreshold = 64;
static constexpr int64_t kThreadUnrollThreshold = 8;

constexpr bool isPowerOf2(int64_t n) { return (!(n & (n - 1))); }

constexpr int64_t nextPowerOf2(int64_t n) {
  return (n <= 1) ? 1 : (isPowerOf2(n) ? n : (2 * nextPowerOf2((n + 1) / 2)));
}

bool isMappedToGPUWarps(scf::ForallOp forallOp) {
  if (auto mapping = forallOp.getMappingAttr()) {
    if (llvm::any_of(mapping.getValue(), [](Attribute attr) {
          return isa<gpu::GPUWarpMappingAttr>(attr);
        })) {
      return true;
    }
  }

  return false;
}

bool isMappedToGPUWarps(Operation *op) {
  if (auto forallOp = llvm::dyn_cast_or_null<scf::ForallOp>(op)) {
    return isMappedToGPUWarps(forallOp);
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

int64_t getParallelism(linalg::GenericOp genericOp) {
  SmallVector<unsigned> parallelDims;
  genericOp.getParallelDims(parallelDims);
  auto staticLoopRanges = genericOp.getStaticLoopRanges();
  if (parallelDims.size() == 0) {
    return 1;
  }
  int64_t parallelism = 1;
  for (auto idx : parallelDims) {
    if (ShapedType::isDynamic(staticLoopRanges[idx])) {
      return ShapedType::kDynamic;
    }
    parallelism *= staticLoopRanges[idx];
  }
  return parallelism;
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
    if (auto dimExpr = dyn_cast<AffineDimExpr>(en.value())) {
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
  for (size_t i = 0; i < staticLoopRanges.size(); ++i) {
    if (ShapedType::isDynamic(staticLoopRanges[i])) {
      ret.push_back(i);
    }
  }
  return ret;
}

static void promoteAllTensorsWithinOp(ImplicitLocOpBuilder &b, Value parentOp,
                                      gpu::AddressSpaceAttr memAddrSpace) {
  // get corresponding empty tensor
  auto emptyTensorType = transform::OperationType::get(
      b.getContext(), tensor::EmptyOp::getOperationName());
  auto emptyTensor = b.create<transform::MatchOp>(
      emptyTensorType, parentOp, tensor::EmptyOp::getOperationName());

  // // empty tensor to alloc tensor
  auto allocTensorType = transform::OperationType::get(
      b.getContext(), bufferization::AllocTensorOp::getOperationName());
  auto allocTensor = b.create<transform::EmptyTensorToAllocTensorOp>(
      allocTensorType, emptyTensor);
  auto memorySpaceAttrName =
      bufferization::AllocTensorOp::getMemorySpaceAttrName(OperationName(
          bufferization::AllocTensorOp::getOperationName(), b.getContext()));

  Value paramV = b.create<transform::ParamConstantOp>(
      /* type */ pdl::AttributeType::get(b.getContext()),
      /* value */ memAddrSpace);
  b.create<transform::AnnotateOp>(
      /* target */ allocTensor,
      /* name */ memorySpaceAttrName,
      /* param */ paramV);
}

//----------------------------------------------------------------------------//
// configuration structs
//----------------------------------------------------------------------------//

// tag for linalg operation
static constexpr StringLiteral kGridReduction = "__grid_reduction__";
static constexpr StringLiteral kBlockReduction = "__block_reduction__";
static constexpr StringLiteral kWarpReduction = "__warp_reduction__";
static constexpr StringLiteral kThreadReduction = "__thread_reduction__";

// tag for forall operation
static constexpr StringLiteral kMapInnerLinalgReductionDimToThread =
    "__map_inner_linalg_reduction_dim_to_thread__";

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
  int64_t redDim;
  int64_t numLoops;
  gpu::MappingId mapping;

  void apply(ImplicitLocOpBuilder &b, Value pdlV);
};

struct GridTileConfig {
  SmallVector<int64_t> tileSizes;
  SmallVector<gpu::MappingId> mapping;
  std::vector<ProducerSelector> fuseCandidates;
  int64_t parallelismPerBlock;
  bool asNumThreads;
  bool mapReductionDimToThread;

  void apply(ImplicitLocOpBuilder &b, Value pdlV);
};

struct BlockSplitConfig {
  SmallVector<int64_t> splitFactors;
  SmallVector<int64_t> dimensions;
  SmallVector<int64_t> padDims;
  SmallVector<Attribute> padValues;

  void apply(ImplicitLocOpBuilder &b, Value pdlV);
};

struct BlockTileConfig {
  bool usingTileReduction;
  bool mappingToWarp;
  int64_t numLoops;
  int64_t redDim;
  SmallVector<int64_t> tileSizes;
  SmallVector<gpu::MappingId> mapping;
  std::vector<ProducerSelector> fuseCandidates;

  void apply(ImplicitLocOpBuilder &b, Value pdlV);
};

struct ThreadTileConfig {
  bool applyLoopUnroll;
  utils::IteratorType iterType;
  SmallVector<int64_t> tileSizes;
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
                        bool asNumThreads) {
  SmallVector<Value> toBeFused;
  processProducerSelectors(b, fuseCandidates, toTile, toBeFused);

  transform::TileUsingForallOp tileOp;
  if (asNumThreads) {
    tileOp = b.create<transform::TileUsingForallOp>(
        /* target */ toTile,
        /* numThreads */ tileSizes,
        /* ctor tag */ transform::NumThreadsSpec(),
        /* mapping */ b.getArrayAttr(mapping));
  } else {
    tileOp = b.create<transform::TileUsingForallOp>(
        /* target */ toTile,
        /* staticTileSizes */ tileSizes,
        /* ctor tag */ transform::TileSizesSpec(),
        /* mapping */ b.getArrayAttr(mapping));
  }
  for (auto &&producerOp : toBeFused) {
    b.create<transform::FuseIntoContainingOp>(
        /* producerOp */ producerOp,
        /* containingOp */ tileOp.getForallOp());
  }
  return tileOp;
}

void tileReductionToForallAndFuseImpl(ImplicitLocOpBuilder &b, Value toTile,
                                      const SmallVector<int64_t> &numThreads,
                                      const Attribute mapping,
                                      StringRef annotation) {
  auto tileReductionOp = b.create<transform::TileReductionUsingForallOp>(
      /* target */ toTile,
      /* num_threads */ numThreads,
      /*staticTileSizes*/ SmallVector<int64_t>{},
      /*mapping*/ b.getArrayAttr(mapping));

  b.create<transform::AnnotateOp>(
      /* target */ tileReductionOp.getSplitLinalgOp(),
      /* name */ annotation,
      /* param */ Value());

  b.create<transform::AnnotateOp>(
      /* target */ tileReductionOp.getCombiningLinalgOp(),
      /* name */ annotation,
      /* param */ Value());

  b.create<transform::FuseIntoContainingOp>(
      /* producerOp */ tileReductionOp.getFillOp()[0],
      /* containingOp */ tileReductionOp.getForallOp());
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
    auto mappingAttr = gpu::GPUBlockMappingAttr::get(b.getContext(), mapping);

    SmallVector<int64_t> numThreads(numLoops, 0);
    numThreads[redDim] = splitFactor;

    tileReductionToForallAndFuseImpl(b, pdlV, numThreads, mappingAttr,
                                     kGridReduction);
  } else {
    b.create<transform::AnnotateOp>(
        /* target */ pdlV,
        /* name */ kGridReduction,
        /* param */ Value());
  }
}

void GridTileConfig::apply(ImplicitLocOpBuilder &b, Value pdlV) {
  auto mappingAttrs = llvm::to_vector(
      llvm::map_range(mapping, [&](gpu::MappingId dim) -> Attribute {
        return gpu::GPUBlockMappingAttr::get(b.getContext(), dim);
      }));
  auto tiledOp =
      tileToForallAndFuseImpl(b, pdlV, tileSizes, mappingAttrs, fuseCandidates,
                              /* asNumThreads = */ asNumThreads);

  if (mapReductionDimToThread) {
    b.create<transform::AnnotateOp>(
        /* target */ tiledOp.getForallOp(),
        /* name */ kMapInnerLinalgReductionDimToThread,
        /* param */ Value());
  } else if (!asNumThreads && parallelismPerBlock > 1) {
    SmallVector<int64_t> forTileSizes = tileSizes;
    for (size_t i = 0; i < forTileSizes.size(); ++i) {
      if (forTileSizes[i])
        forTileSizes[i] = 1;
    }

    auto pdlType = pdl::OperationType::get(b.getContext());
    auto fuseOp = b.create<transform::FuseOp>(
        /* transformed */ pdlType,
        /* loops */
        SmallVector<Type>(getNumTiledLoops(forTileSizes), pdlType),
        /* target */ tiledOp.getTiledOp(),
        /* tile_sizes */ b.getI64ArrayAttr(forTileSizes),
        /* tile_interchange */ ArrayAttr());

    b.create<transform::ApplyPatternsOp>(
        fuseOp.getLoops()[0], [](OpBuilder &b, Location loc) {
          b.create<transform::ApplyCanonicalizationPatternsOp>(loc);
          b.create<transform::ApplyFoldUnitExtentDimsViaReshapesPatternsOp>(
              loc);
        });
    b.create<transform::AnnotateOp>(
        /* target */ fuseOp.getTransformed(),
        /* name */ kBlockReduction,
        /* param */ Value());
  }
}

void BlockSplitConfig::apply(ImplicitLocOpBuilder &b, Value pdlV) {
  if (!padDims.empty()) {
    auto padOp = b.create<transform::PadOp>(
        TypeRange{pdlV.getType(), pdlV.getType(), pdlV.getType()}, pdlV,
        /*padding_values=*/b.getArrayAttr(padValues),
        /*padding_dimensions=*/
        b.getI64ArrayAttr(padDims),
        /*pad_to_multiple_of=*/ValueRange{},
        /*static_pad_to_multiple_of=*/b.getDenseI64ArrayAttr({}),
        /*pack_paddings=*/ArrayAttr{},
        /*transpose_paddings=*/ArrayAttr{},
        /*copy_back_op=*/transform::PadOp::kCopyOpNone);
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

void BlockTileConfig::apply(ImplicitLocOpBuilder &b, Value pdlV) {

  if (mappingToWarp) {
    b.create<transform::AnnotateOp>(
        /* target */ pdlV,
        /* name */ kWarpReduction,
        /* param */ Value());
  } else {
    auto mappingAttrs = llvm::to_vector(
        llvm::map_range(mapping, [&](gpu::MappingId dim) -> Attribute {
          return gpu::GPUThreadMappingAttr::get(b.getContext(), dim);
        }));
    if (usingTileReduction) {
      SmallVector<int64_t> numThreads = tileSizes;
      SmallVector<int64_t> staticTileSizes = llvm::to_vector(llvm::map_range(
          tileSizes, [](int64_t val) -> int64_t { return val != 0; }));

      auto tiledRedutionOp = b.create<transform::TileReductionUsingForallOp>(
          /* target */ pdlV,
          /* num_threads */ numThreads,
          /*staticTileSizes*/ staticTileSizes,
          /*mapping*/ b.getArrayAttr(mappingAttrs));

      b.create<transform::FuseIntoContainingOp>(
          /* producerOp */ tiledRedutionOp.getFillOp()[0],
          /* containingOp */ tiledRedutionOp.getForallOp());

      // attch block_redution to combineOp
      b.create<transform::AnnotateOp>(
          /* target */ tiledRedutionOp.getCombiningLinalgOp(),
          /* name */ kBlockReduction,
          /* param */ Value());

      if (numLoops > 1) {
        SmallVector<int64_t> combineTileSizes(numLoops, 1);
        // excluding reduction dim.
        combineTileSizes[redDim] = 0;
        auto tileCombineOp = b.create<transform::TileUsingForOp>(
            /* target */ tiledRedutionOp.getCombiningLinalgOp(),
            /* staticTileSizes */ combineTileSizes);

        b.create<transform::ApplyPatternsOp>(
            tileCombineOp.getLoops()[0], [](OpBuilder &b, Location loc) {
              b.create<transform::ApplyCanonicalizationPatternsOp>(loc);
              b.create<transform::ApplyFoldUnitExtentDimsViaReshapesPatternsOp>(
                  loc);
            });

        b.create<transform::AnnotateOp>(
            /* target */ tileCombineOp.getTiledLinalgOp(),
            /* name */ kBlockReduction,
            /* param */ Value());
      }

      {
        // get corresponding empty tensor
        auto forall = b.create<transform::GetParentOp>(
            tiledRedutionOp.getForallOp().getType(),
            tiledRedutionOp.getForallOp(),
            /* isolated_from_above */ false,
            /* allow_empty_results */ false,
            /* op_name */ b.getStringAttr(scf::ForallOp::getOperationName()),
            /* deduplicate */ false,
            /* nth_parent */ 1);
        auto workgroupMemoryAddressSpace = gpu::AddressSpaceAttr::get(
            b.getContext(), gpu::GPUDialect::getWorkgroupAddressSpace());

        promoteAllTensorsWithinOp(b, forall, workgroupMemoryAddressSpace);
      }
    } else {
      auto tiledOp =
          tileToForallAndFuseImpl(b, pdlV, tileSizes, mappingAttrs,
                                  fuseCandidates, /* asNumThreads = */ false);
    }
  }
}

void ThreadTileConfig::apply(ImplicitLocOpBuilder &b, Value pdlV) {
  auto pdlType = pdl::OperationType::get(b.getContext());
  auto numTiledParallelLoops = getNumTiledLoops(tileSizes);
  SmallVector<Value> loops;
  if (iterType == utils::IteratorType::parallel) {
    auto fuseOp = b.create<transform::FuseOp>(
        /* transformed */ pdlType,
        /* loops */
        SmallVector<Type>(numTiledParallelLoops, pdlType),
        /* target */ pdlV,
        /* tile_sizes */ b.getI64ArrayAttr(tileSizes),
        /* tile_interchange */ ArrayAttr());
    loops = fuseOp.getLoops();
    pdlV = fuseOp.getTransformed();
  } else {
    auto tileOp = b.create<transform::TileUsingForOp>(
        /* target */ pdlV,
        /* tileSizes */ tileSizes);
    for (auto loop : tileOp.getLoops()) {
      loops.emplace_back(loop);
    }
  }

  if (applyLoopUnroll) {
    for (auto &&[loop, factor] :
         llvm::reverse(llvm::zip(loops, unrollFactors))) {
      b.create<transform::LoopUnrollOp>(loop, factor);
    }
  }
}

//----------------------------------------------------------------------------//
// codegen strategies
//----------------------------------------------------------------------------//

bool isReductionOp(linalg::GenericOp genericOp) {
  if (genericOp.getNumReductionLoops() != 1)
    return false;

  if (!llvm::all_of(genericOp.getIndexingMapsArray(), [](AffineMap affineMap) {
        return affineMap.isProjectedPermutation(
            /* allowZeroInResults */ false);
      }))
    return false;

  return true;
}

bool isRedDimInInnermostLoop(linalg::GenericOp genericOp) {
  if (!isReductionOp(genericOp))
    return false;
  int64_t numLoops = genericOp.getNumLoops();
  auto maybeRedDim = getReductionDim(genericOp);
  if (!maybeRedDim.has_value()) {
    return false;
  }
  int64_t redDim = maybeRedDim.value();
  for (auto &&affineMap : genericOp.getIndexingMapsArray()) {
    if (affineMap.isPermutation()) {
      auto dim = affineMap.getDimPosition(numLoops - 1);
      if (dim == redDim) {
        return true;
      }
      break;
    }
  }
  return false;
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

bool isMappedReductionToThread(linalg::GenericOp genericOp) {
  if (!isReductionOp(genericOp))
    return false;

  if (auto forallOp =
          llvm::dyn_cast_or_null<scf::ForallOp>(genericOp->getParentOp())) {
    return forallOp->hasAttr(kMapInnerLinalgReductionDimToThread);
  }
  return false;
}

bool isWarpReductionOp(linalg::GenericOp genericOp) {
  if (!isReductionOp(genericOp))
    return false;

  // early return for manual tag
  if (genericOp->hasAttr(kWarpReduction))
    return true;

  // nested in op which is mapped to GPU warp
  if (isMappedToGPUWarps(genericOp->getParentOp()))
    return true;

  return false;
}

bool isThreadReductionOp(linalg::GenericOp genericOp) {
  if (!isReductionOp(genericOp))
    return false;

  // early return for manual tag
  if (genericOp->hasAttr(kThreadReduction))
    return true;

  if (auto forallOp = genericOp->getParentOfType<scf::ForallOp>()) {
    return isMappedToGPUThreads(forallOp);
  }

  // nested in op which is mapped to GPU threads
  if (isMappedToGPUThreads(genericOp->getParentOp()))
    return true;

  return false;
}

std::optional<GridSplitConfig> getGridSplitConfig(linalg::GenericOp genericOp,
                                                  int64_t splitFactor) {
  if (!isGridReductionOp(genericOp))
    return std::nullopt;

  int64_t numLoops = genericOp.getNumLoops();
  auto redDim = *getReductionDim(genericOp);
  auto staticLoopRanges = genericOp.getStaticLoopRanges();
  int64_t parallelism = getParallelism(genericOp);

  if (parallelism > 1 || parallelism == ShapedType::kDynamic) {
    return std::nullopt;
  }

  if (!isRedDimInInnermostLoop(genericOp)) {
    return std::nullopt;
  }

  int64_t redDimSize = staticLoopRanges[redDim];
  if (isRedDimInInnermostLoop(genericOp)) {
    if (ShapedType::isDynamic(redDimSize) ||
        staticLoopRanges[redDim] <= kGridSplitThreshold) {
      return std::nullopt;
    }
  }

  // at least 2:  split reduction & grid tile
  int64_t blockMappingNum = std::max(numLoops, static_cast<int64_t>(2));
  return GridSplitConfig{splitFactor, redDim, numLoops,
                         static_cast<gpu::MappingId>(
                             static_cast<int64_t>(gpu::MappingId::LinearDim0) +
                             blockMappingNum - 1)};
}

std::optional<GridTileConfig> getGridTileConfig(linalg::GenericOp genericOp,
                                                int64_t warpSize,
                                                int64_t blockSize) {
  if (!isGridReductionOp(genericOp))
    return std::nullopt;

  int64_t numLoops = genericOp.getNumLoops();
  SmallVector<int64_t> tileSizes(numLoops, 1);
  auto redDim = getReductionDim(genericOp).value();
  int64_t totalParallelism = getParallelism(genericOp);
  tileSizes[redDim] = 0;

  bool asNumThreads = false;
  bool mapReductionDimToThread = true;
  auto loopSizes =
      cast<linalg::LinalgOp>(genericOp.getOperation()).getStaticLoopRanges();

  int64_t parallelismPerBlock = blockSize;
  int64_t redDimSize = loopSizes[redDim];

  if (isRedDimInInnermostLoop(genericOp)) {
    if (!ShapedType::isDynamic(redDimSize) && redDimSize < warpSize &&
        totalParallelism / blockSize >= kGridTileNumThreshold) {
      parallelismPerBlock = blockSize;
      mapReductionDimToThread = true;
    } else {
      parallelismPerBlock = 1;
      mapReductionDimToThread = false;
    }
  } else {
    if (!ShapedType::isDynamic(totalParallelism)) {
      while (totalParallelism / parallelismPerBlock < kGridTileNumThreshold &&
             parallelismPerBlock > 1) {
        parallelismPerBlock /= 2;
      }
    }
  }

  if (parallelismPerBlock == 1) {
    mapReductionDimToThread = false;
  }

  SmallVector<unsigned> parallelDims;
  genericOp.getParallelDims(parallelDims);
  int64_t remainParallelism = parallelismPerBlock;
  int64_t lastTilingDim = -1;
  for (auto &&affineMap : genericOp.getIndexingMapsArray()) {
    if (affineMap.isPermutation()) {
      for (int64_t i = numLoops - 1; i >= 0; --i) {
        if (remainParallelism == 1) {
          break;
        }
        auto dim = affineMap.getDimPosition(i);
        if (llvm::find(parallelDims, dim) == parallelDims.end())
          continue;
        if (ShapedType::isDynamic(loopSizes[dim])) {
          tileSizes[dim] = remainParallelism;
          remainParallelism = 1;
        } else {
          int64_t dimSize = nextPowerOf2(loopSizes[dim]);
          if (dimSize <= remainParallelism) {
            tileSizes[dim] = 0;
            remainParallelism /= dimSize;
          } else {
            tileSizes[dim] = remainParallelism;
            remainParallelism = 1;
          }
        }
        lastTilingDim = dim;
      }
      break;
    }
  }

  std::vector<ProducerSelector> fuseCandidates;
  for (OpOperand &opOperand : genericOp.getDpsInitsMutable()) {
    ProducerSelector::detectFillOperand(&opOperand, fuseCandidates);
  }

  auto numTiledLoops = getNumTiledLoops(tileSizes);
  if (numTiledLoops == 0) {
    numTiledLoops = 1;
    if (lastTilingDim != -1) {
      // parallelism is too small.
      // using last tiling dimension to generate forallOp with unit mapping
      // size.
      tileSizes[lastTilingDim] = loopSizes[lastTilingDim];
    } else if (genericOp.hasSingleReductionLoop()) {
      if (ShapedType::isDynamic(loopSizes[redDim])) {
        asNumThreads = true;
        tileSizes[redDim] = 1;
      } else {
        tileSizes[redDim] = loopSizes[redDim];
      }
    } else {
      return std::nullopt;
    }
  }

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

    return GridTileConfig{
        tileSizes,
        llvm::to_vector(llvm::map_range(
            mapping, [](int64_t i) { return static_cast<gpu::MappingId>(i); })),
        fuseCandidates,
        parallelismPerBlock,
        asNumThreads,
        mapReductionDimToThread};
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

  for (; splitFactor > 2; splitFactor >>= 1) {
    splitFactors.push_back(splitFactor / 2);
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
      cast<linalg::LinalgOp>(genericOp.getOperation()).getStaticLoopRanges();

  int64_t remainBlockSize = blockSize;
  auto redDim = getReductionDim(genericOp).value();

  bool usingTileReduction = false;
  bool mappingToWarp = false;
  // mapping to warp redution directly
  int64_t redDimSize = loopSizes[redDim];
  if (numLoops == 1 && redDimSize != ShapedType::kDynamic &&
      redDimSize <= warpSize && isPowerOf2(redDimSize)) {
    mappingToWarp = true;
    return BlockTileConfig{usingTileReduction,
                           mappingToWarp,
                           numLoops,
                           redDim,
                           tileSizes,
                           SmallVector<gpu::MappingId>{},
                           std::vector<ProducerSelector>{}};
  } else if (isMappedReductionToThread(genericOp)) {
    for (int64_t i = 0; i < numLoops; ++i)
      tileSizes[i] = 1;
    tileSizes[redDim] = 0;
  } else {
    usingTileReduction = true;
    tileSizes[redDim] = blockSize;
    if (!ShapedType::isDynamic(redDimSize)) {
      if (redDimSize <= blockSize) {
        tileSizes[redDim] = std::max(nextPowerOf2(redDimSize), warpSize);
      }

      while (tileSizes[redDim] * 2 > redDimSize &&
             tileSizes[redDim] / 2 >= warpSize) {
        tileSizes[redDim] /= 2;
      }
    }
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
    if (usingTileReduction && mapping.size() != 1)
      return std::nullopt;

    if (!usingTileReduction && mapping.size() != numTiledLoops)
      return std::nullopt;

    return BlockTileConfig{
        usingTileReduction,
        mappingToWarp,
        numLoops,
        redDim,
        tileSizes,
        llvm::to_vector(llvm::map_range(
            mapping, [](int64_t i) { return static_cast<gpu::MappingId>(i); })),
        fuseCandidates};
  }
  return std::nullopt;
}

std::optional<ThreadTileConfig>
getThreadTileConfig(linalg::GenericOp genericOp,
                    const utils::IteratorType &iterType) {
  if (!isThreadReductionOp(genericOp))
    return std::nullopt;

  bool applyLoopUnroll = true;
  SmallVector<int64_t> unrollFactors;
  int64_t numLoops = genericOp.getNumLoops();
  SmallVector<int64_t> tileSizes(numLoops, 0);
  auto reductionDim = *getReductionDim(genericOp);
  SmallVector<unsigned> dims;
  if (iterType == utils::IteratorType::parallel) {
    genericOp.getParallelDims(dims);
  } else {
    genericOp.getReductionDims(dims);
  }

  if (dims.size() == 0) {
    return std::nullopt;
  }

  SmallVector<int64_t> loopSizes =
      cast<linalg::LinalgOp>(genericOp.getOperation()).getStaticLoopRanges();

  for (auto d : dims) {
    tileSizes[d] = 1;
    unrollFactors.emplace_back(loopSizes[d]);
    if (ShapedType::isDynamic(loopSizes[d]) ||
        loopSizes[d] > kThreadUnrollThreshold) {
      applyLoopUnroll = false;
    }
  }

  std::vector<ProducerSelector> initOperands;
  for (OpOperand &opOperand : genericOp.getDpsInitsMutable()) {
    ProducerSelector::detectFillOperand(&opOperand, initOperands);
  }

  return ThreadTileConfig{applyLoopUnroll, iterType, tileSizes, unrollFactors,
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

void createGPUTileGridReductionTransformImpl(OpPassManager &pm,
                                             const std::string &anchor,
                                             const std::string &prefix,
                                             int64_t warpSize,
                                             int64_t blockSize) {
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
    tileConfig.apply(b, pdlV);
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

void createGPUTileBlockReductionTransformImpl(OpPassManager &pm,
                                              const std::string &anchor,
                                              const std::string &prefix,
                                              int64_t warpSize,
                                              int64_t blockSize) {
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
      tileConfig.apply(b, pdlV);
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

void createGPUTileSplitWarpReductionTransformImpl(OpPassManager &pm,
                                                  const std::string &anchor,
                                                  const std::string &prefix,
                                                  int64_t blockSize,
                                                  int64_t warpSize) {
  TransformInsertionConfig config;
  config.funcAnchor = anchor;
  config.matchPrefix = prefix;
  config.opFilter = [=](Operation *op) {
    if (auto genericOp = llvm::dyn_cast_or_null<linalg::GenericOp>(op)) {
      if (isBlockReductionOp(genericOp)) {
        int64_t numLoops = genericOp.getNumLoops();
        int64_t redDim = -1;
        SmallVector<unsigned> reductionDims;
        genericOp.getReductionDims(reductionDims);
        if (reductionDims.size() == 1) {
          redDim = reductionDims[0];
        }
        auto staticLoopRanges = genericOp.getStaticLoopRanges();
        int64_t redDimSize = staticLoopRanges[redDim];

        if (numLoops == 1 && redDim != -1 && redDimSize % warpSize == 0 &&
            redDimSize > warpSize)
          return true;
      }
    }
    return false;
  };

  config.transformBuilder = [=](ImplicitLocOpBuilder &b, Operation *op,
                                Value pdlV) {
    auto genericOp = llvm::dyn_cast_or_null<linalg::GenericOp>(op);
    ::mlir::OperandRange initRange = genericOp.getDpsInits();
    int64_t redDim = *getReductionDim(genericOp);
    auto staticLoopRanges = genericOp.getStaticLoopRanges();
    int64_t redDimSize = staticLoopRanges[redDim];
    int64_t splitFactor = redDimSize / warpSize;
    // tile redution dim & mapping parallel dim to warp
    auto mappingWarpAttr = gpu::GPUWarpMappingAttr::get(
        b.getContext(), gpu::MappingId::LinearDim0);

    SmallVector<int64_t> numThreads{splitFactor};

    auto tileReductionOp = b.create<transform::TileReductionUsingForallOp>(
        /* target */ pdlV,
        /* num_threads */ numThreads,
        /*staticTileSizes*/ SmallVector<int64_t>{},
        /*mapping*/ b.getArrayAttr(mappingWarpAttr));

    b.create<transform::AnnotateOp>(
        /* target */ tileReductionOp.getSplitLinalgOp(),
        /* name */ kWarpReduction,
        /* param */ Value());

    b.create<transform::AnnotateOp>(
        /* target */ tileReductionOp.getCombiningLinalgOp(),
        /* name */ kWarpReduction,
        /* param */ Value());

    int64_t initStart = initRange.getBeginOperandIndex();
    int64_t initEnd = initStart + initRange.size();
    for (int64_t i = initStart; i < initEnd; ++i) {
      // get the neutral tensor.empty()
      auto producer = b.create<transform::GetProducerOfOperand>(
          /* producer type */ transform::OperationType::get(
              b.getContext(), tensor::EmptyOp::getOperationName()),
          /* target */ tileReductionOp.getFillOp()[0],
          /* operand number */ i);

      // promote to WorkGroup
      auto workgroupMemoryAddressSpace = gpu::AddressSpaceAttr::get(
          b.getContext(), gpu::GPUDialect::getWorkgroupAddressSpace());

      promoteAllTensorsWithinOp(b, producer, workgroupMemoryAddressSpace);
    }

    // fuse fill
    b.create<transform::FuseIntoContainingOp>(
        /* producerOp */ tileReductionOp.getFillOp()[0],
        /* containingOp */ tileReductionOp.getForallOp());
  };

  pm.addPass(createGenericTransformInsertionPass(config));
}

void createGPUTileWarpReductionTransformImpl(OpPassManager &pm,
                                             const std::string &anchor,
                                             const std::string &prefix,
                                             int64_t warpSize) {
  TransformInsertionConfig config;
  config.funcAnchor = anchor;
  config.matchPrefix = prefix;
  config.opFilter = [=](Operation *op) {
    if (auto genericOp = llvm::dyn_cast_or_null<linalg::GenericOp>(op)) {
      if (isWarpReductionOp(genericOp) || isBlockReductionOp(genericOp)) {
        int64_t numLoops = genericOp.getNumLoops();
        int64_t redDim = -1;
        SmallVector<unsigned> reductionDims;
        genericOp.getReductionDims(reductionDims);
        if (reductionDims.size() == 1) {
          redDim = reductionDims[0];
        } else {
          return false;
        }
        auto staticLoopRanges = genericOp.getStaticLoopRanges();
        if (staticLoopRanges[redDim] != ShapedType::kDynamic && numLoops == 1 &&
            redDim != -1 && staticLoopRanges[redDim] <= warpSize)
          return true;
      }
    }
    return false;
  };

  config.transformBuilder = [=](ImplicitLocOpBuilder &b, Operation *op,
                                Value pdlV) {
    auto genericOp = llvm::cast<linalg::GenericOp>(op);
    scf::ForallOp parentOp = genericOp->getParentOfType<scf::ForallOp>();
    Value toVectorize = pdlV;
    Value forall = b.create<transform::GetParentOp>(
        pdlV.getType(), pdlV,
        /* isolated_from_above */ false,
        /* allow_empty_results */ false,
        /* op_name */ b.getStringAttr(scf::ForallOp::getOperationName()),
        /* deduplicate */ false,
        /* nth_parent */ 1);
    if (!parentOp || !isMappedToGPUWarps(parentOp)) {
      std::vector<ProducerSelector> fuseCandidates;
      for (OpOperand &opOperand : genericOp.getDpsInitsMutable()) {
        ProducerSelector::detectFillOperand(&opOperand, fuseCandidates);
      }

      SmallVector<Attribute> mapping{gpu::GPUWarpMappingAttr::get(
          b.getContext(), gpu::MappingId::LinearDim0)};
      SmallVector<int64_t> numThreads(1, 1);
      auto tileOp =
          tileToForallAndFuseImpl(b, pdlV, numThreads, mapping, fuseCandidates,
                                  /* asNumThreads = */ true);
      forall = tileOp.getForallOp();
      toVectorize = tileOp.getTiledOp();
    }

    // convert inner redution to vector multi_reduction
    b.create<transform::VectorizeOp>(
        /* target */ toVectorize,
        /* vector_sizes */ ValueRange{},
        /* static_vector_sizes */ SmallVector<int64_t>{},
        /* vectorize_nd_extract */ b.getUnitAttr(),
        /* scalable_sizes */ SmallVector<bool>{});

    // lower vector.multi_reduction to vector.reduction
    b.create<transform::ApplyPatternsOp>(
        forall, [](OpBuilder &b, Location loc) {
          b.create<transform::ApplyLowerMultiReductionPatternsOp>(
              loc, vector::VectorMultiReductionLowering::InnerReduction);
        });
  };
  pm.addPass(createGenericTransformInsertionPass(config));
}

void createGPUTileThreadReductionTransformImpl(
    OpPassManager &pm, const std::string &anchor, const std::string &prefix,
    const utils::IteratorType iterType) {
  TransformInsertionConfig config;
  config.funcAnchor = anchor;
  config.matchPrefix = prefix;
  config.opFilter = [=](Operation *op) {
    if (auto genericOp = llvm::dyn_cast_or_null<linalg::GenericOp>(op)) {
      return getThreadTileConfig(genericOp, iterType).has_value();
    }
    return false;
  };

  config.transformBuilder = [=](ImplicitLocOpBuilder &b, Operation *op,
                                Value pdlV) {
    auto tileConfig =
        getThreadTileConfig(llvm::cast<linalg::GenericOp>(op), iterType)
            .value();
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
                              options.warpSize, options.blockSize);
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
                              options.warpSize, options.blockSize);
}

void mlir::createGPUTileSplitWarpReductionTransform(
    OpPassManager &pm, const GPUTileSplitWarpReductionOptions &options) {
  invokeOpPassPipelineBuilder(createGPUTileSplitWarpReductionTransformImpl, pm,
                              options.funcAnchor, options.annotatePrefix,
                              options.blockSize, options.warpSize);
}

void mlir::createGPUTileWarpReductionTransform(
    OpPassManager &pm, const GPUTileWarpReductionOptions &options) {
  invokeOpPassPipelineBuilder(createGPUTileWarpReductionTransformImpl, pm,
                              options.funcAnchor, options.annotatePrefix,
                              options.warpSize);
}

void mlir::createGPUTileThreadReductionTransform(
    OpPassManager &pm, const GPUTileThreadReductionOptions &options) {
  invokeOpPassPipelineBuilder(createGPUTileThreadReductionTransformImpl, pm,
                              options.funcAnchor, options.annotatePrefix,
                              options.iteratorType);
}
