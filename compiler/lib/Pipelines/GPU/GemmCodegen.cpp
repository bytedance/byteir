//===- GemmCodegen.cpp ---------------------------------------*--- C++ -*-===//
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

#include "byteir/Pipelines/GPU/GemmCodegen.h"
#include "byteir/Conversion/ToGPU/ToGPU.h"
#include "byteir/Conversion/ToLLVM/ToLLVM.h"
#include "byteir/Dialect/GPU/Transforms/Utils.h"
#include "byteir/Dialect/Linalg/TransformOps/LinalgExtTransformOps.h"
#include "byteir/Dialect/Linalg/Transforms/LinalgPrefetch.h"
#include "byteir/Dialect/Transform/IR/TransformExtOps.h"
#include "byteir/Dialect/Transform/Transforms/TransformInsertion.h"
#include "byteir/Pipelines/Common/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/MemRef/TransformOps/MemRefTransformOps.h"
#include "mlir/Dialect/NVGPU/TransformOps/NVGPUTransformOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SmallSet.h"

#include <optional>

using namespace mlir;

namespace {

constexpr StringRef getLinalgToGPUAttrName() { return "__byteir_to_gpu__"; }

constexpr StringRef getLinalgMMALevelAttrName() {
  return "__byteir_mma_level__";
}

constexpr StringRef getMMAPatternAttrName() { return "__byteir_mma__"; }

constexpr StringRef getLinalgTargetAttrName() { return "__byteir_target__"; }

constexpr StringRef getEpilogueMarker() { return "__byteir_epilogue__"; }

void createGPUTileGemmTransformImpl(OpPassManager &pm,
                                    const std::string &anchor,
                                    const std::string &prefix) {
  TransformInsertionConfig config;
  config.funcAnchor = anchor;
  config.matchPrefix = prefix;
  config.opFilter = [=](Operation *op) {
    if (!isLinalgOpMatmul(op))
      return false;
    return true;
  };

  config.transformBuilder = [=](ImplicitLocOpBuilder &b, Operation *op,
                                Value pdlV) {
    func::FuncOp funcOp = op->getParentOfType<func::FuncOp>();
    linalg::LinalgOp linalgOp = cast<linalg::LinalgOp>(op);
    Operation *user = *linalgOp->getUsers().begin();
    bool hasEpilogue = isa<linalg::GenericOp>(user);

    if (hasEpilogue) {
      setMarker(user, getEpilogueMarker());
    }

    bool isBMM = linalgOp.getNumParallelLoops() == 3;

    SmallVector<int64_t, 3> tileSizeConfig = getGemmTileSize(funcOp).value();

    auto func = b.create<transform::GetParentOp>(
        pdlV.getType(), pdlV,
        /* isolated_from_above */ false,
        /* allow_empty_results */ false,
        /* op_name */ b.getStringAttr(func::FuncOp::getOperationName()),
        /* deduplicate */ false,
        /* nth_parent */ 1);

    auto anyType = transform::AnyOpType::get(b.getContext());
    auto linalgFillType = transform::OperationType::get(
        b.getContext(), linalg::FillOp::getOperationName());
    auto linalgFill = b.create<transform::MatchOp>(
        linalgFillType, func, linalg::FillOp::getOperationName());

    Value mmaLevel = b.create<transform::ParamConstantOp>(
        /* type */ pdl::AttributeType::get(b.getContext()),
        /* value */ b.getStringAttr("Threadblock"));
    Value target = b.create<transform::ParamConstantOp>(
        /* type */ pdl::AttributeType::get(b.getContext()),
        /* value */ b.getStringAttr("nv_sm_80"));

    SmallVector<int64_t> mappingIdx;
    if (isBMM) {
      mappingIdx = {2, 1, 0};
    } else {
      mappingIdx = {1, 0};
    }
    auto mapping = llvm::to_vector(llvm::map_range(
        mappingIdx, [](int64_t i) { return static_cast<gpu::MappingId>(i); }));
    auto mappingAttrs = llvm::to_vector(
        llvm::map_range(mapping, [&](gpu::MappingId dim) -> Attribute {
          return gpu::GPUBlockMappingAttr::get(b.getContext(), dim);
        }));

    SmallVector<int64_t> parrallelTileSizes;
    if (isBMM) {
      parrallelTileSizes = {1, tileSizeConfig[0], tileSizeConfig[1]};
    } else {
      parrallelTileSizes = {tileSizeConfig[0], tileSizeConfig[1]};
    }
    Value tiledMatmulOp;
    if (hasEpilogue) {
      auto linalgGenericType = transform::OperationType::get(
          b.getContext(), linalg::GenericOp::getOperationName());
      auto epilogue = b.create<transform::MatchOp>(
          linalgGenericType, func,
          b.getStrArrayAttr({linalg::GenericOp::getOperationName()}),
          /*matchInterfaceEnum=*/transform::MatchInterfaceEnumAttr(),
          /*opAttrs=*/
          b.getDictionaryAttr({NamedAttribute(
              b.getStringAttr(getEpilogueMarker()), b.getUnitAttr())}),
          /*filterResultType=*/TypeAttr(),
          /*filterOperandTYpes=*/ArrayAttr());

      transform::TileUsingForallOp tileOp =
          b.create<transform::TileUsingForallOp>(
              /* target */ epilogue,
              /* staticTileSizes */ parrallelTileSizes,
              /* ctor tag */ transform::TileSizesSpec(),
              /* mapping */ b.getArrayAttr(mappingAttrs));
      transform::FuseIntoContainingOp fuse =
          b.create<transform::FuseIntoContainingOp>(
              /* producerOp */ pdlV,
              /* containingOp */ tileOp.getForallOp());
      b.create<transform::FuseIntoContainingOp>(
          /* producerOp */ linalgFill,
          /* containingOp */ fuse.getNewContainingOp());
      tiledMatmulOp = fuse.getFusedOp();
    } else {
      transform::TileUsingForallOp tileOp =
          b.create<transform::TileUsingForallOp>(
              /* target */ pdlV,
              /* staticTileSizes */ parrallelTileSizes,
              /* ctor tag */ transform::TileSizesSpec(),
              /* mapping */ b.getArrayAttr(mappingAttrs));

      b.create<transform::FuseIntoContainingOp>(
          /* producerOp */ linalgFill,
          /* containingOp */ tileOp.getForallOp());
      tiledMatmulOp = tileOp.getTiledOp();
    }

    SmallVector<int64_t> reductionTileSizes;
    if (isBMM)
      reductionTileSizes = {0, 0, 0, tileSizeConfig[2]};
    else
      reductionTileSizes = {0, 0, tileSizeConfig[2]};
    auto tileKMatmulOp =
        b.create<transform::TileUsingForOp>(tiledMatmulOp, reductionTileSizes);
    auto matmulKOp = tileKMatmulOp.getTiledLinalgOp();

    b.create<transform::AnnotateOp>(matmulKOp, getLinalgMMALevelAttrName(),
                                    mmaLevel);
    b.create<transform::AnnotateOp>(matmulKOp, getLinalgTargetAttrName(),
                                    target);
    b.create<transform::AnnotateOp>(matmulKOp, getMMAPatternAttrName(),
                                    Value());
  };

  pm.addPass(createGenericTransformInsertionPass(config));
}

} // namespace

void mlir::createGPUTileGemmTransform(OpPassManager &pm,
                                      const GPUGemmGeneralOptions &options) {
  invokeOpPassPipelineBuilder(createGPUTileGemmTransformImpl, pm,
                              options.funcAnchor, options.annotatePrefix);
}

namespace {

void createGPUAddGemmCodegenLoweringConfigTransformImpl(
    OpPassManager &pm, const std::string &anchor, const std::string &prefix,
    ArrayRef<int64_t> tileSizeConfig, ArrayRef<int64_t> workgroupSize,
    int64_t stages) {

  SmallVector<int64_t> tileSizeConfigVec{tileSizeConfig};
  SmallVector<int64_t> workgroupSizeVec{workgroupSize};

  TransformInsertionConfig config;
  config.funcAnchor = anchor;
  config.matchPrefix = prefix;

  config.opFilter = [=](Operation *op) {
    if (isLinalgOpMatmul(op)) {
      // TODO: check if the matmul op is already annotated
      // TODO: Add different lowering config for different matmul op size
      return true;
    }
    return false;
  };

  config.transformBuilder = [=](ImplicitLocOpBuilder &b, Operation *op,
                                Value pdlV) {
    // auto linalgOp = llvm::cast<linalg::LinalgOp>(op);
    auto tileSizeConfigAttrs = b.getAttr<ArrayAttr>(llvm::to_vector(
        llvm::map_range(tileSizeConfigVec, [&](int64_t i) -> Attribute {
          return b.getI64IntegerAttr(i);
        })));
    auto workgroupSizeAttrs = b.getAttr<ArrayAttr>(llvm::to_vector(
        llvm::map_range(workgroupSizeVec, [&](int64_t i) -> Attribute {
          return b.getI64IntegerAttr(i);
        })));
    auto stagesAttr = b.getI64IntegerAttr(stages);

    auto func = b.create<transform::GetParentOp>(
        pdlV.getType(), pdlV,
        /* isolated_from_above */ true,
        /* allow_empty_results */ false,
        /* op_name */ b.getStringAttr(func::FuncOp::getOperationName()),
        /* deduplicate */ false,
        /* nth_parent */ 1);

    Value tileSizeConfigValue = b.create<transform::ParamConstantOp>(
        /* type */ pdl::AttributeType::get(b.getContext()),
        /* value */ tileSizeConfigAttrs);
    Value workgroupSizeValue = b.create<transform::ParamConstantOp>(
        /* type */ pdl::AttributeType::get(b.getContext()),
        /* value */ workgroupSizeAttrs);
    Value stagesValue = b.create<transform::ParamConstantOp>(
        /* type */ pdl::AttributeType::get(b.getContext()),
        /* value */ stagesAttr);

    b.create<transform::AnnotateOp>(func, getGemmTileConfigAttrName(),
                                    tileSizeConfigValue);
    b.create<transform::AnnotateOp>(func, getGemmBlockSizeAttrName(),
                                    workgroupSizeValue);
    b.create<transform::AnnotateOp>(func, getGemmPipelineDepthAttrName(),
                                    stagesValue);
  };
  pm.addPass(createGenericTransformInsertionPass(config));
}
} // namespace

void mlir::createGPUAddGemmCodegenLoweringConfigTransform(
    OpPassManager &pm, const GPUGemmCodegenConfigOptions &options) {
  invokeOpPassPipelineBuilder(
      createGPUAddGemmCodegenLoweringConfigTransformImpl, pm,
      options.funcAnchor, options.annotatePrefix, options.tileSizeConfig,
      options.workgroupSize, options.stages);
}

namespace {

int numIterations(scf::ForOp forOp) {
  Value lowerBound = forOp.getLowerBound();
  Value upperBound = forOp.getUpperBound();
  Value step = forOp.getStep();

  // get def constant value
  auto defLowerBound = lowerBound.getDefiningOp<arith::ConstantOp>();
  auto defUpperBound = upperBound.getDefiningOp<arith::ConstantOp>();
  auto defStep = step.getDefiningOp<arith::ConstantOp>();

  if (defLowerBound && defUpperBound && defStep) {
    auto lowerBoundValue = defLowerBound.getValue();
    auto upperBoundValue = defUpperBound.getValue();
    auto stepValue = defStep.getValue();

    auto lowerBoundInt = cast<IntegerAttr>(lowerBoundValue).getInt();
    auto upperBoundInt = cast<IntegerAttr>(upperBoundValue).getInt();
    auto stepInt = cast<IntegerAttr>(stepValue).getInt();
    return (upperBoundInt - lowerBoundInt) / stepInt;
  }
  return -1;
}
void createGPUPipeliningTransformImpl(OpPassManager &pm,
                                      const std::string &anchor,
                                      const std::string &prefix) {

  TransformInsertionConfig config;
  config.funcAnchor = anchor;
  config.matchPrefix = prefix;

  config.opFilter = [=](Operation *op) {
    if (auto forallOp = llvm::dyn_cast_or_null<scf::ForallOp>(op)) {
      if (!isMappedToGPUBlocks(forallOp)) {
        return false;
      }
      func::FuncOp funcOp = forallOp->getParentOfType<func::FuncOp>();
      auto pipelineStageOptional = getGemmPipelineDepth(funcOp);
      if (!pipelineStageOptional) {
        return false;
      }
      SmallVector<scf::ForOp> forOps;
      forallOp.walk([&](scf::ForOp forOp) { forOps.push_back(forOp); });
      if (forOps.size() != 1)
        return false;
      scf::ForOp forOp = forOps[0];
      if (numIterations(forOp) <= pipelineStageOptional.value())
        return false;
      else
        return true;
    }
    return false;
  };

  config.transformBuilder = [=](ImplicitLocOpBuilder &b, Operation *op,
                                Value pdlV) {
    func::FuncOp funcOp = op->getParentOfType<func::FuncOp>();
    auto pipelineStageOptional = getGemmPipelineDepth(funcOp);
    if (!pipelineStageOptional) {
      return;
    }
    int pipelineStage = *pipelineStageOptional;
    auto anyType = transform::AnyOpType::get(b.getContext());

    auto memrefAllocType = transform::OperationType::get(
        b.getContext(), memref::AllocOp::getOperationName());
    auto memrefAllocMatrixLHS = b.create<transform::MatchOp>(
        memrefAllocType, pdlV,
        b.getStrArrayAttr({memref::AllocOp::getOperationName()}),
        /*matchInterfaceEnum=*/transform::MatchInterfaceEnumAttr(),
        /*opAttrs=*/
        b.getDictionaryAttr({NamedAttribute(
            b.getStringAttr(getAllocSharedMemoryAMarker()), b.getUnitAttr())}),
        /*filterResultType=*/TypeAttr(),
        /*filterOperandTYpes=*/ArrayAttr());
    b.create<transform::MemRefMultiBufferOp>(
        anyType, memrefAllocMatrixLHS, pipelineStage, /* skip_analysis */ true);

    auto memrefAllocMatrixRHS = b.create<transform::MatchOp>(
        memrefAllocType, pdlV,
        b.getStrArrayAttr({memref::AllocOp::getOperationName()}),
        /*matchInterfaceEnum=*/transform::MatchInterfaceEnumAttr(),
        /*opAttrs=*/
        b.getDictionaryAttr({NamedAttribute(
            b.getStringAttr(getAllocSharedMemoryBMarker()), b.getUnitAttr())}),
        /*filterResultType=*/TypeAttr(),
        /*filterOperandTYpes=*/ArrayAttr());
    b.create<transform::MemRefMultiBufferOp>(
        anyType, memrefAllocMatrixRHS, pipelineStage, /* skip_analysis */ true);

    // fold memref alias for subview of multi-buffers
    b.create<transform::ApplyPatternsOp>(pdlV, [](OpBuilder &b, Location loc) {
      b.create<transform::ApplyFoldMemrefAliasOpsPatternsOp>(loc);
    });

    // match scf::for op
    auto scfForOpType = transform::OperationType::get(
        b.getContext(), scf::ForOp::getOperationName());
    auto scfForOp = b.create<transform::MatchOp>(
        scfForOpType, pdlV, scf::ForOp::getOperationName());
    b.create<transform::PipelineSharedMemoryCopiesOp>(anyType, scfForOp,
                                                      pipelineStage);
  };
  pm.addPass(createGenericTransformInsertionPass(config));
}
} // namespace

void mlir::createGPUPipeliningTransform(OpPassManager &pm,
                                        const GPUGemmGeneralOptions &options) {
  invokeOpPassPipelineBuilder(createGPUPipeliningTransformImpl, pm,
                              options.funcAnchor, options.annotatePrefix);
}