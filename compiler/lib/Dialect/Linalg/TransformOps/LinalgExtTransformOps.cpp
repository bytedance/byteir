//===- LinalgExtTransformOps.cpp - Implementation of Linalg transform ops -===//
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
// Some code comes from LinalgExtTransformOps.cpp in IREE project
// Original license:
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/Linalg/TransformOps/LinalgExtTransformOps.h"

#include "byteir/Dialect/Linalg/IR/LinalgExtOps.h"
#include "byteir/Dialect/Linalg/Transforms/Transforms.h"
#include "byteir/Utils/Hoist.h"
#include "byteir/Utils/Utils.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDL/IR/PDLTypes.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace mlir::linalg;
using namespace mlir::linalg_ext;
using namespace mlir::scf;
using namespace mlir::transform;

#define DEBUG_TYPE "linalg-ext-transforms"

namespace {
/// A simple pattern rewriter that implements no special logic.
class SimpleRewriter : public PatternRewriter {
public:
  SimpleRewriter(MLIRContext *context) : PatternRewriter(context) {}
};

} // namespace

//===----------------------------------------------------------------------===//
// AnnotateOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::AnnotateOp::apply(TransformResults &transformResults,
                             TransformState &state) {

  ArrayRef<Operation *> targets = state.getPayloadOps(getTarget());

  auto attrs = getOperation()->getAttrs();
  for (auto &target : targets) {
    addAttrs(target, attrs);
  }

  return DiagnosedSilenceableFailure::success();
}

ParseResult transform::AnnotateOp::parse(OpAsmParser &parser,
                                         OperationState &result) {
  OpAsmParser::UnresolvedOperand target;
  auto pdlOperationType = pdl::OperationType::get(parser.getContext());
  if (parser.parseOperand(target) ||
      parser.resolveOperand(target, pdlOperationType, result.operands) ||
      parser.parseOptionalAttrDict(result.attributes)) {
    return ParseResult::failure();
  }
  return success();
}

void AnnotateOp::print(OpAsmPrinter &p) {
  p << ' ' << getTarget();
  p.printOptionalAttrDict((*this)->getAttrs());
}

void transform::AnnotateOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  consumesHandle(getTarget(), effects);
  modifiesPayload(effects);
}

//===----------------------------------------------------------------------===//
// FuseExtOp
//===----------------------------------------------------------------------===//

namespace {

/// Apply a tiling transformation to all payload ops and store both the
/// tiled operation as well as the created tile loops.
static LogicalResult applyTilingToAll(
    Operation *transformOp, ArrayRef<Operation *> payloadOps, unsigned numLoops,
    transform::TransformResults &transformResults,
    function_ref<FailureOr<scf::SCFTileAndFuseResult>(TilingInterface)>
        applyFn) {
  SmallVector<Operation *> tiledLinalgOps;
  SmallVector<SmallVector<Operation *>> loopOps(numLoops);
  for (unsigned int i = 0; i < numLoops; ++i)
    loopOps[i].reserve(payloadOps.size());

  auto ctx = transformOp->getContext();
  RewritePatternSet patterns(ctx);
  ForOp::getCanonicalizationPatterns(patterns, ctx);
  FrozenRewritePatternSet frozenPatterns(std::move(patterns));

  for (Operation *target : payloadOps) {
    auto funcOp = target->getParentOfType<func::FuncOp>();

    // simplify tensor::DimOp
    simplifyTensorDimOpUsedInLinalgWithinOp(*funcOp.getOperation());
    auto tilingInterfaceOp = dyn_cast<TilingInterface>(target);
    if (!tilingInterfaceOp)
      return transformOp->emitError("only TilingInterface ops are supported");

    SimpleRewriter rewriter(target->getContext());
    rewriter.setInsertionPoint(target);
    FailureOr<scf::SCFTileAndFuseResult> tiledResults =
        applyFn(tilingInterfaceOp);
    if (failed(tiledResults))
      return failure();

    auto domInfo = DominanceInfo(target->getParentOfType<func::FuncOp>());
    auto postDomInfo =
        PostDominanceInfo(target->getParentOfType<func::FuncOp>());

    // Perform the replacement of tiled and fused values.
    llvm::SmallSetVector<Operation *, 8> opsToReplace;
    opsToReplace.insert(target);
    SmallVector<Operation *> unfusedUser;
    for (auto fusedOp : tiledResults->fusedProducers) {
      opsToReplace.insert(fusedOp);
    }

    for (Operation *toReplace : opsToReplace) {
      SmallVector<Value> replacements;
      replacements.reserve(toReplace->getNumResults());
      for (OpResult res : toReplace->getResults()) {
        auto it = tiledResults->replacements.find(res);
        if (it == tiledResults->replacements.end()) {
          replacements.push_back(res); // unchanged
        } else {
          replacements.push_back(it->getSecond()); // replaced
        }

        // collect unfusedUser
        for (auto user : res.getUsers()) {
          if (opsToReplace.contains(user))
            continue;
          unfusedUser.push_back(user);
        }
      }

      llvm::SmallSetVector<Operation *, 8> replacementDefs;
      for (auto replacement : replacements) {
        if (auto defOp = replacement.getDefiningOp()) {
          if (!replacementDefs.contains(defOp)) {
            replacementDefs.insert(defOp);
          }
        }
      }

      for (auto user : unfusedUser) {
        hoistDownDescendantUsers(user, postDomInfo);
      }

      auto allowReplacement = [&](OpOperand &use) {
        for (auto replaceVal : replacements) {
          if (auto def = replaceVal.getDefiningOp()) {
            if (!domInfo.properlyDominates(def, use.getOwner())) {
              return false;
            }
          }
        }
        return true;
      };

      rewriter.replaceOpWithIf(toReplace, replacements, allowReplacement);

      // simplify tensor::DimOp
      simplifyTensorDimOpUsedInLinalgWithinOp(*funcOp.getOperation());
    }

    // Report back the relevant handles to the transform op.
    tiledLinalgOps.push_back(tiledResults->tiledAndFusedOps.front());
    assert(tiledResults->loops.size() == numLoops &&
           "Mismatched number of loops, tile and fuse transform should have "
           "failed");
    for (unsigned int i = 0; i < numLoops; ++i)
      loopOps[i].push_back(tiledResults->loops[i]);
  } // for (Operation *target : payloadOps)

  transformResults.set(transformOp->getOpResult(0), tiledLinalgOps);
  for (unsigned int i = 0; i < numLoops; ++i)
    transformResults.set(transformOp->getOpResult(i + 1), loopOps[i]);

  return success();
}

/// Parse a tiling-like operation that returns the tiled op as well as the
/// created tile loops. The function counts the non-zero tile sizes to compute
/// the number of results.
static ParseResult parseTileLikeOp(OpAsmParser &parser, OperationState &result,
                                   StringRef sizesAttrName) {
  OpAsmParser::UnresolvedOperand targetOperand;
  SMLoc opLoc = parser.getCurrentLocation();
  if (parser.parseOperand(targetOperand) ||
      parser.parseOptionalAttrDict(result.attributes))
    return failure();
  Attribute sizesAttr = result.attributes.get(sizesAttrName);
  if (!sizesAttr)
    return parser.emitError(opLoc)
           << "expected '" << sizesAttrName << "' attribute";
  auto sizesArrayAttr = sizesAttr.dyn_cast<ArrayAttr>();
  if (!sizesArrayAttr)
    return parser.emitError(opLoc)
           << "'" << sizesAttrName << "' attribute must be an array";
  Type pdlOpType = parser.getBuilder().getType<pdl::OperationType>();
  size_t numExpectedLoops =
      sizesArrayAttr.size() -
      llvm::count(extractFromI64ArrayAttr(sizesArrayAttr), 0);
  result.addTypes(SmallVector<Type>(numExpectedLoops + 1, pdlOpType));
  if (parser.resolveOperand(targetOperand, pdlOpType, result.operands))
    return failure();
  return success();
}

} // namespace

DiagnosedSilenceableFailure
transform::FuseExtOp::apply(mlir::transform::TransformResults &transformResults,
                            mlir::transform::TransformState &state) {
  SmallVector<int64_t> tileSizes = extractFromI64ArrayAttr(getTileSizes());
  SmallVector<int64_t> tileInterchange =
      extractFromI64ArrayAttr(getTileInterchange());

  scf::SCFTilingOptions tilingOptions;
  tilingOptions.interchangeVector = tileInterchange;
  tilingOptions = tilingOptions.setTileSizes(tileSizes);
  scf::SCFTileAndFuseOptions tileAndFuseOptions;
  tileAndFuseOptions.tilingOptions = tilingOptions;
  LogicalResult result = applyTilingToAll(
      getOperation(), state.getPayloadOps(getTarget()),
      tileSizes.size() - llvm::count(tileSizes, 0), transformResults,
      [&](TilingInterface tilingInterfaceOp)
          -> FailureOr<scf::SCFTileAndFuseResult> {
        SimpleRewriter rewriter(getContext());
        return tileConsumerAndFuseProducerUsingSCFForOpExt(
            rewriter, tilingInterfaceOp, tileAndFuseOptions,
            /*simplifyLoopIter*/ true);
      });

  return failed(result) ? DiagnosedSilenceableFailure::definiteFailure()
                        : DiagnosedSilenceableFailure::success();
}

ParseResult transform::FuseExtOp::parse(OpAsmParser &parser,
                                        OperationState &result) {
  return parseTileLikeOp(
      parser, result,
      transform::FuseExtOp::getTileSizesAttrName(result.name).getValue());
}

void transform::FuseExtOp::print(OpAsmPrinter &p) {
  p << ' ';
  p << getTarget();
  p.printOptionalAttrDict((*this)->getAttrs());
}

LogicalResult transform::FuseExtOp::verify() {
  SmallVector<int64_t> permutation =
      extractFromI64ArrayAttr(getTileInterchange());
  auto sequence = llvm::to_vector(llvm::seq<int64_t>(0, permutation.size()));
  if (!std::is_permutation(sequence.begin(), sequence.end(),
                           permutation.begin(), permutation.end())) {
    return emitOpError() << "expects interchange to be a permutation, found "
                         << getTileInterchange();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// TileLoopHintOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::TileLoopHintOp::apply(TransformResults &transformResults,
                                 TransformState &state) {
  ArrayRef<Operation *> targets = state.getPayloadOps(getTarget());
  for (auto op : targets) {
    if (!isa<TilingInterface>(op)) {
      DiagnosedSilenceableFailure diag =
          emitSilenceableError() << "only TilingInterface ops are supported";
      diag.attachNote(op->getLoc()) << "target op";
      return diag;
    }

    SmallVector<scf::ForOp> loops;
    Operation *cur = op;

    while (cur) {
      if (auto forOp = cur->getParentOfType<scf::ForOp>()) {
        loops.push_back(forOp);
        cur = forOp;
      } else {
        break;
      }
    }
    loops = llvm::to_vector(llvm::reverse(loops));

    if (loops.empty()) {
      DiagnosedSilenceableFailure diag = emitSilenceableError()
                                         << "no loops at";
      diag.attachNote(op->getLoc()) << "target op";
      return diag;
    }

    labelTileLoopType(op, loops);
  }
  return DiagnosedSilenceableFailure::success();
}

ParseResult transform::TileLoopHintOp::parse(OpAsmParser &parser,
                                             OperationState &result) {
  OpAsmParser::UnresolvedOperand target;
  auto pdlOperationType = pdl::OperationType::get(parser.getContext());
  if (parser.parseOperand(target) ||
      parser.resolveOperand(target, pdlOperationType, result.operands) ||
      parser.parseOptionalAttrDict(result.attributes))
    return ParseResult::failure();
  return success();
}

void TileLoopHintOp::print(OpAsmPrinter &p) {
  p << ' ' << getTarget();
  p.printOptionalAttrDict((*this)->getAttrs());
}

void transform::TileLoopHintOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getTarget(), effects);
  onlyReadsPayload(effects);
}

//===----------------------------------------------------------------------===//
// TileExtOp
//===----------------------------------------------------------------------===//

namespace {

// We want to parse `DenseI64ArrayAttr` using the short form without the
// `array` prefix to be consistent in the IR with `parseDynamicIndexList`.
static ParseResult parseOptionalInterchange(OpAsmParser &parser,
                                            OperationState &result) {
  if (succeeded(parser.parseOptionalLBrace())) {
    if (failed(parser.parseKeyword("interchange")))
      return parser.emitError(parser.getNameLoc()) << "expect `interchange`";
    if (failed(parser.parseEqual()))
      return parser.emitError(parser.getNameLoc()) << "expect `=`";
    result.addAttribute("interchange",
                        DenseI64ArrayAttr::parse(parser, Type{}));
    if (failed(parser.parseRBrace()))
      return parser.emitError(parser.getNameLoc()) << "expect `}`";
  }
  return success();
}

static void printOptionalInterchange(OpAsmPrinter &p,
                                     ArrayRef<int64_t> interchangeVals) {
  if (!interchangeVals.empty()) {
    p << " {interchange = [";
    llvm::interleaveComma(interchangeVals, p,
                          [&](int64_t integer) { p << integer; });
    p << "]}";
  }
}

} // namespace

DiagnosedSilenceableFailure
transform::TileExtOp::apply(TransformResults &transformResults,
                            TransformState &state) {
  ArrayRef<int64_t> tileSizes = getStaticSizes();

  ArrayRef<Operation *> targets = state.getPayloadOps(getTarget());
  SmallVector<ArrayRef<Operation *>> dynamicSizeProducers;
  dynamicSizeProducers.reserve(getDynamicSizes().size());
  for (Value dynamicSizeProducerHandle : getDynamicSizes()) {
    dynamicSizeProducers.push_back(
        state.getPayloadOps(dynamicSizeProducerHandle));

    if (dynamicSizeProducers.back().size() != targets.size()) {
      DiagnosedSilenceableFailure diag =
          emitSilenceableError()
          << "expected as many dynamic size-producing operations ("
          << dynamicSizeProducers.back().size() << ") as target ops ("
          << targets.size() << ")";
      diag.attachNote(dynamicSizeProducerHandle.getLoc()) << "for this handle";
      return diag;
    }

    for (Operation *op : dynamicSizeProducers.back()) {
      if (op->getNumResults() == 1 &&
          op->getResult(0).getType().isa<IndexType>())
        continue;
      DiagnosedSilenceableFailure diag =
          emitSilenceableError() << "expected sizes to be produced by ops "
                                    "with a single index-type result";
      diag.attachNote(op->getLoc()) << "size producer op";
      diag.attachNote(dynamicSizeProducerHandle.getLoc()) << "for this handle";
      return diag;
    }
  }

  SmallVector<Operation *> tiled;
  SmallVector<SmallVector<Operation *, 4>, 4> loops;
  loops.resize(getLoops().size());
  for (auto &en : llvm::enumerate(targets)) {
    if (!isa<TilingInterface>(en.value())) {
      DiagnosedSilenceableFailure diag = emitSilenceableError()
                                         << "only linalg ops are supported";
      diag.attachNote(en.value()->getLoc()) << "target op";
      return diag;
    }

    scf::SCFTilingOptions tilingOptions;
    unsigned index = en.index();
    if (!tileSizes.empty()) {
      tilingOptions.setTileSizeComputationFunction(
          [&, index](OpBuilder &b, Operation *) {
            SmallVector<Value, 4> sizes;
            sizes.reserve(tileSizes.size());
            unsigned dynamicIdx = 0;
            for (OpFoldResult ofr : getMixedSizes()) {
              if (auto attr = ofr.dyn_cast<Attribute>()) {
                sizes.push_back(b.create<arith::ConstantIndexOp>(
                    getLoc(), attr.cast<IntegerAttr>().getInt()));
              } else {
                sizes.push_back(
                    dynamicSizeProducers[dynamicIdx++][index]->getResult(0));
              }
            }
            return sizes;
          });
    }

    tilingOptions.setInterchange(getInterchange());
    SimpleRewriter rewriter(en.value()->getContext());

    FailureOr<scf::SCFTilingResult> maybeTilingResult = tileUsingSCFForOp(
        rewriter, cast<TilingInterface>(en.value()), tilingOptions);

    if (failed(maybeTilingResult))
      return DiagnosedSilenceableFailure::definiteFailure();

    if (failed(isValidTiling(maybeTilingResult->tiledOps.back()))) {
      DiagnosedSilenceableFailure diag = emitSilenceableError()
                                         << "unsupported tiling dim at ";
      diag.attachNote(en.value()->getLoc()) << "target op";
      return diag;
    }

    rewriter.replaceOp(en.value(),
                       maybeTilingResult->loops.front()->getResults());

    tiled.push_back(maybeTilingResult->tiledOps.back());
    for (const auto &en2 : llvm::enumerate(maybeTilingResult->loops))
      loops[en2.index()].push_back(en2.value());
  }

  transformResults.set(getTiledLinalgOp().cast<OpResult>(), tiled);
  for (const auto &en : llvm::enumerate(loops))
    transformResults.set(getLoops()[en.index()].cast<OpResult>(), en.value());

  return DiagnosedSilenceableFailure::success();
}

SmallVector<OpFoldResult> transform::TileExtOp::getMixedSizes() {
  ValueRange dynamic = getDynamicSizes();
  ArrayRef<int64_t> tileSizes = getStaticSizes();
  SmallVector<OpFoldResult> results;
  results.reserve(tileSizes.size());
  unsigned dynamicPos = 0;
  Builder builder(getContext());
  for (int64_t size : tileSizes) {
    if (size == ShapedType::kDynamic) {
      results.push_back(dynamic[dynamicPos++]);
    } else {
      results.push_back(builder.getIndexAttr(size));
    }
  }
  return results;
}

ParseResult transform::TileExtOp::parse(OpAsmParser &parser,
                                        OperationState &result) {
  OpAsmParser::UnresolvedOperand target;
  SmallVector<OpAsmParser::UnresolvedOperand> dynamicSizes;
  DenseI64ArrayAttr staticSizes;
  auto pdlOperationType = pdl::OperationType::get(parser.getContext());
  if (parser.parseOperand(target) ||
      parser.resolveOperand(target, pdlOperationType, result.operands) ||
      parseDynamicIndexList(parser, dynamicSizes, staticSizes) ||
      parser.resolveOperands(dynamicSizes, pdlOperationType, result.operands))
    return ParseResult::failure();

  // Parse optional interchange.
  if (failed(parseOptionalInterchange(parser, result)))
    return ParseResult::failure();
  result.addAttribute(getStaticSizesAttrName(result.name), staticSizes);
  size_t numExpectedLoops =
      staticSizes.size() - llvm::count(staticSizes.asArrayRef(), 0);
  result.addTypes(SmallVector<Type>(numExpectedLoops + 1, pdlOperationType));
  return success();
}

void TileExtOp::print(OpAsmPrinter &p) {
  p << ' ' << getTarget();
  printDynamicIndexList(p, getOperation(), getDynamicSizes(), getStaticSizes());
  printOptionalInterchange(p, getInterchange());
}

void transform::TileExtOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  consumesHandle(getTarget(), effects);
  onlyReadsHandle(getDynamicSizes(), effects);
  producesHandle(getTiledLinalgOp(), effects);
  producesHandle(getLoops(), effects);
  modifiesPayload(effects);
}

//===----------------------------------------------------------------------===//
// Transform op registration
//===----------------------------------------------------------------------===//

namespace {
/// Registers new ops and declares PDL as dependent dialect since the
/// additional ops are using PDL types for operands and results.
class LinalgExtTransformDialectExtension
    : public transform::TransformDialectExtension<
          LinalgExtTransformDialectExtension> {
public:
  using Base::Base;

  void init() {
    // TODO remove unused ones
    declareDependentDialect<pdl::PDLDialect>();
    declareDependentDialect<LinalgDialect>();
    declareDependentDialect<LinalgExtDialect>();
    declareGeneratedDialect<AffineDialect>();
    declareGeneratedDialect<arith::ArithDialect>();
    declareGeneratedDialect<scf::SCFDialect>();
    declareGeneratedDialect<vector::VectorDialect>();
    declareGeneratedDialect<gpu::GPUDialect>();

    registerTransformOps<
#define GET_OP_LIST
#include "byteir/Dialect/Linalg/TransformOps/LinalgExtTransformOps.cpp.inc"
        >();
  }
};
} // namespace

#define GET_OP_CLASSES
#include "byteir/Dialect/Linalg/TransformOps/LinalgExtTransformOps.cpp.inc"

void mlir::linalg_ext::registerTransformDialectExtension(
    DialectRegistry &registry) {
  registry.addExtensions<LinalgExtTransformDialectExtension>();
}
