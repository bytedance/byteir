//===- HloToLhlo.cpp ------------------------------------------*--- C++ -*-===//
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
// Some code from hlo_legalize_to_lhlo.cc in TensorFlow
// Original license:
//
// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
//
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
//===----------------------------------------------------------------------===//

#include "byteir/Conversion/HloToLHlo/HloToLHlo.h"
#include "lhlo/IR/lhlo_ops.h"
#include "lhlo/transforms/map_hlo_to_lhlo_op.h"
#include "mhlo/IR/hlo_ops.h"
#include "mhlo/transforms/rewriters.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Shape/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "../PassDetail.h"

using namespace llvm;
using namespace mlir;
using namespace mlir::arith;
using namespace mlir::lmhlo;
using namespace mlir::memref;
using namespace mlir::mhlo;
using namespace mlir::shape;

namespace mlir {
namespace mhlo {
#define MAP_HLO_TO_LHLO(OpName)                                                \
  template <> struct HloToLhloOpImpl<mhlo::OpName> {                           \
    using Type = lmhlo::OpName;                                                \
  }

MAP_HLO_TO_LHLO(MapOp);
MAP_HLO_TO_LHLO(ReverseOp);
MAP_HLO_TO_LHLO(ScatterOp);
MAP_HLO_TO_LHLO(SelectAndScatterOp);
MAP_HLO_TO_LHLO(SortOp);
#undef MAP_HLO_TO_LHLO

} // namespace mhlo
} // namespace mlir

namespace {
template <typename T> using BaseOpConversion = OpConversionPattern<T>;

/// LWC: this one modified the upstream one.
/// InsertAlloc(Location loc, OpResult result, ConversionPatternRewriter*
/// rewriter) Simply remap result_type == result.getType(), def_op ==
/// result.getDefiningOp()
Value InsertAlloc(Location loc, Type result_type, Operation *def_op,
                  ConversionPatternRewriter *rewriter) {
  auto result_tensor_type = result_type.dyn_cast<RankedTensorType>();
  if (!result_tensor_type || !result_tensor_type.hasStaticShape()) {
    def_op->emitOpError()
        << "tensor to buffer conversion expects statically shaped results";
  }
  auto memref_type = MemRefType::get(result_tensor_type.getShape(),
                                     result_tensor_type.getElementType());
  OpBuilder::InsertionGuard guard(*rewriter);
  rewriter->setInsertionPoint(def_op);
  auto alloc = rewriter->create<memref::AllocOp>(loc, memref_type);
  return alloc;
}

/// LWC: this one modified the upstream one.
/// InsertDynamicAlloc(Location loc, Value result, Value shape_operand,
/// ConversionPatternRewriter* rewriter) Simply remap result_type ==
/// result.getType(), def_op == result.getDefiningOp()
Value InsertDynamicAlloc(Location loc, Type result_type, Operation *def_op,
                         Value shape_operand,
                         ConversionPatternRewriter *rewriter) {
  auto result_tensor_type = result_type.dyn_cast<RankedTensorType>();
  if (!result_tensor_type) {
    def_op->emitOpError()
        << "tensor to buffer conversion expects ranked results";
  }
  auto memref_type = MemRefType::get(result_tensor_type.getShape(),
                                     result_tensor_type.getElementType());

  // Extract the required element out of the vector.
  SmallVector<Value, 4> dynamic_operands;
  for (auto shape_element : llvm::enumerate(result_tensor_type.getShape())) {
    if (shape_element.value() != ShapedType::kDynamic)
      continue;
    Value index = rewriter->create<ConstantIndexOp>(loc, shape_element.index());
    Value alloc_operand =
        rewriter->create<tensor::ExtractOp>(loc, shape_operand, index);
    if (!alloc_operand.getType().isIndex()) {
      alloc_operand = rewriter->create<arith::IndexCastOp>(
          loc, rewriter->getIndexType(), alloc_operand);
    }
    dynamic_operands.push_back(alloc_operand);
  }

  return rewriter->create<memref::AllocOp>(loc, memref_type, dynamic_operands);
}

/// Converts the results of the operation `op` to memref types and append them
/// to the `results` vector.
LogicalResult ConvertResults(Operation *op, SmallVectorImpl<Value> &results,
                             ConversionPatternRewriter &rewriter) {
  size_t num_operands = results.size();
  SmallVector<Value, 2> tensor_operands;

  auto convert_result = [&](Type result_ty, size_t index) {
    RankedTensorType resultType = result_ty.dyn_cast<RankedTensorType>();

    if (!resultType)
      return failure();

    if (resultType.hasStaticShape()) {
      results.push_back(InsertAlloc(op->getLoc(), resultType, op, &rewriter));
      return success();
    }
    auto shape_type_op = dyn_cast<InferShapedTypeOpInterface>(op);
    if (!shape_type_op)
      return failure();

    if (tensor_operands.empty()) {
      for (auto operand : ArrayRef<Value>(results).take_front(num_operands)) {
        auto operand_type = operand.getType().dyn_cast<MemRefType>();
        if (!operand_type)
          return failure();
        tensor_operands.push_back(rewriter.create<bufferization::ToTensorOp>(
            op->getLoc(),
            RankedTensorType::get(operand_type.getShape(),
                                  operand_type.getElementType()),
            operand));
      }
    }

    SmallVector<Value, 1> results_shape;
    auto status = shape_type_op.reifyReturnTypeShapes(rewriter, tensor_operands,
                                                      results_shape);
    if (failed(status))
      return failure();
    results.push_back(InsertDynamicAlloc(op->getLoc(), resultType, op,
                                         results_shape[index], &rewriter));
    return success();
  };

  if (op->getNumResults() == 1 && op->getResult(0).getType().isa<TupleType>()) {
    SmallVector<Type, 4> fattenedTypes;
    op->getResult(0).getType().cast<TupleType>().getFlattenedTypes(
        fattenedTypes);
    for (auto result : llvm::enumerate(fattenedTypes)) {
      if (failed(convert_result(result.value(), result.index())))
        return failure();
    }
  } else {
    for (auto result : llvm::enumerate(op->getResults())) {
      if (failed(convert_result(result.value().getType(), result.index())))
        return failure();
    }
  }
  return success();
}

template <typename HloOpTy>
class HloToLhloOpConverterLocal : public BaseOpConversion<HloOpTy> {
public:
  using BaseOpConversion<HloOpTy>::BaseOpConversion;
  LogicalResult
  matchAndRewrite(HloOpTy hloOp, typename HloOpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Operation *op = hloOp.getOperation();
    ValueRange operands = adaptor.getOperands();
    SmallVector<Value, 4> buffer_args(operands.begin(), operands.end());
    if (failed(ConvertResults(op, buffer_args, rewriter)))
      return failure();
    rewriter.create<mhlo::HloToLhloOp<HloOpTy>>(op->getLoc(), std::nullopt,
                                                buffer_args, op->getAttrs());
    rewriter.replaceOp(op,
                       llvm::ArrayRef(buffer_args).drop_front(operands.size()));
    return success();
  }
};

template <typename HloOpTy>
class HloWithTupleToLhloOpConverter : public BaseOpConversion<HloOpTy> {
public:
  using BaseOpConversion<HloOpTy>::BaseOpConversion;
  LogicalResult
  matchAndRewrite(HloOpTy hloOp, typename HloOpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Operation *op = hloOp.getOperation();
    ValueRange operands = adaptor.getOperands();

    // check all user are get_tuple_element
    // Currently, we only support users are get_tuple_element
    SmallVector<Operation *, 4> allUsers;
    for (auto *user : op->getUsers()) {
      allUsers.push_back(user);
      if (!isa<mhlo::GetTupleElementOp>(user)) {
        return failure();
      }
    }

    auto inputNum = op->getNumOperands();
    SmallVector<Value, 4> buffer_args(operands.begin(), operands.end());

    if (failed(ConvertResults(op, buffer_args, rewriter))) {
      return failure();
    }

    rewriter.create<mhlo::HloToLhloOp<HloOpTy>>(op->getLoc(), std::nullopt,
                                                buffer_args, op->getAttrs());

    // rewrite all tuple user
    for (auto *user : allUsers) {
      auto getElementOp = cast<mhlo::GetTupleElementOp>(user);
      unsigned index = inputNum + getElementOp.getIndex();
      rewriter.replaceOp(getElementOp, {buffer_args[index]});
    }

    // erase op
    rewriter.eraseOp(op);

    return success();
  }
};

struct HloToLhloReduceWindowOpConverter
    : public BaseOpConversion<mhlo::ReduceWindowOp> {
public:
  using BaseOpConversion<mhlo::ReduceWindowOp>::BaseOpConversion;

  LogicalResult
  matchAndRewrite(mhlo::ReduceWindowOp op, mhlo::ReduceWindowOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    ValueRange operands = adaptor.getOperands();
    auto loc = op.getLoc();
    if (!llvm::hasSingleElement(op.getBody())) {
      return op.emitOpError()
             << "tensor to buffer conversion expects a single block "
                "in the region containing the operation";
    }
    SmallVector<Value, 4> buffer_args(operands.begin(), operands.end());
    if (failed(ConvertResults(op, buffer_args, rewriter)))
      return failure();
    auto new_op = rewriter.create<lmhlo::ReduceWindowOp>(
        loc, std::nullopt, buffer_args, op->getAttrs());

    // Copy over the operations inside the region.
    rewriter.inlineRegionBefore(op.getBody(), new_op.getBody(),
                                new_op.getBody().end());

    // Convert the region signature to memref and add extra result.
    auto &entry_block = new_op.getBody().front();
    TypeConverter::SignatureConversion sig_conversion(operands.size());
    for (auto arg : entry_block.getArguments()) {
      auto old_type = arg.getType().cast<TensorType>();
      auto new_type =
          MemRefType::get(old_type.getShape(), old_type.getElementType());
      sig_conversion.addInputs(arg.getArgNumber(), new_type);
    }
    auto return_op = cast<mhlo::ReturnOp>(entry_block.getTerminator());
    if (auto tuple_ty =
            return_op.getResults().front().getType().dyn_cast<TupleType>()) {
      auto tuple_op = return_op.getODSOperands(0).front().getDefiningOp();
      return_op.getOperation()->dropAllReferences();
      rewriter.eraseOp(tuple_op);
      return_op.getOperation()->setOperands(tuple_op->getOperands());
      for (auto ty : tuple_ty) {
        auto tensor_ty = ty.cast<TensorType>();
        sig_conversion.addInputs(
            MemRefType::get(tensor_ty.getShape(), tensor_ty.getElementType()));
      }
    } else {
      auto result_type =
          return_op.getResults().front().getType().cast<TensorType>();
      sig_conversion.addInputs({MemRefType::get(result_type.getShape(),
                                                result_type.getElementType())});
    }
    rewriter.applySignatureConversion(&new_op.getBody(), sig_conversion);

    rewriter.replaceOp(op, ArrayRef<Value>(buffer_args).slice(operands.size()));

    return success();
  }
};

struct HloToLhloCustomCallOpConverter
    : public BaseOpConversion<mhlo::CustomCallOp> {
public:
  using BaseOpConversion<mhlo::CustomCallOp>::BaseOpConversion;

  LogicalResult
  matchAndRewrite(mhlo::CustomCallOp hloOp, mhlo::CustomCallOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Operation *op = hloOp.getOperation();
    ValueRange operands = adaptor.getOperands();

    // check all user are get_tuple_element
    // Currently, we only support users are get_tuple_element
    SmallVector<Operation *, 4> allUsers;
    for (auto *user : op->getUsers()) {
      allUsers.push_back(user);
      if (!isa<mhlo::GetTupleElementOp>(user)) {
        return failure();
      }
    }

    auto inputNum = op->getNumOperands();
    SmallVector<Value, 2> buffer_args(operands.begin(), operands.end());
    if (failed(ConvertResults(op, buffer_args, rewriter)))
      return failure();

    auto lhloOp = rewriter.create<lmhlo::CustomCallOp>(
        op->getLoc(), std::nullopt, buffer_args, op->getAttrs());
    // Setup AttrSizedOperandSegments attribute to indicate number of operands
    // for args and outputs.
    const int32_t segments[2] = {
        static_cast<int32_t>(operands.size()),
        static_cast<int32_t>(buffer_args.size() - operands.size())};
    lhloOp->setAttr(lhloOp.getOperandSegmentSizeAttr(),
                    rewriter.getDenseI32ArrayAttr(segments));

    // rewrite all tuple user
    for (auto *user : allUsers) {
      auto getElementOp = cast<mhlo::GetTupleElementOp>(user);
      unsigned index = inputNum + getElementOp.getIndex();
      rewriter.replaceOp(getElementOp, {buffer_args[index]});
    }

    // erase op
    rewriter.eraseOp(op);

    return success();
  }
};

template <typename HloOpTy>
class HloToLhloOpWithHloRegionsConverter : public BaseOpConversion<HloOpTy> {
public:
  using BaseOpConversion<HloOpTy>::BaseOpConversion;
  LogicalResult
  matchAndRewrite(HloOpTy hloOp, typename HloOpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Operation *op = hloOp.getOperation();
    ValueRange operands = adaptor.getOperands();
    SmallVector<Value, 4> buffer_args(operands.begin(), operands.end());
    if (failed(ConvertResults(op, buffer_args, rewriter)))
      return failure();

    auto new_op = rewriter.create<mhlo::HloToLhloOp<HloOpTy>>(
        op->getLoc(), std::nullopt, buffer_args, op->getAttrs());

    // Copy over the operations inside regions.
    for (unsigned i = 0, num_region = op->getNumRegions(); i < num_region;
         ++i) {
      rewriter.inlineRegionBefore(op->getRegion(i), new_op->getRegion(i),
                                  new_op->getRegion(i).end());
    }

    rewriter.replaceOp(op,
                       llvm::ArrayRef(buffer_args).drop_front(operands.size()));

    return success();
  }
};

LogicalResult rewriteToTensorOp(bufferization::ToTensorOp op,
                                PatternRewriter &rewriter) {
  if (op.getRestrict())
    return failure();

  op.setRestrict(true);
  return success();
}

struct ConvertHloToLHloPass
    : public ConvertHloToLHloBase<ConvertHloToLHloPass> {
public:
  ConvertHloToLHloPass() = default;
  void runOnOperation() override {
    auto &context = getContext();
    RewritePatternSet patterns(&context);
    ConversionTarget target(context);

    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<bufferization::BufferizationDialect>();
    target.addLegalDialect<cf::ControlFlowDialect>();
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalDialect<lmhlo::LmhloDialect>();
    target.addLegalDialect<memref::MemRefDialect>();
    target.addLegalDialect<shape::ShapeDialect>();
    target.addLegalDialect<tensor::TensorDialect>();

    // non-support op lowering here
    // They are typically mhlo op without lmhlo counterpart
    // Those ops should be hanndled in TrivialFusion by wrapping with calls
    target.addLegalOp<mhlo::RngBitGeneratorOp, mhlo::RngOp>();

    // LWC: lmhlo::ScatterOp's body allow mhlo
    target.addDynamicallyLegalDialect<mhlo::MhloDialect>([&](Operation *op) {
      return isa_and_nonnull<mhlo::MapOp, lmhlo::MapOp, mhlo::ScatterOp,
                             lmhlo::ScatterOp, mhlo::SelectAndScatterOp,
                             lmhlo::SelectAndScatterOp, mhlo::SortOp,
                             lmhlo::SortOp>(op->getParentOp());
    });

    // Declare tensor_store illegal. tensor_load may be used to reify output
    // shape computation during dialect conversion and will be handled later.
    target.addIllegalOp<mlir::memref::TensorStoreOp>();
    // bufferization.to_memref is illegal if it has uses.
    // TODO(b/175670649) Make bufferization.to_memref illegal.
    target.addDynamicallyLegalOp<bufferization::ToMemrefOp>(
        [](auto op) { return op->use_empty(); });

    bufferization::BufferizeTypeConverter converter;
    auto isMemRefType = [](Type type) { return type.isa<BaseMemRefType>(); };
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return converter.isSignatureLegal(op.getFunctionType()) &&
             converter.isLegal(&op.getBody());
    });
    target.addDynamicallyLegalOp<func::CallOp>([&](func::CallOp op) {
      return std::all_of(op.operand_type_begin(), op.operand_type_end(),
                         isMemRefType) &&
             std::all_of(op.result_type_begin(), op.result_type_end(),
                         isMemRefType);
    });
    target.addDynamicallyLegalOp<func::ReturnOp>([&](func::ReturnOp op) {
      return std::all_of(op.operand_type_begin(), op.operand_type_end(),
                         isMemRefType);
    });

    populateHloToLhloConversionPattern(&context, &converter, &patterns);

    populateHLOToLHLOConversionPatternExtension(&context, &converter,
                                                &patterns);

    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns,
                                                                   converter);
    populateCallOpTypeConversionPattern(patterns, converter);
    populateBranchOpInterfaceTypeConversionPattern(patterns, converter);
    populateReturnOpTypeConversionPattern(patterns, converter);
    populateEliminateBufferizeMaterializationsPatterns(converter, patterns);

    // LWC: Comment out this in our pass due to the scope
    // it has no impact
    // Uncommet it after push back to upstream.
    // patterns.insert<HloToLhloTensorStoreOpLegacyConverter>(&context);

    FrozenRewritePatternSet frozenPatterns(std::move(patterns));
    if (failed(
            applyPartialConversion(getOperation(), target, frozenPatterns))) {
      signalPassFailure();
    }

    RewritePatternSet toTensorPatterns(&context);
    toTensorPatterns.add(rewriteToTensorOp);
    FrozenRewritePatternSet toTensorFrozenPatterns(std::move(toTensorPatterns));
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            toTensorFrozenPatterns))) {
      signalPassFailure();
    }
  }
};
} // namespace

// Collection of rewrite patterns for lowering of HLO to LHLO dialect.
void mlir::populateHLOToLHLOConversionPatternExtension(
    MLIRContext *context, bufferization::BufferizeTypeConverter *converter,
    RewritePatternSet *patterns) {

  patterns->insert<HloToLhloOpConverterLocal<mhlo::ReverseOp>,
                   HloToLhloOpConverterLocal<mhlo::ClampOp>,
                   HloToLhloOpWithHloRegionsConverter<mhlo::MapOp>,
                   HloToLhloOpWithHloRegionsConverter<mhlo::ScatterOp>,
                   HloToLhloOpWithHloRegionsConverter<mhlo::SelectAndScatterOp>,
                   HloToLhloOpWithHloRegionsConverter<mhlo::SortOp>,
                   HloToLhloReduceWindowOpConverter,
                   HloToLhloCustomCallOpConverter>(*converter, context);
}

std::unique_ptr<OperationPass<ModuleOp>> mlir::createConvertHloToLHloPass() {
  return std::make_unique<ConvertHloToLHloPass>();
}
