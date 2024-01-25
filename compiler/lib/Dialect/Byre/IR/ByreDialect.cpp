//===- ByreDialect.cpp - MLIR Dialect for Runtime implementation -------===//
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
//
// This file implements the Runtime-related dialect and its operations.
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/Byre/ByreDialect.h"

#include "byteir/Utils/Utils.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include <algorithm> // for std::any_of

using namespace mlir;
using namespace mlir::byre;

#include "byteir/Dialect/Byre/ByreEnums.cpp.inc"
#include "byteir/Dialect/Byre/ByreOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Common Utilities
//===----------------------------------------------------------------------===//

namespace {
/// This is a common class used for patterns of the form
/// ```
///    someop(memrefcast(%src)) -> someop(%src)
/// ```
/// It folds the source of the memref.cast into the root operation directly.
static LogicalResult foldMemRefCast(Operation *op) {
  bool folded = false;
  for (OpOperand &operand : op->getOpOperands()) {
    auto castOp = operand.get().getDefiningOp<memref::CastOp>();
    if (castOp && memref::CastOp::canFoldIntoConsumerOp(castOp)) {
      operand.set(castOp.getOperand());
      folded = true;
    }
  }
  return success(folded);
}

static LogicalResult verifyOpInEntryPointFunc(Operation *op) {
  auto func = op->getParentOfType<func::FuncOp>();
  if (!func->hasAttrOfType<UnitAttr>(
          ByreDialect::getEntryPointFunctionAttrName()) &&
      !func->hasAttrOfType<UnitAttr>(getAttrPlaceholderName(
          ByreDialect::getEntryPointFunctionAttrName()))) {
    return op->emitError("expected '")
           << ByreDialect::getEntryPointFunctionAttrName()
           << "' attribute to be attached to '"
           << func::FuncOp::getOperationName() << "' " << func.getName();
  }
  return success();
}

static bool validEntryFuncArgType(EntryFuncArgType argType) {
  return argType == EntryFuncArgType::Input ||
         argType == EntryFuncArgType::Output ||
         argType == EntryFuncArgType::Weight;
}
} // namespace

//===----------------------------------------------------------------------===//
// ByreDialect
//===----------------------------------------------------------------------===//

void ByreDialect::initialize() {
  addTypes<AsyncTokenType>();
  addOperations<
#define GET_OP_LIST
#include "byteir/Dialect/Byre/ByreOps.cpp.inc"
      >();
}

Type ByreDialect::parseType(DialectAsmParser &parser) const {
  // Parse the main keyword for the type.
  StringRef keyword;
  if (parser.parseKeyword(&keyword))
    return Type();
  MLIRContext *context = getContext();

  // Handle 'async token' types.
  if (keyword == "async.token")
    return AsyncTokenType::get(context);

  parser.emitError(parser.getNameLoc(), "unknown byre type: " + keyword);
  return Type();
}

void ByreDialect::printType(Type type, DialectAsmPrinter &os) const {
  TypeSwitch<Type>(type)
      .Case<AsyncTokenType>([&](Type) { os << "async.token"; })
      .Default([](Type) { llvm_unreachable("unexpected 'byre' type kind"); });
}

LogicalResult ByreDialect::verifyOperationAttribute(Operation *op,
                                                    NamedAttribute attr) {
  // ContainerModuleAttr only applied to ModuleOp
  if (attr.getValue().isa<UnitAttr>() &&
      attr.getName().getValue() == getContainerModuleAttrName()) {
    if (!isa<ModuleOp>(op)) {
      return op->emitError("expected '")
             << getContainerModuleAttrName()
             << "' attribute to be attached to '"
             << ModuleOp::getOperationName() << '\'';
    }

    // handle possible ModuleMemorySpaceAttr
    if (auto memSpace =
            op->getAttrOfType<ArrayAttr>(getModuleMemorySpaceAttrName())) {
      // if odd
      if (memSpace.size() & 1) {
        return op->emitError("expected '")
               << getModuleMemorySpaceAttrName() << "' has Even numbers";
      }

      bool isEven = true;
      for (auto it = memSpace.begin(); it != memSpace.end(); ++it) {
        if (isEven && !it->isa<IntegerAttr>()) {
          return op->emitError("expected '") << getModuleMemorySpaceAttrName()
                                             << "'has IntegerAttr in Even";
        }

        if (!isEven && !it->isa<StringAttr>()) {
          return op->emitError("expected '")
                 << getModuleMemorySpaceAttrName() << "'has StringAttr in Odd";
        }
        isEven = !isEven;
      }
    }
  }

  // ModuleMemorySpaceAttr only applied to ModuleOp with ContainerModuleAttr
  if (attr.getValue().isa<ArrayAttr>() &&
      attr.getName().getValue() == getModuleMemorySpaceAttrName()) {
    if (!op->hasAttrOfType<UnitAttr>(getContainerModuleAttrName())) {
      return op->emitError("expected '")
             << getModuleMemorySpaceAttrName()
             << "' attribute to be attached to '"
             << ModuleOp::getOperationName() << "' with '"
             << getContainerModuleAttrName() << '\'';
    }
  }

  // EntryPointFunctionAttr only applied to FuncOp,
  // which under ModuleOp with ContainerModuleAttrName
  if (attr.getValue().isa<UnitAttr>() &&
      attr.getName().getValue() == getEntryPointFunctionAttrName()) {
    if (!isa<func::FuncOp>(op)) {
      return op->emitError("expected '")
             << getEntryPointFunctionAttrName()
             << "' attribute to be attached to '"
             << func::FuncOp::getOperationName() << '\'';
    }
    auto funcOp = llvm::cast<func::FuncOp>(op);

    // FuncOp's parent must be ModuleOp with ContainerModuleAttr
    auto parentOp = op->getParentOp();
    if (parentOp == nullptr || !isa<ModuleOp>(parentOp)) {
      return op->emitError("expected '")
             << getEntryPointFunctionAttrName()
             << "' attribute to be attached to '"
             << func::FuncOp::getOperationName() << "' under '"
             << ModuleOp::getOperationName() << '\'';
    }

    // FIXME

    // This become optional
    if (!parentOp->hasAttrOfType<UnitAttr>(getContainerModuleAttrName())) {
      return op->emitError("expected '")
             << getEntryPointFunctionAttrName()
             << "' attribute to be attached to '"
             << func::FuncOp::getOperationName() << "' under '"
             << ModuleOp::getOperationName() << "' with '"
             << getContainerModuleAttrName() << '\'';
    }

    // check weights, inputs and outputs
    size_t numInputs = 0, numOutputs = 0, numWeights = 0;
    using ArgType = EntryFuncArgType;
    for (size_t idx = 0; idx < funcOp.getNumArguments(); ++idx) {
      // check argument type
      if (auto argTypeAttr = funcOp.getArgAttrOfType<EntryFuncArgTypeAttr>(
              idx, ByreDialect::getEntryPointFuncArgTypeAttrName())) {
        ArgType argType = argTypeAttr.getValue();
        if (!validEntryFuncArgType(argType)) {
          return op->emitError("invalid argtype '")
                 << stringifyEnum(argType) << "' attached to the argument of '"
                 << func::FuncOp::getOperationName() << "' under '"
                 << ModuleOp::getOperationName() << '\'';
        }
        if (bitEnumContainsAll(argType, ArgType::Input)) {
          numInputs++;
        }
        if (bitEnumContainsAll(argType, ArgType::Output)) {
          numOutputs++;
        }
        if (bitEnumContainsAll(argType, ArgType::Weight)) {
          numWeights++;
        }
      } else {
        return op->emitError("expected attribute '")
               << getEntryPointFuncArgTypeAttrName()
               << "' to be attached to the argument of '"
               << func::FuncOp::getOperationName() << "' under '"
               << ModuleOp::getOperationName() << '\'';
      }

      // check argument name
      if (auto argNameAttr = funcOp.getArgAttr(
              idx, ByreDialect::getEntryPointFuncArgNameAttrName())) {
        if (!argNameAttr.isa<StringAttr>()) {
          return op->emitError("expected StringAttr in '")
                 << ByreDialect::getEntryPointFuncArgNameAttrName() << '\'';
        }
      } else {
        return op->emitError("expected attribute '")
               << getEntryPointFuncArgNameAttrName()
               << "' to be attached to the argument of '"
               << func::FuncOp::getOperationName() << "' under '"
               << ModuleOp::getOperationName() << '\'';
      }
    }

    // FuncOp has no return
    if (funcOp.getNumResults() != 0) {
      return op->emitError("expected no return in ")
             << funcOp.getName() << '\'';
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// ComputeOp
//===----------------------------------------------------------------------===/

void ComputeOp::build(OpBuilder &builder, OperationState &result,
                      StringRef callee, ValueRange inputs, ValueRange outputs) {
  SmallVector<Attribute> memoryEffectAttrs;
  memoryEffectAttrs.append(
      inputs.size(), builder.getAttr<MemoryEffectAttr>(MemoryEffect::Read));
  memoryEffectAttrs.append(
      outputs.size(), builder.getAttr<MemoryEffectAttr>(MemoryEffect::Write));
  build(builder, result, TypeRange{}, callee,
        llvm::to_vector(llvm::concat<Value>(llvm::to_vector(inputs),
                                            llvm::to_vector(outputs))),
        builder.getArrayAttr(memoryEffectAttrs));
}

// verify ComputeOp
LogicalResult ComputeOp::verify() {
  if (verifyOpInEntryPointFunc(this->getOperation()).failed()) {
    return failure();
  }

  auto maybeMemoryEffects = this->getMemoryEffects();
  if (maybeMemoryEffects) {
    if (maybeMemoryEffects->size() != this->getNumOperands()) {
      return emitError("size of memory effects mismatch");
    }
    if (llvm::any_of(maybeMemoryEffects->getValue(), [](Attribute attr) {
          return !attr.isa<MemoryEffectAttr>();
        })) {
      return emitError("invalid memory effect attribute");
    }
  }
  return success();
}

FunctionType mlir::byre::ComputeOp::getType() {
  return FunctionType::get(getContext(), getOperandTypes(), {});
}

void ComputeOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  if (!this->getMemoryEffects()) {
    // if memory effects was not set, assume that all operands are readwrite
    for (auto &&i : this->getOperands()) {
      effects.emplace_back(MemoryEffects::Read::get(), i,
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Write::get(), i,
                           SideEffects::DefaultResource::get());
    }
  } else {
    auto memoryEffects = llvm::to_vector(
        this->getMemoryEffects()->getAsValueRange<MemoryEffectAttr>());
    for (auto &&pi : llvm::zip(this->getOperands(), memoryEffects)) {
      auto value = std::get<0>(pi);
      MemoryEffect effect = std::get<1>(pi);
      if (bitEnumContainsAll(effect, MemoryEffect::Read)) {
        effects.emplace_back(MemoryEffects::Read::get(), value,
                             SideEffects::DefaultResource::get());
      }
      if (bitEnumContainsAll(effect, MemoryEffect::Write)) {
        effects.emplace_back(MemoryEffects::Write::get(), value,
                             SideEffects::DefaultResource::get());
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// ComputeShapeOp
//===----------------------------------------------------------------------===/

LogicalResult ComputeShapeOp::verify() {
  return verifyOpInEntryPointFunc(this->getOperation());
}

std::string ComputeShapeOp::getCalleeName() { return "ComputeShapeOp"; }

//===----------------------------------------------------------------------===//
// CopyOp
//===----------------------------------------------------------------------===/

namespace {
/// Remove copy operations that copy data with the same input and output
struct EraseIdentityCopyOp : public OpRewritePattern<CopyOp> {
  using OpRewritePattern<CopyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CopyOp copyOp,
                                PatternRewriter &rewriter) const override {
    if (copyOp.getSource() == copyOp.getTarget()) {
      rewriter.eraseOp(copyOp);
      return success();
    }
    return failure();
  }
};
} // namespace

void CopyOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                         MLIRContext *context) {
  results.add<EraseIdentityCopyOp>(context);
}

LogicalResult CopyOp::fold(FoldAdaptor, SmallVectorImpl<OpFoldResult> &) {
  return foldMemRefCast(*this);
}

LogicalResult CopyOp::verify() {
  return verifyOpInEntryPointFunc(this->getOperation());
}

//===----------------------------------------------------------------------===//
// GroupCopyOp
//===----------------------------------------------------------------------===/

namespace {
/// Remove copy operations that copy data with the same input and output
struct EraseIdentityGroupCopyOp : public OpRewritePattern<GroupCopyOp> {
  using OpRewritePattern<GroupCopyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GroupCopyOp op,
                                PatternRewriter &rewriter) const override {
    auto refs = op.getOperands();
    auto array_size = refs.size() >> 1;
    SmallVector<Value> srcRefs, dstRefs;
    for (size_t i = 0; i < array_size; ++i) {
      Value src = refs[i];
      Value dst = refs[array_size + i];
      if (src != dst) {
        srcRefs.push_back(src);
        dstRefs.push_back(dst);
      }
    }
    if (srcRefs.size() == 0) {
      rewriter.eraseOp(op);
      return success();
    }
    if (srcRefs.size() == 1) {
      std::string groupCallee =
          op->getAttrOfType<::mlir::StringAttr>("callee").getValue().str();
      // naming convention: group copy callee ends with '_array'
      std::string newCallee = groupCallee.substr(0, groupCallee.size() - 6);
      auto new_op =
          rewriter.replaceOpWithNewOp<byre::CopyOp>(op, srcRefs[0], dstRefs[0]);
      new_op->setAttr("callee", StringAttr::get(getContext(), newCallee));
      return success();
    }
    if (srcRefs.size() != array_size) {
      rewriter.replaceOpWithNewOp<byre::GroupCopyOp>(
          op, ArrayRef<Value>(srcRefs), ArrayRef<Value>(dstRefs));
      return success();
    }
    return failure();
  }
};
} // namespace

void GroupCopyOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results.add<EraseIdentityGroupCopyOp>(context);
}

LogicalResult GroupCopyOp::verify() {
  // check if callee ends with '_array' suffix
  auto calleeAttr =
      this->getOperation()->getAttrOfType<::mlir::StringAttr>("callee");
  if (!calleeAttr)
    return this->emitError("lacking callee");
  std::string calleeStr = calleeAttr.getValue().str();
  size_t len = calleeStr.size();
  if (len < 6 || calleeStr.find("_array", len - 6) == std::string::npos)
    return this->emitError("invalid callee");
  return verifyOpInEntryPointFunc(this->getOperation());
}

//===----------------------------------------------------------------------===//
// AliasOp
//===----------------------------------------------------------------------===/

namespace {
struct CollapseAliasChain : public OpRewritePattern<AliasOp> {
  using OpRewritePattern<AliasOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AliasOp aliasOp,
                                PatternRewriter &rewriter) const override {
    if (auto sourceOp = aliasOp.getSource().getDefiningOp<AliasOp>()) {
      rewriter.replaceOpWithNewOp<AliasOp>(
          aliasOp, aliasOp.getTarget().getType(), sourceOp.getSource(),
          aliasOp.getOffset() + sourceOp.getOffset());
      return success();
    }
    return failure();
  }
};
struct RemoveIdentityAliasOp : public OpRewritePattern<AliasOp> {
  using OpRewritePattern<AliasOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AliasOp aliasOp,
                                PatternRewriter &rewriter) const override {
    if (aliasOp.getSource().getType() == aliasOp.getTarget().getType() &&
        aliasOp.getOffset() == 0) {
      rewriter.replaceOp(aliasOp, aliasOp.getSource());
      return success();
    }
    return failure();
  }
};
} // namespace

// verify AliasOp
LogicalResult AliasOp::verify() {
  return verifyOpInEntryPointFunc(this->getOperation());
}

void AliasOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                          MLIRContext *context) {
  results.add<CollapseAliasChain, RemoveIdentityAliasOp>(context);
}

std::string AliasOp::getCalleeName() { return "AliasOp"; }

Value AliasOp::getViewSource() { return getSource(); }

//===----------------------------------------------------------------------===//
// CustomOp
//===----------------------------------------------------------------------===/

void CustomOp::build(OpBuilder &builder, OperationState &result,
                     StringRef lib_path, StringRef api_name, ValueRange inputs,
                     ValueRange outputs, ArrayAttr extra_args) {
  SmallVector<Attribute> memoryEffectAttrs;
  memoryEffectAttrs.append(
      inputs.size(), builder.getAttr<MemoryEffectAttr>(MemoryEffect::Read));
  memoryEffectAttrs.append(
      outputs.size(), builder.getAttr<MemoryEffectAttr>(MemoryEffect::Write));
  build(builder, result, TypeRange{}, lib_path, api_name,
        llvm::to_vector(llvm::concat<Value>(llvm::to_vector(inputs),
                                            llvm::to_vector(outputs))),
        extra_args, builder.getArrayAttr(memoryEffectAttrs));
}

std::string CustomOp::getCalleeName() { return "custom"; }

LogicalResult CustomOp::verify() {
  return verifyOpInEntryPointFunc(this->getOperation());
}

// LWC: ignore Async for now
//
//===----------------------------------------------------------------------===//
// AsyncOpInterface
//===----------------------------------------------------------------------===//

void byre::addAsyncDependency(Operation *op, Value token) {
  op->insertOperands(0, {token});
  if (!op->template hasTrait<OpTrait::AttrSizedOperandSegments>())
    return;
  auto attrName =
      OpTrait::AttrSizedOperandSegments<void>::getOperandSegmentSizeAttr();
  auto sizeAttr = op->template getAttrOfType<DenseI32ArrayAttr>(attrName);
  if (!sizeAttr)
    return; // Async dependencies is the only variadic operand.
  SmallVector<int32_t, 8> sizes(sizeAttr.asArrayRef());
  ++sizes.front();
  op->setAttr(attrName, Builder(op->getContext()).getI32VectorAttr(sizes));
}

SmallVector<Value> ByreOp::getInputs() {
  auto op = getOperation();
  if (auto iface = llvm::dyn_cast<MemoryEffectOpInterface>(op)) {
    if (!iface.hasNoEffect()) {
      return llvm::to_vector(
          llvm::make_filter_range(op->getOperands(), [&](Value value) {
            return iface.getEffectOnValue<MemoryEffects::Read>(value)
                .has_value();
          }));
    }
  }
  return op->getOperands();
}

SmallVector<Value> ByreOp::getOutputs() {
  auto op = getOperation();
  if (auto iface = llvm::dyn_cast<MemoryEffectOpInterface>(op)) {
    if (!iface.hasNoEffect()) {
      auto outputs = llvm::to_vector(
          llvm::make_filter_range(op->getOperands(), [&](Value value) {
            return iface.getEffectOnValue<MemoryEffects::Write>(value)
                .has_value();
          }));
      outputs.append(op->result_begin(), op->result_end());
      return outputs;
    }
  }
  return op->getResults();
}

#include "byteir/Dialect/Byre/ByreOpInterfaces.cpp.inc"

#define GET_OP_CLASSES
#include "byteir/Dialect/Byre/ByreOps.cpp.inc"
