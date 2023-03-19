//===- Common.h -----------------------------------------------*--- C++ -*-===//
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

#ifndef BYTEIR_CONVERSION_TOBYRE_COMMON_H
#define BYTEIR_CONVERSION_TOBYRE_COMMON_H

#include "byteir/Dialect/Byre/ByreDialect.h"
#include "byteir/Utils/Utils.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include <string>

namespace mlir {

std::string getByreKey(StringRef original, TypeRange in_types,
                       TypeRange out_types, bool appendArgTypes);

template <typename OpTy>
std::enable_if_t<OpTy::template hasTrait<MemoryEffectOpInterface::Trait>(),
                 byre::ComputeOp>
replaceLmhloOpWithByreComputeOp(PatternRewriter &rewriter, OpTy op,
                                StringRef callee, ValueRange newOperands) {
  assert(newOperands.size() == op->getNumOperands());
  auto iface = llvm::cast<MemoryEffectOpInterface>(op.getOperation());
  SmallVector<Attribute> memoryEffectAttrs;
  memoryEffectAttrs.reserve(op->getNumOperands());
  for (auto oldValue : op->getOperands()) {
    auto effect = byre::MemoryEffect::None;
    if (iface.template getEffectOnValue<MemoryEffects::Read>(oldValue)) {
      effect = effect | byre::MemoryEffect::Read;
    }
    if (iface.template getEffectOnValue<MemoryEffects::Write>(oldValue)) {
      effect = effect | byre::MemoryEffect::Write;
    }
    memoryEffectAttrs.push_back(
        rewriter.getAttr<byre::MemoryEffectAttr>(effect));
  }
  return rewriter.replaceOpWithNewOp<byre::ComputeOp>(
      op, callee, newOperands, rewriter.getArrayAttr(memoryEffectAttrs));
}

template <typename SrcOpTy>
class ConvertToByrePattern : public OpConversionPattern<SrcOpTy> {
public:
  ConvertToByrePattern(MLIRContext *ctx, const llvm::StringMap<StringRef> &lut,
                       bool appendTypes)
      : OpConversionPattern<SrcOpTy>(ctx), srcToCallee(lut),
        appendArgTypes(appendTypes) {}

  LogicalResult
  matchAndRewrite(SrcOpTy op, typename SrcOpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto found = srcToCallee.find(op.getOperation()->getName().getStringRef());
    if (found == srcToCallee.end()) {
      // TODO adding more error message
      return failure();
    }

    SmallVector<Type> inputTypes, outputTypes;
    if (auto iface = llvm::cast<MemoryEffectOpInterface>(op.getOperation()))
      for (auto operand : op->getOperands()) {
        if (iface.template getEffectOnValue<MemoryEffects::Read>(operand))
          inputTypes.push_back(operand.getType());
        if (iface.template getEffectOnValue<MemoryEffects::Write>(operand))
          outputTypes.push_back(operand.getType());
      }
    auto key =
        getByreKey(found->second, inputTypes, outputTypes, appendArgTypes);

    // Note all attrs will be removed
    replaceLmhloOpWithByreComputeOp(rewriter, op, key, adaptor.getOperands());

    return success();
  }

protected:
  const llvm::StringMap<StringRef> &srcToCallee;
  bool appendArgTypes;
};

template <typename SrcOpTy>
class ConvertToByrePatternWithAllAttrs : public OpConversionPattern<SrcOpTy> {
public:
  ConvertToByrePatternWithAllAttrs(MLIRContext *ctx,
                                   const llvm::StringMap<StringRef> &lut,
                                   bool appendTypes)
      : OpConversionPattern<SrcOpTy>(ctx), srcToCallee(lut),
        appendArgTypes(appendTypes) {}

  LogicalResult
  matchAndRewrite(SrcOpTy op, typename SrcOpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto found = srcToCallee.find(op.getOperation()->getName().getStringRef());
    if (found == srcToCallee.end()) {
      // TODO adding more error message
      return failure();
    }

    SmallVector<Type> inputTypes, outputTypes;
    if (auto iface = llvm::cast<MemoryEffectOpInterface>(op.getOperation()))
      for (auto operand : op->getOperands()) {
        if (iface.template getEffectOnValue<MemoryEffects::Read>(operand))
          inputTypes.push_back(operand.getType());
        if (iface.template getEffectOnValue<MemoryEffects::Write>(operand))
          outputTypes.push_back(operand.getType());
      }
    auto key =
        getByreKey(found->second, inputTypes, outputTypes, appendArgTypes);

    auto computeOp = replaceLmhloOpWithByreComputeOp(rewriter, op, key,
                                                     adaptor.getOperands());
    addAttrs(computeOp.getOperation(), op->getAttrs());
    return success();
  }

protected:
  const llvm::StringMap<StringRef> &srcToCallee;
  bool appendArgTypes;
};

} // namespace mlir

#endif // BYTEIR_CONVERSION_TOBYRE_COMMON_H