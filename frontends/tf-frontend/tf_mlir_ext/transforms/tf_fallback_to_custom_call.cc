//===- tf_fallback_to_custom_call.cc --------------------------*--- C++ -*-===//
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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include "byteir/Dialect/Ace/AceDialect.h"
#include "tf_mlir_ext/transforms/passes_detail.h"
#include "tf_mlir_ext/transforms/tf_fallback_to_custom_call.h"
#include "tf_mlir_ext/utils/customcall.h"

#include "mhlo/IR/hlo_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

using namespace mlir;
using namespace llvm;
using namespace mlir::tfext;

namespace {

void ReplaceOp(Operation *old_op, Operation *new_op) {
  // replace uses of old op results to new op
  for (auto old_and_new_results :
       llvm::zip(old_op->getResults(), new_op->getResults())) {
    Value old_result = std::get<0>(old_and_new_results);
    Value new_result = std::get<1>(old_and_new_results);
    for (OpOperand &use : llvm::make_early_inc_range(old_result.getUses())) {
      use.set(new_result);
    }
  }
  old_op->erase();
}

void ReplaceTFTypeToAceType(SmallVector<Type> &types, MLIRContext *ctx) {
  for (Type &ty : types) {
    auto tensor_type = ty.dyn_cast<mlir::TensorType>();
    if (tensor_type) {
      if (tensor_type.getElementType().isa<TF::StringType>()) {
        ty = tensor_type.cloneWith(std::nullopt, ace::StringType::get(ctx));
      } else if (tensor_type.getElementType().isa<TF::ResourceType>()) {
        ty = tensor_type.cloneWith(std::nullopt, ace::ResourceType::get(ctx));
      }
    } else {
      if (ty.isa<TF::StringType>()) {
        ty = ace::StringType::get(ctx);
      } else if (ty.isa<TF::ResourceType>()) {
        ty = ace::ResourceType::get(ctx);
      }
    }
  }
}

void LowerToMhloCustomCall(Operation *op) {
  OpBuilder builder(op);
  bool has_side_effect = false;
  auto iface = dyn_cast<MemoryEffectOpInterface>(op);
  if (iface && !iface.hasNoEffect()) {
    has_side_effect = true;
  }

  mhlo::CustomCallOp custom_call_op = builder.create<mhlo::CustomCallOp>(
      op->getLoc(), op->getResults().getTypes(), op->getOperands(),
      op->getName().getStringRef(), has_side_effect, builder.getStringAttr(""),
      mhlo::CustomCallApiVersion{
          mhlo::CustomCallApiVersion::API_VERSION_ORIGINAL},
      builder.getArrayAttr(ArrayRef<Attribute>{}),
      mhlo::CustomCallSchedule{mhlo::CustomCallSchedule::NONE}, nullptr,
      nullptr, builder.getArrayAttr(ArrayRef<Attribute>{}));
  mlir::DictionaryAttr attrs = getCleanAttr(op);
  custom_call_op->setAttr(getByteIRAttrs(), attrs);
  ReplaceOp(op, custom_call_op.operator->());
}

void LowerToAceCustomCall(Operation *op) {
  OpBuilder builder(op);

  // TODO(lyq): handle side effect
  llvm::SmallVector<Type> new_types =
      llvm::to_vector(op->getResults().getTypes());
  ReplaceTFTypeToAceType(new_types, op->getContext());
  ace::CustomCallOp custom_call_op = builder.create<ace::CustomCallOp>(
      op->getLoc(), new_types, op->getOperands(), op->getName().getStringRef());
  mlir::DictionaryAttr attrs = getCleanAttr(op);
  custom_call_op->setAttr(getByteIRAttrs(), attrs);
  ReplaceOp(op, custom_call_op.operator->());
}

void LowerToAceConstant(TF::ConstOp op) {
  ShapedType ty = op.getOutput().getType().dyn_cast<ShapedType>();
  // TODO(lyq): handle resource type
  if (!ty || !ty.getElementType().isa<TF::StringType>()) {
    return;
  }
  OpBuilder builder(op);

  auto new_ty =
      ty.cloneWith(std::nullopt, ace::StringType::get(op->getContext()));
  llvm::SmallVector<llvm::StringRef> value =
      llvm::to_vector(op.getValue().getValues<llvm::StringRef>());
  ace::ConstOp ace_const_op = builder.create<ace::ConstOp>(
      op->getLoc(), new_ty, DenseStringElementsAttr::get(new_ty, value));
  ReplaceOp(op, ace_const_op);
}

void LowerToAceReshape(TF::SqueezeOp op) {
  ShapedType ty = op.getOutput().getType().dyn_cast<ShapedType>();
  // TODO(lyq): handle resource type
  if (!ty || !ty.getElementType().isa<TF::StringType>() ||
      !ty.hasStaticShape()) {
    return;
  }
  OpBuilder builder(op);

  auto new_result_ty =
      ty.cloneWith(std::nullopt, ace::StringType::get(op->getContext()));
  ace::ReshapeOp ace_reshape_op = builder.create<ace::ReshapeOp>(
      op->getLoc(), new_result_ty, op->getOperand(0));
  ReplaceOp(op, ace_reshape_op);
}

void RewriteTFPrint(Operation *op) {
  auto operands = llvm::to_vector(op->getOperands());
  auto results = op->getResults();
  if (results.size() == 1 && operands[0].getType() == results[0].getType()) {
    results[0].replaceAllUsesWith(operands[0]);
    op->erase();
  }
}

struct TfFallbackToCustomCallPass
    : public TfFallbackToCustomCallBase<TfFallbackToCustomCallPass> {
  TfFallbackToCustomCallPass() = default;

  void runOnOperation() override final {
    MLIRContext *ctx = &getContext();
    func::FuncOp funcOp = getOperation();

    auto check_is_string_or_resource = [](Type t) {
      auto tensor_type = t.dyn_cast<mlir::TensorType>();
      Type elementType =
          tensor_type == nullptr ? t : tensor_type.getElementType();
      return elementType.isa<TF::StringType>() ||
             elementType.isa<ace::StringType>() ||
             elementType.isa<TF::ResourceType>();
    };

    func::ReturnOp returnOp;
    funcOp.walk([&](Operation *op) {
      if (llvm::isa<func::ReturnOp>(op)) {
        returnOp = llvm::cast<func::ReturnOp>(op);
        return;
      }
      if (op->getDialect() !=
          op->getContext()->getLoadedDialect<TF::TensorFlowDialect>())
        return;
      if (op->getName().getStringRef() == "tf.Print") {
        RewriteTFPrint(op);
        return;
      }
      if (llvm::isa<TF::ConstOp>(op)) {
        LowerToAceConstant(llvm::cast<TF::ConstOp>(op));
        return;
      }
      if (llvm::isa<TF::SqueezeOp>(op)) {
        LowerToAceReshape(llvm::cast<TF::SqueezeOp>(op));
        return;
      }
      if (llvm::any_of(op->getOperandTypes(), check_is_string_or_resource) ||
          llvm::any_of(op->getResultTypes(), check_is_string_or_resource)) {
        LowerToAceCustomCall(op);
      } else {
        LowerToMhloCustomCall(op);
      }
    });

    llvm::SmallVector<Type> new_arguments_type =
        llvm::to_vector(funcOp.getArgumentTypes());
    ReplaceTFTypeToAceType(new_arguments_type, ctx);
    size_t argu_size = funcOp.getArguments().size();
    for (size_t i = 0; i < argu_size; i++) {
      funcOp.getArgument(i).setType(new_arguments_type[i]);
    }
    assert(returnOp && "returnOp must exist");
    llvm::SmallVector<Type> new_results_type =
        llvm::to_vector(returnOp->getOperandTypes());
    funcOp.setType(FunctionType::get(funcOp.getContext(), new_arguments_type,
                                     new_results_type));
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::tfext::createTfFallbackToCustomCallPass() {
  return std::make_unique<TfFallbackToCustomCallPass>();
}
