//===- rewrite_func_attr_to_byteir.cc -------------------------*--- C++ -*-===//
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

#include "tf_mlir_ext/transforms/passes_detail.h"
#include "tf_mlir_ext/transforms/rewrite_func_attr_to_byteir.h"
#include "tf_mlir_ext/utils/customcall.h"

#include <string>
#include <vector>

using namespace mlir;
using namespace llvm;

namespace {

struct RewriteFuncAttrToByteIRPass
    : public RewriteFuncAttrToByteIRBase<RewriteFuncAttrToByteIRPass> {
  RewriteFuncAttrToByteIRPass(const std::unordered_map<std::string, Attribute>
                                  &additional_main_func_attrs) {
    this->additional_main_func_attrs = additional_main_func_attrs;
  }

  void runOnOperation() override final {
    MLIRContext *context = &getContext();
    func::FuncOp funcOp = getOperation();
    OpBuilder builder(context);

    if (funcOp->hasAttr("tf.entry_function")) {
      auto entryFunction =
          funcOp->getAttrOfType<DictionaryAttr>("tf.entry_function");
      StringRef inputs =
          entryFunction.get("inputs").cast<StringAttr>().getValue();
      StringRef outputs =
          entryFunction.get("outputs").cast<StringAttr>().getValue();
      // TODO(lyq): handle "controls" attribute
      SmallVector<StringRef> inputsName;
      inputs.split(inputsName, ",");
      SmallVector<StringRef> outputsName;
      outputs.split(outputsName, ",");
      assert(inputsName.size() == funcOp.getNumArguments());
      assert(outputsName.size() == funcOp.getNumResults());

      ArrayAttr inputsAttr = builder.getArrayAttr(llvm::to_vector(
          llvm::map_range(inputsName, [&](StringRef name) -> Attribute {
            return builder.getStringAttr(name);
          })));
      ArrayAttr outputsAttr = builder.getArrayAttr(llvm::to_vector(
          llvm::map_range(outputsName, [&](StringRef name) -> Attribute {
            return builder.getStringAttr(name);
          })));
      SmallVector<NamedAttribute> byteirAttrs = {
          NamedAttribute(builder.getStringAttr("inputs"), inputsAttr),
          NamedAttribute(builder.getStringAttr("outputs"), outputsAttr)};

      funcOp->setAttr("byteir.entry_point",
                      builder.getDictionaryAttr(byteirAttrs));
      funcOp->removeAttr("tf.entry_function");

      for (auto it : additional_main_func_attrs) {
        funcOp->setAttr(it.first, it.second);
      }

      // remove argument attributes
      funcOp->removeAttr("arg_attrs");
    }
  }

private:
  std::unordered_map<std::string, Attribute> additional_main_func_attrs;
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::tfext::createRewriteFuncAttrToByteIRPass(
    const std::unordered_map<std::string, Attribute>
        &additional_main_func_attrs) {
  return std::make_unique<RewriteFuncAttrToByteIRPass>(
      additional_main_func_attrs);
}
