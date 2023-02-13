//===- OFModifyEntryPoint.cpp ---------------------------------------------===//
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

#include "onnx-frontend/src/Conversion/OFModifyEntryPoint.hpp"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace {

struct OFModifyEntryPointPass
    : public onnx_frontend::OFModifyEntryPointBase<OFModifyEntryPointPass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OFModifyEntryPointPass)

  OFModifyEntryPointPass(){};

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    std::set<std::string> entryPointNames;

    // remove ONNXEntryPointOp from module
    moduleOp.walk([&](mlir::ONNXEntryPointOp entryPointOp) {
      entryPointNames.insert(
          entryPointOp.getFuncAttr().getLeafReference().getValue().str());
      entryPointOp.erase();
    });

    if (entryPointNames.size() != 1) {
      moduleOp->emitError()
          << "Expect 1 onnx entry point, got " << entryPointNames.size();
      signalPassFailure();
      return;
    }
    std::string entryPointName = *entryPointNames.begin();

    for (auto funcOp : moduleOp.getOps<func::FuncOp>()) {
      if (funcOp.getSymName().str() != entryPointName) {
        continue; // not an entry point
      }

      if (!funcOp->hasAttrOfType<ArrayAttr>("input_names")) {
        funcOp->emitOpError()
            << "ArrayAttr input_names not found in main funcOp";
        signalPassFailure();
        break;
      }
      if (!funcOp->hasAttrOfType<ArrayAttr>("output_names")) {
        funcOp->emitOpError()
            << "ArrayAttr output_names not found in main funcOp";
        signalPassFailure();
        break;
      }
      ArrayAttr inputNames = funcOp->getAttrOfType<ArrayAttr>("input_names");
      ArrayAttr outputNames = funcOp->getAttrOfType<ArrayAttr>("output_names");

      if (inputNames.size() != funcOp.getNumArguments()) {
        funcOp->emitOpError()
            << "Incorrect number of input_names, expect "
            << funcOp.getNumArguments() << ", got " << inputNames.size();
        signalPassFailure();
        break;
      }
      if (outputNames.size() != funcOp.getNumResults()) {
        funcOp->emitOpError()
            << "Incorrect number of output_names, expect "
            << funcOp.getNumResults() << ", got " << outputNames.size();
        signalPassFailure();
        break;
      }

      OpBuilder builder(&getContext());
      llvm::SmallVector<NamedAttribute> byteirAttrs = {
          NamedAttribute(builder.getStringAttr("inputs"), inputNames),
          NamedAttribute(builder.getStringAttr("outputs"), outputNames)};

      funcOp->setAttr("byteir.entry_point",
                      builder.getDictionaryAttr(byteirAttrs));
      funcOp->removeAttr("input_names");
      funcOp->removeAttr("output_names");
      funcOp.setSymNameAttr(builder.getStringAttr("main"));
    }
  }
};

} // namespace

namespace onnx_frontend {
std::unique_ptr<mlir::Pass> createOFModifyEntryPointPass() {
  return std::make_unique<OFModifyEntryPointPass>();
}
} // namespace onnx_frontend