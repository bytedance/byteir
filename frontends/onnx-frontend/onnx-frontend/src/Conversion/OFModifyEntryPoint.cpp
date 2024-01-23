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

      ArrayAttr argAttrs = funcOp.getArgAttrsAttr();
      ArrayAttr resAttrs = funcOp.getResAttrsAttr();

      SmallVector<Attribute, 4> inputNames;
      SmallVector<Attribute, 4> outputNames;

      for (unsigned int i = 0; i < funcOp.getNumArguments(); i++) {
        if (argAttrs) {
          DictionaryAttr dictAttrs =
              llvm::dyn_cast<DictionaryAttr>(argAttrs[i]);
          if (dictAttrs && dictAttrs.contains("onnx.name")) {
            Attribute inputName =
                dictAttrs.getNamed("onnx.name").value().getValue();
            inputNames.push_back(inputName);
          } else {
            funcOp->emitOpError() << "Attr onnx.name not found in arg";
            signalPassFailure();
            break;
          }
        }
      }

      for (unsigned int i = 0; i < funcOp.getNumResults(); i++) {
        if (resAttrs) {
          DictionaryAttr dictAttrs =
              llvm::dyn_cast<DictionaryAttr>(resAttrs[i]);
          if (dictAttrs && dictAttrs.contains("onnx.name")) {
            Attribute outputName =
                dictAttrs.getNamed("onnx.name").value().getValue();
            outputNames.push_back(outputName);
          } else {
            funcOp->emitOpError() << "Attr onnx.name not found in res";
            signalPassFailure();
            break;
          }
        }
      }

      OpBuilder builder(&getContext());
      ArrayAttr inputNamesAttr = builder.getArrayAttr(inputNames);
      ArrayAttr outputNamesAttr = builder.getArrayAttr(outputNames);
      llvm::SmallVector<NamedAttribute> byteirAttrs = {
          NamedAttribute(builder.getStringAttr("inputs"), inputNamesAttr),
          NamedAttribute(builder.getStringAttr("outputs"), outputNamesAttr)};

      funcOp->setAttr("byteir.entry_point",
                      builder.getDictionaryAttr(byteirAttrs));
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
