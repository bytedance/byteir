//===- GenTITConfig.cpp ---------------------------------------*--- C++ -*-===//
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

#include "byteir/Conversion/ToTIT/ToTIT.h"
#include "byteir/Dialect/Byre/Common.h"
#include "byteir/Dialect/mhlo/Transforms/HloFuser.h"
#include "byteir/Utils/FuncUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/Debug.h"

#include "../PassDetail.h"

using namespace mlir;

namespace {

static LogicalResult AttachTITConfigToAttr(
    func::FuncOp func,
    const std::string &titPtxPath,
    const std::string &gridsizeXArg,
    const std::string &gridsizeYArg,
    const std::string &gridsizeZArg,
    const std::string &blocksizeXArg,
    const std::string &blocksizeYArg,
    const std::string &blocksizeZArg) {
  
  
  std::string device_name;
  std::string byreKernelName;
  if (titPtxPath.find(".ptx") != std::string::npos) {
    device_name = "cuda";
    byreKernelName="PTXOp";
  }

  if (device_name.empty()|| byreKernelName.empty()) {
    return func.emitError("Invalid device type for TIT configuration");
  }
  addGenericFuncAttrs(func, byreKernelName);

  mlir::OpBuilder opBuilder(func);
  llvm::StringMap<mlir::Attribute> titConfig;

  // Attach the Byre Tensor Info
  titConfig["call_convention"] = opBuilder.getStringAttr("bare_ptr");
  titConfig["device"] = opBuilder.getStringAttr(device_name);
  titConfig["device_file_name"] = opBuilder.getStringAttr(titPtxPath);


  llvm::StringMap<llvm::StringRef> gpuLaunchArgs = {
      {"BlockSize.x", blocksizeXArg},
      {"BlockSize.y", blocksizeYArg},
      {"BlockSize.z", blocksizeZArg},
      {"GridSize.x", gridsizeXArg},
      {"GridSize.y", gridsizeYArg},
      {"GridSize.z", gridsizeZArg}};

  for (auto &kv : gpuLaunchArgs) {
    int val;
    if (kv.second.getAsInteger(0, val)) {
      return func.emitError("Invalid integer format for ") << kv.first();
    }
    if (val <= 0) {
      return func.emitError("Value must be positive for ") << kv.first();
    }
    titConfig[kv.first()] = opBuilder.getI32IntegerAttr(val);
  }

  for (auto &kv : titConfig) {
    func->setAttr(byre::getByrePrefix() + kv.first().str(), kv.second);
  }

  return success();
}

struct GenTITConfigPass : public GenTITConfigBase<GenTITConfigPass> {
  GenTITConfigPass(
      ArrayRef<std::string> funcNames,
      ArrayRef<std::string> titPtxPaths,
      ArrayRef<std::string> gridsizeXArgs,
      ArrayRef<std::string> gridsizeYArgs,
      ArrayRef<std::string> gridsizeZArgs,
      ArrayRef<std::string> blocksizeXArgs,
      ArrayRef<std::string> blocksizeYArgs,
      ArrayRef<std::string> blocksizeZArgs
      )
      : GenTITConfigBase() {
    this->funcNames = funcNames;
    this->titPtxPaths = titPtxPaths;
    this->gridsizeXArgs = gridsizeXArgs;
    this->gridsizeYArgs = gridsizeYArgs;
    this->gridsizeZArgs = gridsizeZArgs;
    this->blocksizeXArgs = blocksizeXArgs;
    this->blocksizeYArgs = blocksizeYArgs;
    this->blocksizeZArgs = blocksizeZArgs;
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (!func->hasAttr(getByteIRCatFusionAttrName()))
      return;
    for (size_t i = 0; i < funcNames.size(); ++i)
      if (func.getSymName() == funcNames[i]) {
        if (failed(AttachTITConfigToAttr(func, titPtxPaths[i], gridsizeXArgs[i], 
                gridsizeYArgs[i], gridsizeZArgs[i], blocksizeXArgs[i], 
                blocksizeYArgs[i], blocksizeZArgs[i]))) {
          return signalPassFailure();
        }
      }
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createGenTITConfigPass(
  ArrayRef<std::string> funcNames,
  ArrayRef<std::string> titPtxPaths,
  ArrayRef<std::string> gridsizeXArgs,
  ArrayRef<std::string> gridsizeYArgs,
  ArrayRef<std::string> gridsizeZArgs,
  ArrayRef<std::string> blocksizeXArgs,
  ArrayRef<std::string> blocksizeYArgs,
  ArrayRef<std::string> blocksizeZArgs
) {
  return std::make_unique<GenTITConfigPass>(funcNames, titPtxPaths, gridsizeXArgs, gridsizeYArgs, gridsizeZArgs, blocksizeXArgs, blocksizeYArgs, blocksizeZArgs);
}
