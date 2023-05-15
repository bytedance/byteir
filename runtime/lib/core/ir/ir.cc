//===- ir.cc --------------------------------------------------*--- C++ -*-===//
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

#include "brt/core/ir/ir.h"

#include "brt/core/common/common.h"
#include "brt/core/ir/util.h"
#include "byteir/Dialect/Ace/AceDialect.h" // include ace.string
#include "byteir/Dialect/Byre/ByreDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include <memory>
#include <unordered_map>

using namespace brt;
using namespace brt::common;
using namespace brt::ir;
using namespace llvm;
using namespace mlir;
using namespace mlir::byre;

namespace brt {
namespace ir {

// IRHandleImp defintion
struct ByREHandleImpl {
  DialectRegistry registry;
  std::unique_ptr<MLIRContext> mlir_ctx_ptr;
  OwningOpRef<ModuleOp> module_ref;
};

/**
 * InitializeMLIR initialize MLIR related state.
 */
static common::Status InitializeMLIR(ByREHandleImpl &impl) {
  // register ByteIR's dialects here
  impl.registry.insert<mlir::func::FuncDialect, mlir::memref::MemRefDialect,
                       mlir::byre::ByreDialect, mlir::ace::AceDialect>();

  // create mlir_context
  impl.mlir_ctx_ptr =
      std::unique_ptr<MLIRContext>(new MLIRContext(impl.registry));

  return Status::OK();
}

ByREHandle::ByREHandle() : IRHandle(), impl_(new ByREHandleImpl()) {}

ByREHandle::~ByREHandle() {}

/**
 * InitializeIR function initialize IR related state.
 * It is assumed to be once called once in the beginning.
 */
common::Status ByREHandle::Initialize() {
  auto mlirStatus = InitializeMLIR(getImpl());
  return mlirStatus;
}

/**
 * Load a file
 */
common::Status ByREHandle::Load(const std::string &url,
                                const std::string &fmt) {

  if (fmt != "byre") {
    Status status = Status(BRT, INVALID_ARGUMENT, "unsupported format " + fmt);
    return status;
  }
  ir_path_ = url;
  // Set up the input file.
  std::string errorMessage;
  auto file = openInputFile(url, &errorMessage);
  if (!file) {
    Status status = Status(BRT, FAIL, errorMessage);
    return status;
  }

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(file), llvm::SMLoc());
  // get OwningModuleRef
  impl_->module_ref =
      parseSourceFile<ModuleOp>(sourceMgr, impl_->mlir_ctx_ptr.get());
  return Status::OK();
}

/**
 * Load IR from a memory
 */
common::Status ByREHandle::LoadFromMemory(const void *ptr,
                                          const std::string &fmt) {

  if (fmt != "byre") {
    Status status = Status(BRT, INVALID_ARGUMENT, "unsupported format " + fmt);
    return status;
  }

  // get OwningModuleRef
  impl_->module_ref = mlir::ModuleOp::getFromOpaquePointer(ptr);

  return Status::OK();
}

void ByREHandle::dump() {
  if (impl_ == nullptr || impl_->module_ref.get() == nullptr) {
    return;
  }

  return impl_->module_ref->dump();
}

common::Status ByREHandle::IterateNode(
    std::function<mlir::WalkResult(mlir::Operation *)> func) {
  // iterate all entry function ops in this module.
  for (auto entry : impl_->module_ref->getOps<func::FuncOp>()) {

    // skip non entry-point function
    if (!entry->hasAttr(ByreDialect::getEntryPointFunctionAttrName())) {
      continue;
    }

    auto walkResult = entry.walk(func);
    if (walkResult.wasInterrupted()) {
      return Status(BRT, FAIL, "IR Iterate interrupted");
    }
  }
  return Status::OK();
}

common::Status ByREHandle::IterateEntryFuncArg(
    std::function<common::Status(mlir::BlockArgument)> func) {
  // iterate all entry function ops in this module.
  for (auto entry : impl_->module_ref->getOps<func::FuncOp>()) {

    // skip non entry-point function
    if (!entry->hasAttr(ByreDialect::getEntryPointFunctionAttrName())) {
      continue;
    }

    // TODO how to handle multiple entry function vs multiple model
    for (auto block_arg : entry.getArguments()) {
      auto status_result = func(block_arg);
      if (!status_result.IsOK()) {
        return status_result;
      }
    }
  }
  return Status::OK();
}

std::string ByREHandle::GetOpKind(mlir::byre::ByreOp op) {
  return op.getCalleeName();
}

std::string ByREHandle::GetKey(mlir::byre::ByreOp op) {
  return op.getCalleeName();
}

std::string ByREHandle::GetOpUID(mlir::byre::ByreOp op) {
  std::string name = op.getCalleeName();
  name += std::to_string((uint64_t)op.getAsOpaquePointer());
  return name;
}

std::string &ByREHandle::GetIRPath() { return ir_path_; }

void ByREHandle::InitGraphInfoNameAndArgOffset(GraphInfo &info) {

  // create name_to_arg_offset
  for (func::FuncOp entry : impl_->module_ref->getOps<func::FuncOp>()) {
    // skip non entry-point function
    if (!entry->hasAttr(ByreDialect::getEntryPointFunctionAttrName())) {
      continue;
    }

    info.weight_count = info.io_count = 0;
    for (size_t idx = 0; idx < entry.getNumArguments(); ++idx) {
      if (auto argTypeAttr = entry.getArgAttrOfType<EntryFuncArgTypeAttr>(
              idx, ByreDialect::getEntryPointFuncArgTypeAttrName())) {
        if (auto argNameAttr = entry.getArgAttrOfType<StringAttr>(
                idx, ByreDialect::getEntryPointFuncArgNameAttrName())) {
          using ATy = EntryFuncArgType;
          auto argType = argTypeAttr.getValue();
          if (argType == ATy::None) {
            continue;
          }

          std::string argName = argNameAttr.getValue().str();
          info.name_to_arg_offset.emplace(argName, idx);
          if (bitEnumContainsAll(argType, ATy::Input)) {
            info.input_names.push_back(argName);
            info.input_arg_offsets.push_back(idx);
          }
          if (bitEnumContainsAll(argType, ATy::Output)) {
            info.output_names.push_back(argName);
            info.output_arg_offsets.push_back(idx);
          }
          if (bitEnumContainsAll(argType, ATy::Weight)) {
            info.weight_names.push_back(argName);
            info.weight_arg_offsets.push_back(idx);
            info.weight_count++;
          }
          if (bitEnumContainsAny(argType, ATy::Input | ATy::Output)) {
            info.io_count++;
          }
        }
      }
      if (auto argAliasIndexAttr = entry.getArgAttrOfType<IntegerAttr>(
              idx, ByreDialect::getEntryPointFuncArgAliasIndexAttrName())) {
        auto aliasindex = argAliasIndexAttr.getValue().getSExtValue();
        info.arg_to_arg_alias_offset[idx] = aliasindex;
      }
    }
  }
}

} // namespace ir
} // namespace brt
