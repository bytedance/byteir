//===- builder.cc ---------------------------------------------*--- C++ -*-===//
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

#include "brt/core/ir/builder.h"
#include "byteir/Dialect/Byre/ByreDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/Support/InitLLVM.h"
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

// ByREBuilderStructImpl defintion
struct ByREBuilderStructImpl {
  DialectRegistry registry;
  std::unique_ptr<MLIRContext> mlir_ctx_ptr;
  ModuleOp module_op;
  Operation *record_op = nullptr;
};

ByREBuilder::ByREBuilder() : impl_(new ByREBuilderStructImpl()) {
  // registerAllDialects(impl_->registry);
  //  register ByteIR's dialects here
  impl_->registry.insert<mlir::func::FuncDialect, mlir::memref::MemRefDialect,
                         mlir::byre::ByreDialect>();

  impl_->mlir_ctx_ptr =
      std::unique_ptr<MLIRContext>(new MLIRContext(impl_->registry));
  MLIRContext *ctx = impl_->mlir_ctx_ptr.get();

  // load func and byre
  ctx->loadDialect<byre::ByreDialect>();
  ctx->loadDialect<mlir::func::FuncDialect>();
  ctx->loadDialect<mlir::memref::MemRefDialect>();
  // ctx->loadAllAvailableDialects();  // avoid loading all

  // create module
  impl_->module_op = ModuleOp::create(UnknownLoc::get(ctx));

  // insert an attr "byre.container_module"
  impl_->module_op->setAttr(ByreDialect::getContainerModuleAttrName(),
                            UnitAttr::get(ctx));
}

ByREBuilder::~ByREBuilder() {}

func::FuncOp ByREBuilder::CreateEntryPointFuncSignature(
    const std::string &func_name,
    const std::vector<TypeAndArgAttrsPack> &types) {

  MLIRContext *ctx = impl_->mlir_ctx_ptr.get();

  OpBuilder op_builder(ctx);

  auto func_type = op_builder.getFunctionType(
      llvm::to_vector<16>(
          llvm::map_range(types, [](auto &&i) { return std::get<0>(i); })),
      {} /*results*/);

  // create a func
  func::FuncOp entry_func = op_builder.create<func::FuncOp>(
      UnknownLoc::get(ctx), func_name, func_type);

  // insert an attr "byre.entry_point"
  entry_func->setAttr(ByreDialect::getEntryPointFunctionAttrName(),
                      UnitAttr::get(ctx));

  for (size_t idx = 0; idx < entry_func.getNumArguments(); ++idx) {
    auto argType = std::get<1>(types[idx]);
    auto argName = std::get<2>(types[idx]);
    entry_func.setArgAttr(idx, ByreDialect::getEntryPointFuncArgNameAttrName(),
                          StringAttr::get(ctx, argName));
    entry_func.setArgAttr(idx, ByreDialect::getEntryPointFuncArgTypeAttrName(),
                          EntryFuncArgTypeAttr::get(ctx, argType));
  }

  // insert func to module
  impl_->module_op.push_back(entry_func);

  return entry_func;
}

mlir::ModuleOp ByREBuilder::GetModuleOp() { return impl_->module_op; }

mlir::MLIRContext *ByREBuilder::GetMLIRContext() {
  return impl_->mlir_ctx_ptr.get();
}

Block *ByREBuilder::GetEntryPointFuncBodyBlock() {

  // get first entyr func
  for (func::FuncOp entry : impl_->module_op.getOps<func::FuncOp>()) {
    // skip non entry-point function
    if (!entry->hasAttr(ByreDialect::getEntryPointFunctionAttrName())) {
      continue;
    }

    Region &body = entry.getBody();
    if (body.empty()) {
      return nullptr;
    }

    Block &block = body.front();

    return &block;
  }

  return nullptr;
}

void ByREBuilder::RecordOperation(Operation *op) { impl_->record_op = op; }

Operation *ByREBuilder::GetRecordOperation() { return impl_->record_op; }

} // namespace ir
} // namespace brt
