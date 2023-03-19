//===- builder_test.cc ----------------------------------------*--- C++ -*-===//
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
#include "brt/test/common/models.h"
#include "byteir/Dialect/Byre/ByreDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "gtest/gtest.h"
#include <string>

using namespace brt;
using namespace brt::common;
using namespace brt::ir;
using namespace brt::test;
using namespace mlir;

using AT = mlir::byre::EntryFuncArgType;

static inline void Cleanup(ByREBuilder &builder) {
  OwningOpRef<ModuleOp> _ = builder.GetModuleOp();
}

TEST(IRBuilderTest, ByREBuilder) {
  ByREBuilder byre_builder;
  CreateCustom(byre_builder, "cpu");

  // dump
  // byre_builder.GetModuleOp().dump();
  Cleanup(byre_builder);
}

TEST(IRBuilderTest, ProgressiveBuildMethod1) {
  ByREBuilder byre_builder;

  // init
  {
    mlir::ModuleOp m = byre_builder.GetModuleOp();
    static_cast<void>(m);
    auto ctx = byre_builder.GetMLIRContext();
    auto op_builder = OpBuilder(ctx);

    llvm::SmallVector<int64_t, 4> shape = {100, 32};
    auto arg_type = MemRefType::get(shape, op_builder.getF32Type());

    // create an entry func
    func::FuncOp func_op = byre_builder.CreateEntryPointFuncSignature(
        "test", {{arg_type, AT::Input, "A"},
                 {arg_type, AT::Input, "B"},
                 {arg_type, AT::Output, "C"},
                 {arg_type, AT::Output, "D"}});

    // add entry function body
    func_op.addEntryBlock();
  }

  // Insert Op1
  {
    auto ctx = byre_builder.GetMLIRContext();
    auto op_builder = OpBuilder(ctx);
    mlir::Block *entry_block = byre_builder.GetEntryPointFuncBodyBlock();
    op_builder.setInsertionPointToEnd(entry_block);

    // insert AddOp
    SmallVector<Value> inputs{entry_block->getArgument(0),
                              entry_block->getArgument(1)};
    SmallVector<Value> outputs{entry_block->getArgument(2)};
    op_builder.create<byre::ComputeOp>(UnknownLoc::get(ctx), "AddOp_f32f32_f32",
                                       inputs, outputs);
  }

  // Insert Op2
  {
    auto ctx = byre_builder.GetMLIRContext();
    auto op_builder = OpBuilder(ctx);
    mlir::Block *entry_block = byre_builder.GetEntryPointFuncBodyBlock();
    op_builder.setInsertionPointToEnd(entry_block);

    SmallVector<Value> inputs{entry_block->getArgument(1),
                              entry_block->getArgument(2)};
    SmallVector<Value> outputs{entry_block->getArgument(3)};
    op_builder.create<byre::ComputeOp>(UnknownLoc::get(ctx), "CustomAddOp",
                                       inputs, outputs);
  }

  // Insert Return to end the entry point Func
  {
    auto ctx = byre_builder.GetMLIRContext();
    auto op_builder = OpBuilder(ctx);
    mlir::Block *entry_block = byre_builder.GetEntryPointFuncBodyBlock();
    op_builder.setInsertionPointToEnd(entry_block);

    op_builder.create<mlir::func::ReturnOp>(UnknownLoc::get(ctx));
  }

  // dump
  // byre_builder.GetModuleOp().dump();
  Cleanup(byre_builder);
}

TEST(IRBuilderTest, ProgressiveBuildMethod2) {
  ByREBuilder byre_builder;

  // init
  {
    mlir::ModuleOp m = byre_builder.GetModuleOp();
    static_cast<void>(m);
    auto ctx = byre_builder.GetMLIRContext();
    auto op_builder = OpBuilder(ctx);

    llvm::SmallVector<int64_t, 4> shape = {100, 32};
    auto arg_type = MemRefType::get(shape, op_builder.getF32Type());

    // create an entry func
    func::FuncOp func_op = byre_builder.CreateEntryPointFuncSignature(
        "test", {{arg_type, AT::Input, "A"},
                 {arg_type, AT::Input, "B"},
                 {arg_type, AT::Output, "C"},
                 {arg_type, AT::Output, "D"}});

    // add entry function body
    func_op.addEntryBlock();
  }

  // Insert Op1, same as Method 1, but record the Op
  {
    auto ctx = byre_builder.GetMLIRContext();
    auto op_builder = OpBuilder(ctx);
    mlir::Block *entry_block = byre_builder.GetEntryPointFuncBodyBlock();
    op_builder.setInsertionPointToEnd(entry_block);

    // insert AddOp
    SmallVector<Value> inputs{entry_block->getArgument(0),
                              entry_block->getArgument(1)};
    SmallVector<Value> outputs{entry_block->getArgument(2)};
    auto op = op_builder.create<byre::ComputeOp>(
        UnknownLoc::get(ctx), "AddOp_f32f32_f32", inputs, outputs);
    byre_builder.RecordOperation(op);
  }

  // Insert Op2
  {
    auto ctx = byre_builder.GetMLIRContext();
    auto op_builder = OpBuilder(ctx);
    mlir::Block *entry_block = byre_builder.GetEntryPointFuncBodyBlock();
    mlir::Operation *record_op = byre_builder.GetRecordOperation();
    op_builder.setInsertionPointAfter(record_op);

    SmallVector<Value> inputs{entry_block->getArgument(1),
                              entry_block->getArgument(2)};
    SmallVector<Value> outputs{entry_block->getArgument(3)};
    auto op = op_builder.create<byre::ComputeOp>(
        UnknownLoc::get(ctx), "CustomAddOp", inputs, outputs);
    byre_builder.RecordOperation(op);
  }

  // Insert Return to end the entry point Func
  {
    auto ctx = byre_builder.GetMLIRContext();
    auto op_builder = OpBuilder(ctx);
    mlir::Operation *record_op = byre_builder.GetRecordOperation();
    op_builder.setInsertionPointAfter(record_op);

    op_builder.create<mlir::func::ReturnOp>(UnknownLoc::get(ctx));
  }

  // dump
  // byre_builder.GetModuleOp().dump();
  Cleanup(byre_builder);
}
