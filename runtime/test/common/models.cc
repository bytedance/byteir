//===- models.cc ----------------------------------------------*--- C++ -*-===//
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

#include "brt/test/common/models.h"
#include "brt/core/common/common.h"
#include "brt/core/common/utils/math_helper.h"
#include "brt/core/ir/builder.h"
#include "brt/core/ir/util.h"
#include "byteir/Dialect/Ace/AceDialect.h"
#include "byteir/Dialect/Byre/ByreDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include <unordered_set>

using namespace brt;
using namespace brt::ir;
using namespace brt::test;
using namespace mlir;
using namespace mlir::byre;
using namespace mlir::memref;

using AT = EntryFuncArgType;

namespace brt {
namespace test {

const void *CreateAddOp2(brt::ir::ByREBuilder &byre_builder,
                         const std::string &space) {

  mlir::ModuleOp m = byre_builder.GetModuleOp();
  auto ctx = byre_builder.GetMLIRContext();
  auto op_builder = OpBuilder(ctx);

  auto space_attr = StringAttr::get(ctx, space);
  llvm::SmallVector<int64_t, 4> shape = {100, 32};
  auto arg_type = MemRefType::get(shape, op_builder.getF32Type(),
                                  MemRefLayoutAttrInterface{}, space_attr);

  // create an entry func
  func::FuncOp func_op = byre_builder.CreateEntryPointFuncSignature(
      "test", {{arg_type, AT::Input, "A"},
               {arg_type, AT::Input, "B"},
               {arg_type, AT::Output, "C"},
               {arg_type, AT::Output, "D"}});

  // add entry function body
  mlir::Block *entry_block = func_op.addEntryBlock();
  op_builder.setInsertionPointToStart(entry_block);

  // insert AddOp
  op_builder.create<byre::ComputeOp>(
      UnknownLoc::get(ctx), "AddOp_f32f32_f32",
      ValueRange{entry_block->getArgument(0), entry_block->getArgument(1)},
      ValueRange{entry_block->getArgument(2)});

  // insert AddOp
  op_builder.create<byre::ComputeOp>(
      UnknownLoc::get(ctx), "AddOp_f32f32_f32",
      ValueRange{entry_block->getArgument(1), entry_block->getArgument(2)},
      ValueRange{entry_block->getArgument(3)});

  //  insert ReturnOp
  op_builder.create<mlir::func::ReturnOp>(UnknownLoc::get(ctx));

  return m.getAsOpaquePointer();
}

const void *CreateAddWeight(brt::ir::ByREBuilder &byre_builder,
                            const std::string &space) {

  mlir::ModuleOp m = byre_builder.GetModuleOp();
  auto ctx = byre_builder.GetMLIRContext();
  auto op_builder = OpBuilder(ctx);

  llvm::SmallVector<int64_t, 4> shape = {100, 32};
  auto space_attr = StringAttr::get(ctx, space);
  auto arg_type = MemRefType::get(shape, op_builder.getF32Type(),
                                  MemRefLayoutAttrInterface{}, space_attr);

  // create an entry func
  func::FuncOp func_op = byre_builder.CreateEntryPointFuncSignature(
      "test", {{arg_type, AT::Weight, "A"},
               {arg_type, AT::Input, "B"},
               {arg_type, AT::Output, "C"},
               {arg_type, AT::Output, "D"}});

  // add entry function body
  mlir::Block *entry_block = func_op.addEntryBlock();
  op_builder.setInsertionPointToStart(entry_block);

  // insert AddOp
  op_builder.create<byre::ComputeOp>(
      UnknownLoc::get(ctx), "AddOp_f32f32_f32",
      ValueRange{entry_block->getArgument(0), entry_block->getArgument(1)},
      ValueRange{entry_block->getArgument(2)});

  // insert AddOp
  op_builder.create<byre::ComputeOp>(
      UnknownLoc::get(ctx), "AddOp_f32f32_f32",
      ValueRange{entry_block->getArgument(1), entry_block->getArgument(2)},
      ValueRange{entry_block->getArgument(3)});

  //  insert ReturnOp
  op_builder.create<mlir::func::ReturnOp>(UnknownLoc::get(ctx));

  return m.getAsOpaquePointer();
}

const void *CreateCopyOp(brt::ir::ByREBuilder &byre_builder,
                         const std::string &src_name,
                         const std::string &dst_name) {

  mlir::ModuleOp m = byre_builder.GetModuleOp();
  auto ctx = byre_builder.GetMLIRContext();
  auto op_builder = OpBuilder(ctx);

  llvm::SmallVector<int64_t, 4> shape = {100, 32};
  auto src_attr = StringAttr::get(ctx, src_name);
  auto dst_attr = StringAttr::get(ctx, dst_name);
  std::string callee = src_name + "2" + dst_name;
  auto callee_attr = StringAttr::get(ctx, callee);

  auto input_type = MemRefType::get(shape, op_builder.getF32Type(),
                                    MemRefLayoutAttrInterface{}, src_attr);
  auto output_type = MemRefType::get(shape, op_builder.getF32Type(),
                                     MemRefLayoutAttrInterface{}, dst_attr);

  // create an entry func
  func::FuncOp func_op = byre_builder.CreateEntryPointFuncSignature(
      "test", {{input_type, AT::Input, "A"}, {output_type, AT::Output, "B"}});

  // add entry function body
  mlir::Block *entry_block = func_op.addEntryBlock();
  op_builder.setInsertionPointToStart(entry_block);

  // insert CopyOp
  auto copy_op = op_builder.create<byre::CopyOp>(UnknownLoc::get(ctx),
                                                 entry_block->getArgument(0),
                                                 entry_block->getArgument(1));

  copy_op->setAttr("callee", callee_attr);

  //  insert ReturnOp
  op_builder.create<mlir::func::ReturnOp>(UnknownLoc::get(ctx));
  return m.getAsOpaquePointer();
}

const void *CreateCustom(brt::ir::ByREBuilder &byre_builder,
                         const std::string &space) {

  mlir::ModuleOp m = byre_builder.GetModuleOp();
  auto ctx = byre_builder.GetMLIRContext();
  auto op_builder = OpBuilder(ctx);

  llvm::SmallVector<int64_t, 4> shape = {100, 32};
  auto space_attr = StringAttr::get(ctx, space);
  auto arg_type = MemRefType::get(shape, op_builder.getF32Type(),
                                  MemRefLayoutAttrInterface{}, space_attr);

  // create an entry func
  func::FuncOp func_op = byre_builder.CreateEntryPointFuncSignature(
      "test", {{arg_type, AT::Input, "A"},
               {arg_type, AT::Input, "B"},
               {arg_type, AT::Output, "C"},
               {arg_type, AT::Output, "D"}});

  // add entry function body
  mlir::Block *entry_block = func_op.addEntryBlock();
  op_builder.setInsertionPointToStart(entry_block);

  // insert AddOp
  op_builder.create<byre::ComputeOp>(
      UnknownLoc::get(ctx), "AddOp_f32f32_f32",
      ValueRange{entry_block->getArgument(0), entry_block->getArgument(1)},
      ValueRange{entry_block->getArgument(2)});

  // insert CustomAddOp
  op_builder.create<byre::ComputeOp>(
      UnknownLoc::get(ctx), "CustomAddOp",
      ValueRange{entry_block->getArgument(1), entry_block->getArgument(2)},
      ValueRange{entry_block->getArgument(3)});

  //  insert ReturnOp
  op_builder.create<mlir::func::ReturnOp>(UnknownLoc::get(ctx));

  return m.getAsOpaquePointer();
}

const void *CreateUnknown(brt::ir::ByREBuilder &byre_builder,
                          const std::string &space) {

  mlir::ModuleOp m = byre_builder.GetModuleOp();
  auto ctx = byre_builder.GetMLIRContext();
  auto op_builder = OpBuilder(ctx);

  llvm::SmallVector<int64_t, 4> shape = {100, 32};
  auto space_attr = StringAttr::get(ctx, space);
  auto arg_type = MemRefType::get(shape, op_builder.getF32Type(),
                                  MemRefLayoutAttrInterface{}, space_attr);

  // create an entry func
  func::FuncOp func_op = byre_builder.CreateEntryPointFuncSignature(
      "test", {{arg_type, AT::Input, "A"},
               {arg_type, AT::Input, "B"},
               {arg_type, AT::Output, "C"},
               {arg_type, AT::Output, "D"}});

  // add entry function body
  mlir::Block *entry_block = func_op.addEntryBlock();
  op_builder.setInsertionPointToStart(entry_block);

  // insert AddOp
  op_builder.create<byre::ComputeOp>(
      UnknownLoc::get(ctx), "AddOp_f32f32_f32",
      ValueRange{entry_block->getArgument(0), entry_block->getArgument(1)},
      ValueRange{entry_block->getArgument(2)});

  // insert UnknownOp
  op_builder.create<byre::ComputeOp>(
      UnknownLoc::get(ctx), "UnknownOp",
      ValueRange{entry_block->getArgument(1), entry_block->getArgument(2)},
      ValueRange{entry_block->getArgument(3)});

  //  insert ReturnOp
  op_builder.create<mlir::func::ReturnOp>(UnknownLoc::get(ctx));

  return m.getAsOpaquePointer();
}

const void *CreateMatmul(brt::ir::ByREBuilder &byre_builder, DTypeEnum dataType,
                         const std::string &space, int64_t m, int64_t n,
                         int64_t k, int64_t lhs_contracting_dimension /*=1*/,
                         int64_t rhs_contracting_dimension /*=0*/,
                         bool output_transpose /*=false*/,
                         bool compute_on_fp16 /*=false*/) {

  mlir::ModuleOp module_op = byre_builder.GetModuleOp();
  auto ctx = byre_builder.GetMLIRContext();
  auto op_builder = OpBuilder(ctx);

  auto space_attr = StringAttr::get(ctx, space);
  std::string op_name = "MatmulOp";
  mlir::Type type;
  if (dataType == DTypeEnum::Float32) {
    op_name = op_name + "_f32f32_f32";
    type = op_builder.getF32Type();
  } else if (dataType == DTypeEnum::Float16) {
    op_name = op_name + "_f16f16_f16";
    type = op_builder.getF16Type();
  } else {
    BRT_THROW("invalid data type");
  }

  // int64_t m = 128;
  // int64_t k = 32;
  // int64_t n = 64;

  llvm::SmallVector<int64_t, 4> shape_A =
      lhs_contracting_dimension == 1 ? llvm::SmallVector<int64_t>{m, k}
                                     : llvm::SmallVector<int64_t>{k, m};
  auto type_A =
      MemRefType::get(shape_A, type, MemRefLayoutAttrInterface{}, space_attr);

  llvm::SmallVector<int64_t, 4> shape_B =
      rhs_contracting_dimension == 0 ? llvm::SmallVector<int64_t>{k, n}
                                     : llvm::SmallVector<int64_t>{n, k};
  auto type_B =
      MemRefType::get(shape_B, type, MemRefLayoutAttrInterface{}, space_attr);

  llvm::SmallVector<int64_t, 4> shape_C =
      output_transpose ? llvm::SmallVector<int64_t>{n, m}
                       : llvm::SmallVector<int64_t>{m, n};
  auto type_C =
      MemRefType::get(shape_C, type, MemRefLayoutAttrInterface{}, space_attr);

  // create an entry func
  func::FuncOp func_op = byre_builder.CreateEntryPointFuncSignature(
      "test", {{type_A, AT::Input, "A"},
               {type_B, AT::Input, "B"},
               {type_C, AT::Output, "C"}});

  // add entry function body
  mlir::Block *entry_block = func_op.addEntryBlock();
  op_builder.setInsertionPointToStart(entry_block);

  // insert Op
  auto compute_op = op_builder.create<byre::ComputeOp>(
      UnknownLoc::get(ctx), op_name,
      ValueRange{entry_block->getArgument(0), entry_block->getArgument(1)},
      ValueRange{entry_block->getArgument(2)});
  compute_op->setAttr("lhs_contracting_dimension",
                      op_builder.getI64IntegerAttr(lhs_contracting_dimension));
  compute_op->setAttr("rhs_contracting_dimension",
                      op_builder.getI64IntegerAttr(rhs_contracting_dimension));
  if (output_transpose) {
    compute_op->setAttr("output_transpose", op_builder.getUnitAttr());
  }
  if (compute_on_fp16) {
    compute_op->setAttr("compute_on_fp16", op_builder.getUnitAttr());
  }

  //  insert ReturnOp
  op_builder.create<mlir::func::ReturnOp>(UnknownLoc::get(ctx));
  return module_op.getAsOpaquePointer();
}

const void *CreateMatmul2(brt::ir::ByREBuilder &byre_builder,
                          const std::string &space) {

  mlir::ModuleOp module_op = byre_builder.GetModuleOp();
  auto ctx = byre_builder.GetMLIRContext();
  auto op_builder = OpBuilder(ctx);

  auto space_attr = StringAttr::get(ctx, space);

  llvm::SmallVector<int64_t, 4> shape_arg_0 = {128, 32};
  auto type_arg_0 = MemRefType::get(shape_arg_0, op_builder.getF32Type(),
                                    MemRefLayoutAttrInterface{}, space_attr);

  llvm::SmallVector<int64_t, 4> shape_arg_1 = {32, 64};
  auto type_arg_1 = MemRefType::get(shape_arg_1, op_builder.getF32Type(),
                                    MemRefLayoutAttrInterface{}, space_attr);

  llvm::SmallVector<int64_t, 4> shape_arg_2 = {64, 64};
  auto type_arg_2 = MemRefType::get(shape_arg_2, op_builder.getF32Type(),
                                    MemRefLayoutAttrInterface{}, space_attr);

  llvm::SmallVector<int64_t, 4> shape_arg_3 = {128, 64};
  auto type_arg_3 = MemRefType::get(shape_arg_3, op_builder.getF32Type(),
                                    MemRefLayoutAttrInterface{}, space_attr);

  llvm::SmallVector<int64_t, 4> shape_arg_4 = {128, 64};
  auto type_arg_4 = MemRefType::get(shape_arg_4, op_builder.getF32Type(),
                                    MemRefLayoutAttrInterface{}, space_attr);

  // create an entry func
  func::FuncOp func_op = byre_builder.CreateEntryPointFuncSignature(
      "test", {{type_arg_0, AT::Input, "A"},
               {type_arg_1, AT::Input, "B"},
               {type_arg_2, AT::Input, "C"},
               {type_arg_3, AT::Output, "D"},
               {type_arg_4, AT::Output, "E"}});

  // add entry function body
  mlir::Block *entry_block = func_op.addEntryBlock();
  op_builder.setInsertionPointToStart(entry_block);

  // insert Ops
  auto matmul_op1 = op_builder.create<byre::ComputeOp>(
      UnknownLoc::get(ctx), "MatmulOp_f32f32_f32",
      ValueRange{entry_block->getArgument(0), entry_block->getArgument(1)},
      ValueRange{entry_block->getArgument(3)});
  matmul_op1->setAttr("lhs_contracting_dimension",
                      op_builder.getI64IntegerAttr(1));
  matmul_op1->setAttr("rhs_contracting_dimension",
                      op_builder.getI64IntegerAttr(0));

  auto matmul_op2 = op_builder.create<byre::ComputeOp>(
      UnknownLoc::get(ctx), "MatmulOp_f32f32_f32",
      ValueRange{entry_block->getArgument(3), entry_block->getArgument(2)},
      ValueRange{entry_block->getArgument(4)});
  matmul_op2->setAttr("lhs_contracting_dimension",
                      op_builder.getI64IntegerAttr(1));
  matmul_op2->setAttr("rhs_contracting_dimension",
                      op_builder.getI64IntegerAttr(0));

  //  insert ReturnOp
  op_builder.create<mlir::func::ReturnOp>(UnknownLoc::get(ctx));

  return module_op.getAsOpaquePointer();
}

const void *CreateBatchMatmul(brt::ir::ByREBuilder &byre_builder,
                              const std::string &space) {
  mlir::ModuleOp module_op = byre_builder.GetModuleOp();
  auto ctx = byre_builder.GetMLIRContext();
  auto op_builder = OpBuilder(ctx);

  int64_t batch_count0 = 2;
  int64_t batch_count1 = 17;
  int64_t m = 128;
  int64_t k = 32;
  int64_t n = 64;

  auto space_attr = StringAttr::get(ctx, space);

  llvm::SmallVector<int64_t, 4> shape_A{batch_count0, batch_count1, m, k};
  auto type_A = MemRefType::get(shape_A, op_builder.getF32Type(),
                                MemRefLayoutAttrInterface{}, space_attr);

  llvm::SmallVector<int64_t, 4> shape_B{batch_count0, batch_count1, k, n};
  auto type_B = MemRefType::get(shape_B, op_builder.getF32Type(),
                                MemRefLayoutAttrInterface{}, space_attr);

  llvm::SmallVector<int64_t, 4> shape_C{batch_count0, batch_count1, m, n};
  auto type_C = MemRefType::get(shape_C, op_builder.getF32Type(),
                                MemRefLayoutAttrInterface{}, space_attr);

  // create an entry func
  func::FuncOp func_op = byre_builder.CreateEntryPointFuncSignature(
      "test", {{type_A, AT::Input, "A"},
               {type_B, AT::Input, "B"},
               {type_C, AT::Output, "C"}});

  // add entry function body
  mlir::Block *entry_block = func_op.addEntryBlock();
  op_builder.setInsertionPointToStart(entry_block);

  // insert Op
  auto compute_op = op_builder.create<byre::ComputeOp>(
      UnknownLoc::get(ctx), "BatchMatmulOp_f32f32_f32",
      ValueRange{entry_block->getArgument(0), entry_block->getArgument(1)},
      ValueRange{entry_block->getArgument(2)});

  compute_op->setAttr("lhs_batching_dimensions",
                      op_builder.getI64ArrayAttr({0, 1}));
  compute_op->setAttr("rhs_batching_dimensions",
                      op_builder.getI64ArrayAttr({0, 1}));
  compute_op->setAttr("lhs_contracting_dimension",
                      op_builder.getI64IntegerAttr(3));
  compute_op->setAttr("rhs_contracting_dimension",
                      op_builder.getI64IntegerAttr(2));

  //  insert ReturnOp
  op_builder.create<mlir::func::ReturnOp>(UnknownLoc::get(ctx));

  return module_op.getAsOpaquePointer();
}

const void *CreateConv(brt::ir::ByREBuilder &byre_builder, const std::string op,
                       DTypeEnum dataType, const std::string &space, int64_t N,
                       int64_t iC, int64_t iH, int64_t iW, int64_t oC,
                       int64_t kH, int64_t kW, const std::string &layout,
                       int64_t strideH, int64_t strideW, int64_t paddingH,
                       int64_t paddingW, int64_t dilateH, int64_t dilateW) {
  mlir::ModuleOp module_op = byre_builder.GetModuleOp();
  auto ctx = byre_builder.GetMLIRContext();
  auto op_builder = OpBuilder(ctx);

  std::vector<int64_t> shape_input;
  std::vector<int64_t> shape_filter;
  if (layout == "NHWC") {
    shape_input = {N, iH, iW, iC};
    shape_filter = {oC, kH, kW, iC};
  } else if (layout == "NCHW") {
    shape_input = {N, iC, iH, iW};
    shape_filter = {oC, iC, kH, kW};
  } else {
    BRT_THROW("invalid conv layout");
  }
  std::vector<int64_t> shape_output = brt::conv::DeduceOutputShape(
      shape_input, shape_filter, layout, strideH, strideW, paddingH, paddingW,
      dilateH, dilateW);

  std::string op_name;
  mlir::Type type;
  if (dataType == DTypeEnum::Float32) {
    op_name = op + "_f32f32_f32";
    type = op_builder.getF32Type();
  } else if (dataType == DTypeEnum::Float16) {
    op_name = op + "_f16f16_f16";
    type = op_builder.getF16Type();
  } else {
    BRT_THROW("invalid data type");
  }

  auto space_attr = StringAttr::get(ctx, space);
  MemRefType type_A, type_B, type_C;
  if (op == "ConvOp") {
    type_A = MemRefType::get(shape_input, type, MemRefLayoutAttrInterface{},
                             space_attr);
    type_B = MemRefType::get(shape_filter, type, MemRefLayoutAttrInterface{},
                             space_attr);
    type_C = MemRefType::get(shape_output, type, MemRefLayoutAttrInterface{},
                             space_attr);
  } else if (op == "ConvBackwardDataOp") {
    type_A = MemRefType::get(shape_output, type, MemRefLayoutAttrInterface{},
                             space_attr);
    type_B = MemRefType::get(shape_filter, type, MemRefLayoutAttrInterface{},
                             space_attr);
    type_C = MemRefType::get(shape_input, type, MemRefLayoutAttrInterface{},
                             space_attr);
  } else if (op == "ConvBackwardFilterOp") {
    type_A = MemRefType::get(shape_input, type, MemRefLayoutAttrInterface{},
                             space_attr);
    type_B = MemRefType::get(shape_output, type, MemRefLayoutAttrInterface{},
                             space_attr);
    type_C = MemRefType::get(shape_filter, type, MemRefLayoutAttrInterface{},
                             space_attr);
  } else {
    BRT_THROW("invalid conv op name");
  }

  // create an entry func
  func::FuncOp func_op = byre_builder.CreateEntryPointFuncSignature(
      "test", {{type_A, AT::Input, "A"},
               {type_B, AT::Input, "B"},
               {type_C, AT::Output, "C"}});
  // add entry function body
  mlir::Block *entry_block = func_op.addEntryBlock();
  op_builder.setInsertionPointToStart(entry_block);
  // insert Op
  auto compute_op = op_builder.create<byre::ComputeOp>(
      UnknownLoc::get(ctx), op_name,
      ValueRange{entry_block->getArgument(0), entry_block->getArgument(1)},
      ValueRange{entry_block->getArgument(2)});
  compute_op->setAttr("input_layout", op_builder.getStringAttr(layout));
  compute_op->setAttr("kernel_layout", op_builder.getStringAttr(layout));
  compute_op->setAttr("output_layout", op_builder.getStringAttr(layout));
  compute_op->setAttr("window_strides",
                      op_builder.getI64TensorAttr({strideH, strideW}));
  compute_op->setAttr("padding", op_builder.getI64TensorAttr(
                                     {paddingH, paddingH, paddingW, paddingW}));
  compute_op->setAttr("lhs_dilation", op_builder.getI64TensorAttr({1, 1}));
  compute_op->setAttr("rhs_dilation",
                      op_builder.getI64TensorAttr({dilateH, dilateW}));
  compute_op->setAttr("feature_group_count", op_builder.getI64IntegerAttr(1));
  compute_op->setAttr("batch_group_count", op_builder.getI64IntegerAttr(1));

  //  insert ReturnOp
  op_builder.create<mlir::func::ReturnOp>(UnknownLoc::get(ctx));

  return module_op.getAsOpaquePointer();
}

const void *CreatePoolMax(brt::ir::ByREBuilder &byre_builder,
                          DTypeEnum dataType, const std::string &space,
                          std::vector<int64_t> &shape_input,
                          std::vector<int64_t> &shape_output,
                          std::vector<int64_t> &padding,
                          std::vector<int64_t> &window_dimensions,
                          std::vector<int64_t> &window_strides) {
  mlir::ModuleOp module_op = byre_builder.GetModuleOp();
  auto ctx = byre_builder.GetMLIRContext();
  auto op_builder = OpBuilder(ctx);

  std::string op_name;
  mlir::Type type;
  if (dataType == DTypeEnum::Float32) {
    op_name = "PoolMaxOp_f32_f32";
    type = op_builder.getF32Type();
  } else if (dataType == DTypeEnum::Float16) {
    op_name = "PoolMaxOp_f16_f16";
    type = op_builder.getF16Type();
  } else {
    BRT_THROW("invalid data type");
  }

  auto space_attr = StringAttr::get(ctx, space);
  auto type_input = MemRefType::get(shape_input, type,
                                    MemRefLayoutAttrInterface{}, space_attr);
  auto type_output = MemRefType::get(shape_output, type,
                                     MemRefLayoutAttrInterface{}, space_attr);

  // create an entry func
  func::FuncOp func_op = byre_builder.CreateEntryPointFuncSignature(
      "test", {{type_input, AT::Input, "A"}, {type_output, AT::Output, "B"}});
  // add entry function body
  mlir::Block *entry_block = func_op.addEntryBlock();
  op_builder.setInsertionPointToStart(entry_block);
  // insert Op
  auto compute_op = op_builder.create<byre::ComputeOp>(
      UnknownLoc::get(ctx), op_name, ValueRange{entry_block->getArgument(0)},
      ValueRange{entry_block->getArgument(1)});

  auto paddingAttr = DenseIntElementsAttr::get(
      RankedTensorType::get({static_cast<int64_t>(padding.size() / 2), 2},
                            op_builder.getI64Type()),
      padding);
  compute_op->setAttr("padding", paddingAttr);
  compute_op->setAttr("window_dimensions",
                      op_builder.getI64TensorAttr(window_dimensions));
  compute_op->setAttr("window_strides",
                      op_builder.getI64TensorAttr(window_strides));

  //  insert ReturnOp
  op_builder.create<mlir::func::ReturnOp>(UnknownLoc::get(ctx));

  return module_op.getAsOpaquePointer();
}

const void *CreatePoolMaxGrad(brt::ir::ByREBuilder &byre_builder,
                              DTypeEnum dataType, const std::string &space,
                              std::vector<int64_t> &shape_x,
                              std::vector<int64_t> &shape_y,
                              std::vector<int64_t> &padding,
                              std::vector<int64_t> &window_dimensions,
                              std::vector<int64_t> &window_strides) {
  mlir::ModuleOp module_op = byre_builder.GetModuleOp();
  auto ctx = byre_builder.GetMLIRContext();
  auto op_builder = OpBuilder(ctx);

  std::string op_name;
  mlir::Type type;
  if (dataType == DTypeEnum::Float32) {
    op_name = "PoolMaxGradOp_f32f32_f32";
    type = op_builder.getF32Type();
  } else if (dataType == DTypeEnum::Float16) {
    op_name = "PoolMaxGradOp_f16f16_f16";
    type = op_builder.getF16Type();
  } else {
    BRT_THROW("invalid data type");
  }

  auto space_attr = StringAttr::get(ctx, space);
  auto type_x =
      MemRefType::get(shape_x, type, MemRefLayoutAttrInterface{}, space_attr);
  auto type_y =
      MemRefType::get(shape_y, type, MemRefLayoutAttrInterface{}, space_attr);

  // create an entry func
  func::FuncOp func_op = byre_builder.CreateEntryPointFuncSignature(
      "test", {{type_x, AT::Input, "x"},
               {type_y, AT::Input, "dy"},
               {type_x, AT::Output, "dx"}});

  // add entry function body
  mlir::Block *entry_block = func_op.addEntryBlock();
  op_builder.setInsertionPointToStart(entry_block);
  // insert Op
  auto compute_op = op_builder.create<byre::ComputeOp>(
      UnknownLoc::get(ctx), op_name,
      ValueRange{entry_block->getArgument(0), entry_block->getArgument(1)},
      ValueRange{entry_block->getArgument(2)});

  auto paddingAttr = DenseIntElementsAttr::get(
      RankedTensorType::get({static_cast<int64_t>(padding.size() / 2), 2},
                            op_builder.getI64Type()),
      padding);
  compute_op->setAttr("padding", paddingAttr);
  compute_op->setAttr("window_dimensions",
                      op_builder.getI64TensorAttr(window_dimensions));
  compute_op->setAttr("window_strides",
                      op_builder.getI64TensorAttr(window_strides));

  //  insert ReturnOp
  op_builder.create<mlir::func::ReturnOp>(UnknownLoc::get(ctx));
  return module_op.getAsOpaquePointer();
}

const void *CreateBatchNormTraining(brt::ir::ByREBuilder &byre_builder,
                                    DTypeEnum dataType,
                                    const std::string &space,
                                    std::vector<int64_t> &shape_input,
                                    int64_t feature_index, float epsilon) {
  mlir::ModuleOp module_op = byre_builder.GetModuleOp();
  auto ctx = byre_builder.GetMLIRContext();
  auto op_builder = OpBuilder(ctx);

  std::string op_name;
  mlir::Type type;
  if (dataType == DTypeEnum::Float32) {
    op_name = "BatchNormTrainingOp_f32f32f32_f32f32f32";
    type = op_builder.getF32Type();
  } else if (dataType == DTypeEnum::Float16) {
    op_name = "BatchNormTrainingOp_f16f32f32_f16f32f32";
    type = op_builder.getF16Type();
  } else {
    BRT_THROW("invalid data type");
  }

  std::vector<int64_t> shape_scale{shape_input[feature_index]};
  auto space_attr = StringAttr::get(ctx, space);
  auto type_input = MemRefType::get(shape_input, type,
                                    MemRefLayoutAttrInterface{}, space_attr);
  auto type_scale = MemRefType::get(shape_scale, op_builder.getF32Type(),
                                    MemRefLayoutAttrInterface{}, space_attr);

  // create an entry func
  func::FuncOp func_op = byre_builder.CreateEntryPointFuncSignature(
      "test", {{type_input, AT::Input, "Input"},
               {type_scale, AT::Input, "Scale"},
               {type_scale, AT::Input, "Bias"},
               {type_input, AT::Output, "Output"},
               {type_scale, AT::Output, "BatchMean"},
               {type_scale, AT::Output, "BatchVar"}});
  // add entry function body
  mlir::Block *entry_block = func_op.addEntryBlock();
  op_builder.setInsertionPointToStart(entry_block);
  // insert Op
  auto compute_op = op_builder.create<byre::ComputeOp>(
      UnknownLoc::get(ctx), op_name,
      ValueRange{entry_block->getArgument(0), entry_block->getArgument(1),
                 entry_block->getArgument(2)},
      ValueRange{entry_block->getArgument(3), entry_block->getArgument(4),
                 entry_block->getArgument(5)});
  compute_op->setAttr("feature_index",
                      op_builder.getI64IntegerAttr(feature_index));
  compute_op->setAttr("epsilon", op_builder.getF32FloatAttr(epsilon));

  //  insert ReturnOp
  op_builder.create<mlir::func::ReturnOp>(UnknownLoc::get(ctx));

  return module_op.getAsOpaquePointer();
}

const void *CreateBatchNormGrad(brt::ir::ByREBuilder &byre_builder,
                                DTypeEnum dataType, const std::string &space,
                                std::vector<int64_t> &shape_input,
                                int64_t feature_index, float epsilon) {
  mlir::ModuleOp module_op = byre_builder.GetModuleOp();
  auto ctx = byre_builder.GetMLIRContext();
  auto op_builder = OpBuilder(ctx);

  std::string op_name;
  mlir::Type type;
  if (dataType == DTypeEnum::Float32) {
    op_name = "BatchNormGradOp_f32f32f32_f32f32f32";
    type = op_builder.getF32Type();
  } else if (dataType == DTypeEnum::Float16) {
    op_name = "BatchNormGradOp_f16f32f16_f16f32f32";
    type = op_builder.getF16Type();
  } else {
    BRT_THROW("invalid data type");
  }

  std::vector<int64_t> shape_scale{shape_input[feature_index]};
  auto space_attr = StringAttr::get(ctx, space);
  auto type_input = MemRefType::get(shape_input, type,
                                    MemRefLayoutAttrInterface{}, space_attr);
  auto type_scale = MemRefType::get(shape_scale, op_builder.getF32Type(),
                                    MemRefLayoutAttrInterface{}, space_attr);

  // create an entry func
  func::FuncOp func_op = byre_builder.CreateEntryPointFuncSignature(
      "test", {{type_input, AT::Input, "Input"},
               {type_scale, AT::Input, "Scale"},
               {type_input, AT::Input, "GradOutput"},
               {type_input, AT::Output, "GradInput"},
               {type_scale, AT::Output, "GradScale"},
               {type_scale, AT::Output, "GradBias"}});
  // add entry function body
  mlir::Block *entry_block = func_op.addEntryBlock();
  op_builder.setInsertionPointToStart(entry_block);
  // insert Op
  auto compute_op = op_builder.create<byre::ComputeOp>(
      UnknownLoc::get(ctx), op_name,
      ValueRange{entry_block->getArgument(0), entry_block->getArgument(1),
                 entry_block->getArgument(2)},
      ValueRange{entry_block->getArgument(3), entry_block->getArgument(4),
                 entry_block->getArgument(5)});
  compute_op->setAttr("feature_index",
                      op_builder.getI64IntegerAttr(feature_index));
  compute_op->setAttr("epsilon", op_builder.getF32FloatAttr(epsilon));

  //  insert ReturnOp
  op_builder.create<mlir::func::ReturnOp>(UnknownLoc::get(ctx));

  return module_op.getAsOpaquePointer();
}

const void *CreateIndexPut(brt::ir::ByREBuilder &byre_builder,
                           const std::string &space,
                           std::vector<int64_t> src_shape, size_t dim,
                           std::vector<int64_t> idx_shape) {
  mlir::ModuleOp module_op = byre_builder.GetModuleOp();
  auto ctx = byre_builder.GetMLIRContext();
  auto op_builder = OpBuilder(ctx);

  BRT_ENFORCE(dim < src_shape.size() && idx_shape.size() == 1);

  auto space_attr = StringAttr::get(ctx, space);
  auto src = MemRefType::get(src_shape, op_builder.getF32Type(),
                             MemRefLayoutAttrInterface{}, space_attr);
  auto dst = MemRefType::get(src_shape, op_builder.getF32Type(),
                             MemRefLayoutAttrInterface{}, space_attr);
  auto index = MemRefType::get(idx_shape, op_builder.getI64Type(),
                               MemRefLayoutAttrInterface{}, space_attr);
  std::vector<int64_t> update_shape = src_shape;

  for (size_t i = 0; i <= dim; ++i) {
    update_shape[i] = idx_shape[i];
  }

  auto update = MemRefType::get(update_shape, op_builder.getF32Type(),
                                MemRefLayoutAttrInterface{}, space_attr);

  func::FuncOp func_op = byre_builder.CreateEntryPointFuncSignature(
      "test", {{src, AT::Input, "src"},
               {index, AT::Input, "index"},
               {update, AT::Input, "update"},
               {dst, AT::Output, "dst"}});

  // add entry function body
  mlir::Block *entry_block = func_op.addEntryBlock();
  op_builder.setInsertionPointToStart(entry_block);

  // insert Op
  auto compute_op = op_builder.create<byre::ComputeOp>(
      UnknownLoc::get(ctx), "IndexPutOp_f32i64f32_f32",
      ValueRange{entry_block->getArgument(0), entry_block->getArgument(1),
                 entry_block->getArgument(2)},
      ValueRange{entry_block->getArgument(3)});
  compute_op->setAttr("dim", op_builder.getI32IntegerAttr(dim));
  //  insert ReturnOp
  op_builder.create<mlir::func::ReturnOp>(UnknownLoc::get(ctx));
  return module_op.getAsOpaquePointer();
}

const void *CreateIndexSelect(brt::ir::ByREBuilder &byre_builder,
                              const std::string &space,
                              std::vector<int64_t> src_shape, size_t dim,
                              std::vector<int64_t> idx_shape,
                              bool is_ui32_index) {
  mlir::ModuleOp module_op = byre_builder.GetModuleOp();
  auto ctx = byre_builder.GetMLIRContext();
  auto op_builder = OpBuilder(ctx);

  BRT_ENFORCE(dim < src_shape.size() && idx_shape.size() == 1);
  auto space_attr = StringAttr::get(ctx, space);
  auto src = MemRefType::get(src_shape, op_builder.getF32Type(),
                             MemRefLayoutAttrInterface{}, space_attr);
  auto index_elem_type = is_ui32_index
                             ? op_builder.getIntegerType(32, /*isSigned=*/false)
                             : op_builder.getI64Type();
  auto index = MemRefType::get(idx_shape, index_elem_type,
                               MemRefLayoutAttrInterface{}, space_attr);
  std::vector<int64_t> dst_shape = src_shape;
  dst_shape[dim] = idx_shape[0];
  auto dst = MemRefType::get(dst_shape, op_builder.getF32Type(),
                             MemRefLayoutAttrInterface{}, space_attr);

  func::FuncOp func_op = byre_builder.CreateEntryPointFuncSignature(
      "test", {{src, AT::Input, "src"},
               {index, AT::Input, "index"},
               {dst, AT::Output, "dst"}});

  // add entry function body
  mlir::Block *entry_block = func_op.addEntryBlock();
  op_builder.setInsertionPointToStart(entry_block);

  // insert Op
  std::string op_name =
      is_ui32_index ? "IndexSelectOp_f32ui32_f32" : "IndexSelectOp_f32i64_f32";
  auto compute_op = op_builder.create<byre::ComputeOp>(
      UnknownLoc::get(ctx), op_name,
      ValueRange{entry_block->getArgument(0), entry_block->getArgument(1)},
      ValueRange{entry_block->getArgument(2)});
  compute_op->setAttr("dim", op_builder.getI32IntegerAttr(dim));
  //  insert ReturnOp
  op_builder.create<mlir::func::ReturnOp>(UnknownLoc::get(ctx));

  return module_op.getAsOpaquePointer();
}

const void *CreateReduction(brt::ir::ByREBuilder &byre_builder,
                            const std::string &space,
                            std::vector<int64_t> src_shape,
                            std::vector<int64_t> dimensions,
                            std::string reduce_op) {
  mlir::ModuleOp module_op = byre_builder.GetModuleOp();
  auto ctx = byre_builder.GetMLIRContext();
  auto op_builder = OpBuilder(ctx);

  auto dst_shape = brt::reduction::DeduceOutputShape(src_shape, dimensions);

  auto space_attr = StringAttr::get(ctx, space);
  auto src = MemRefType::get(src_shape, op_builder.getF32Type(),
                             MemRefLayoutAttrInterface{}, space_attr);
  auto dst = MemRefType::get(dst_shape, op_builder.getF32Type(),
                             MemRefLayoutAttrInterface{}, space_attr);

  func::FuncOp func_op = byre_builder.CreateEntryPointFuncSignature(
      "test", {{src, AT::Input, "src"}, {dst, AT::Output, "dst"}});

  // add entry function body
  mlir::Block *entry_block = func_op.addEntryBlock();
  op_builder.setInsertionPointToStart(entry_block);

  // insert Op
  auto compute_op = op_builder.create<byre::ComputeOp>(
      UnknownLoc::get(ctx), reduce_op, ValueRange{entry_block->getArgument(0)},
      ValueRange{entry_block->getArgument(1)});
  compute_op->setAttr(
      "dimensions",
      DenseIntElementsAttr::get(
          RankedTensorType::get({static_cast<int64_t>(dimensions.size())},
                                op_builder.getI64Type()),
          dimensions));
  //  insert ReturnOp
  op_builder.create<mlir::func::ReturnOp>(UnknownLoc::get(ctx));

  return module_op.getAsOpaquePointer();
}

const void *CreateTopK(brt::ir::ByREBuilder &byre_builder, DTypeEnum dataType,
                       DTypeEnum indexType, std::vector<int64_t> src_shape,
                       int64_t k, std::vector<int64_t> axis_vec, bool sorted) {
  mlir::ModuleOp module_op = byre_builder.GetModuleOp();
  auto ctx = byre_builder.GetMLIRContext();
  auto op_builder = OpBuilder(ctx);

  auto input_mlir_type = ConvertDTypeToMLIRType(dataType, ctx);
  auto index_mlir_type = ConvertDTypeToMLIRType(indexType, ctx);
  std::vector<int64_t> shape_output = src_shape;
  BRT_ENFORCE(axis_vec.size() == 1);
  int64_t axis = axis_vec[0];
  shape_output[axis] = k;

  auto input_type = MemRefType::get(src_shape, input_mlir_type);
  auto value_type = MemRefType::get(shape_output, input_mlir_type);
  auto indices_type = MemRefType::get(shape_output, index_mlir_type);

  // create an entry func
  func::FuncOp func_op = byre_builder.CreateEntryPointFuncSignature(
      "test", {{input_type, AT::Input, "A"},
               {value_type, AT::Output, "B"},
               {indices_type, AT::Output, "C"}});

  // add entry function body
  mlir::Block *entry_block = func_op.addEntryBlock();
  op_builder.setInsertionPointToStart(entry_block);
  auto compute_op = op_builder.create<byre::ComputeOp>(
      UnknownLoc::get(ctx), "byteir.top_k",
      ValueRange{entry_block->getArgument(0)},
      ValueRange{entry_block->getArgument(1), entry_block->getArgument(2)});
  compute_op->setAttr("k", op_builder.getI64IntegerAttr(k));
  compute_op->setAttr("axis", op_builder.getI64VectorAttr(axis_vec));
  compute_op->setAttr("sorted", op_builder.getBoolAttr(sorted));
  // insert ReturnOp
  op_builder.create<mlir::func::ReturnOp>(UnknownLoc::get(ctx));
  return module_op.getAsOpaquePointer();
}

const void *CreateTranspose(brt::ir::ByREBuilder &byre_builder,
                            DTypeEnum dataType, const std::string &space,
                            std::vector<int64_t> &shape_input,
                            std::vector<int64_t> &shape_output,
                            std::vector<int64_t> &permutation) {

  mlir::ModuleOp module_op = byre_builder.GetModuleOp();
  auto ctx = byre_builder.GetMLIRContext();
  auto op_builder = OpBuilder(ctx);

  std::string op_name;
  mlir::Type type;
  if (dataType == DTypeEnum::Float32) {
    op_name = "TransposeOp_f32_f32";
    type = op_builder.getF32Type();
  } else if (dataType == DTypeEnum::Float16) {
    op_name = "TransposeOp_f16_f16";
    type = op_builder.getF16Type();
  } else {
    BRT_THROW("invalid data type");
  }

  auto space_attr = StringAttr::get(ctx, space);
  auto type_input = MemRefType::get(shape_input, type,
                                    MemRefLayoutAttrInterface{}, space_attr);
  auto type_output = MemRefType::get(shape_output, type,
                                     MemRefLayoutAttrInterface{}, space_attr);

  // create an entry func
  func::FuncOp func_op = byre_builder.CreateEntryPointFuncSignature(
      "test",
      {{type_input, AT::Input, "Input"}, {type_output, AT::Output, "Output"}});

  // add entry function body
  mlir::Block *entry_block = func_op.addEntryBlock();
  op_builder.setInsertionPointToStart(entry_block);
  // insert Op
  auto compute_op = op_builder.create<byre::ComputeOp>(
      UnknownLoc::get(ctx), op_name, ValueRange{entry_block->getArgument(0)},
      ValueRange{entry_block->getArgument(1)});

  compute_op->setAttr("permutation", op_builder.getI64TensorAttr(permutation));

  //  insert ReturnOp
  op_builder.create<mlir::func::ReturnOp>(UnknownLoc::get(ctx));

  return module_op.getAsOpaquePointer();
}

const void *CreateTypecvt(brt::ir::ByREBuilder &byre_builder,
                          DTypeEnum src_dtype, DTypeEnum dst_dtype,
                          const std::vector<int64_t> &shape) {

  mlir::ModuleOp m = byre_builder.GetModuleOp();
  auto ctx = byre_builder.GetMLIRContext();
  auto op_builder = OpBuilder(ctx);

  auto src_mlir_type = ConvertDTypeToMLIRType(src_dtype, ctx);
  auto dst_mlir_type = ConvertDTypeToMLIRType(dst_dtype, ctx);

  auto input_type = MemRefType::get(shape, src_mlir_type);
  auto output_type = MemRefType::get(shape, dst_mlir_type);

  std::string op_name = "Typecvt";
  llvm::raw_string_ostream os(op_name);
  os << "_" << src_mlir_type << "_" << dst_mlir_type;

  // create an entry func
  func::FuncOp func_op = byre_builder.CreateEntryPointFuncSignature(
      "test", {{input_type, AT::Input, "A"}, {output_type, AT::Output, "B"}});

  // add entry function body
  mlir::Block *entry_block = func_op.addEntryBlock();
  op_builder.setInsertionPointToStart(entry_block);

  // insert TypecvtOp
  op_builder.create<byre::ComputeOp>(UnknownLoc::get(ctx), os.str(),
                                     ValueRange{entry_block->getArgument(0)},
                                     ValueRange{entry_block->getArgument(1)});

  //  insert ReturnOp
  op_builder.create<mlir::func::ReturnOp>(UnknownLoc::get(ctx));
  return m.getAsOpaquePointer();
}

const void *CreateAliasThenIndexPut(brt::ir::ByREBuilder &byre_builder,
                                    const std::string &space,
                                    std::vector<int64_t> data_src_shape,
                                    int64_t idx_src_len, int64_t idx_dst_len,
                                    int32_t idx_offset) {

  BRT_ENFORCE(idx_src_len >= (idx_dst_len + idx_offset));
  BRT_ENFORCE(0 < data_src_shape.size());

  mlir::ModuleOp module_op = byre_builder.GetModuleOp();
  auto ctx = byre_builder.GetMLIRContext();
  auto op_builder = OpBuilder(ctx);

  std::vector<int64_t> idx_shape;
  auto space_attr = StringAttr::get(ctx, space);
  auto data_src = MemRefType::get(data_src_shape, op_builder.getF32Type(),
                                  MemRefLayoutAttrInterface{}, space_attr);
  auto data_dst = MemRefType::get(data_src_shape, op_builder.getF32Type(),
                                  MemRefLayoutAttrInterface{}, space_attr);
  auto idx_src_index = MemRefType::get({idx_src_len}, op_builder.getI64Type(),
                                       MemRefLayoutAttrInterface{}, space_attr);
  auto idx_dst_index = MemRefType::get({idx_dst_len}, op_builder.getI64Type(),
                                       MemRefLayoutAttrInterface{}, space_attr);

  std::vector<int64_t> update_shape = data_src_shape;

  for (int i = 0; i <= 0; ++i) {
    update_shape[i] = idx_dst_len;
  }

  auto update = MemRefType::get(update_shape, op_builder.getF32Type(),
                                MemRefLayoutAttrInterface{}, space_attr);

  func::FuncOp func_op = byre_builder.CreateEntryPointFuncSignature(
      "test", {{data_src, AT::Input, "src"},
               {idx_src_index, AT::Input, "index"},
               {update, AT::Input, "update"},
               {data_dst, AT::Output, "dst"}});

  // add entry function body
  mlir::Block *entry_block = func_op.addEntryBlock();
  op_builder.setInsertionPointToStart(entry_block);

  // AliasOp
  auto alias_op =
      op_builder.create<byre::AliasOp>(UnknownLoc::get(ctx), idx_dst_index,
                                       entry_block->getArgument(1), idx_offset);

  // IndexPut
  auto indexput_op = op_builder.create<byre::ComputeOp>(
      UnknownLoc::get(ctx), "IndexPutOp_f32i64f32_f32",
      ValueRange{entry_block->getArgument(0), alias_op.getResult(),
                 entry_block->getArgument(2)},
      ValueRange{entry_block->getArgument(3)});

  indexput_op->setAttr("dim", op_builder.getI32IntegerAttr(0));
  //  insert ReturnOp
  op_builder.create<mlir::func::ReturnOp>(UnknownLoc::get(ctx));
  return module_op.getAsOpaquePointer();
}

const void *CreateRepeat(brt::ir::ByREBuilder &byre_builder, DTypeEnum dataType,
                         DTypeEnum timesType, std::vector<int64_t> data_shape,
                         std::vector<int64_t> times_shape,
                         std::vector<int64_t> output_shape) {
  mlir::ModuleOp m = byre_builder.GetModuleOp();
  auto ctx = byre_builder.GetMLIRContext();
  auto op_builder = OpBuilder(ctx);

  auto data_ele_type = ConvertDTypeToMLIRType(dataType, ctx);
  auto times_ele_type = ConvertDTypeToMLIRType(timesType, ctx);
  auto data_type = MemRefType::get(data_shape, data_ele_type);
  auto times_type = MemRefType::get(times_shape, times_ele_type);
  auto output_type = MemRefType::get(output_shape, data_ele_type);

  // create an entry func
  func::FuncOp func_op = byre_builder.CreateEntryPointFuncSignature(
      "test", {{data_type, AT::Input, "A"},
               {times_type, AT::Input, "B"},
               {output_type, AT::Output, "C"}});

  // add entry function body
  mlir::Block *entry_block = func_op.addEntryBlock();
  op_builder.setInsertionPointToStart(entry_block);
  op_builder.create<byre::ComputeOp>(
      UnknownLoc::get(ctx), "byteir.repeat",
      ValueRange{entry_block->getArgument(0), entry_block->getArgument(1)},
      ValueRange{entry_block->getArgument(2)});
  op_builder.create<mlir::func::ReturnOp>(UnknownLoc::get(ctx));
  return m.getAsOpaquePointer();
}

const void *CreatePTXAddOp(brt::ir::ByREBuilder &byre_builder) {

  mlir::ModuleOp m = byre_builder.GetModuleOp();
  auto ctx = byre_builder.GetMLIRContext();
  auto op_builder = OpBuilder(ctx);

  llvm::SmallVector<int64_t, 4> shape = {8, 128};
  auto space_attr = StringAttr::get(ctx, "cuda");
  auto arg_type = MemRefType::get(shape, op_builder.getF32Type(),
                                  MemRefLayoutAttrInterface{}, space_attr);

  // create an entry func
  func::FuncOp func_op = byre_builder.CreateEntryPointFuncSignature(
      "test", {{arg_type, AT::Input, "A"},
               {arg_type, AT::Input, "B"},
               {arg_type, AT::Output, "C"}});

  // add entry function body
  mlir::Block *entry_block = func_op.addEntryBlock();
  op_builder.setInsertionPointToStart(entry_block);

  // insert PTX
  auto ptx_op = op_builder.create<byre::ComputeOp>(
      UnknownLoc::get(ctx), "PTXOp",
      ValueRange{entry_block->getArgument(0), entry_block->getArgument(1)},
      ValueRange{entry_block->getArgument(2)});
  ptx_op->setAttr("kernel_name", op_builder.getStringAttr("add_kernel"));
  ptx_op->setAttr("GridSize.x", op_builder.getI32IntegerAttr(4));
  ptx_op->setAttr("BlockSize.x", op_builder.getI32IntegerAttr(256));
  ptx_op->setAttr("arg_ranks", op_builder.getI32ArrayAttr({2, 2, 2}));

  //  insert ReturnOp
  op_builder.create<mlir::func::ReturnOp>(UnknownLoc::get(ctx));
  func_op->setAttr("device_file_name", op_builder.getStringAttr(
                                           "test/test_files/llvm_ptx_add.ptx"));

  return m.getAsOpaquePointer();
}

const void *CreateTFWhereOp(brt::ir::ByREBuilder &byre_builder,
                            DTypeEnum input_dtype,
                            const std::vector<int64_t> &shape) {

  mlir::ModuleOp m = byre_builder.GetModuleOp();
  auto ctx = byre_builder.GetMLIRContext();
  auto op_builder = OpBuilder(ctx);

  auto input_mlir_type = ConvertDTypeToMLIRType(input_dtype, ctx);
  int64_t num_elements = LinearizedStaticShape(shape).value();
  std::vector<int64_t> shape_output{num_elements,
                                    static_cast<int64_t>(shape.size())};

  auto input_type = MemRefType::get(shape, input_mlir_type);
  auto output_type = MemRefType::get(shape_output, op_builder.getI64Type());

  // create an entry func
  func::FuncOp func_op = byre_builder.CreateEntryPointFuncSignature(
      "test", {{input_type, AT::Input, "A"}, {output_type, AT::Output, "B"}});

  // add entry function body
  mlir::Block *entry_block = func_op.addEntryBlock();
  op_builder.setInsertionPointToStart(entry_block);
  op_builder.create<byre::ComputeOp>(UnknownLoc::get(ctx), "tf.Where",
                                     ValueRange{entry_block->getArgument(0)},
                                     ValueRange{entry_block->getArgument(1)});
  op_builder.create<mlir::func::ReturnOp>(UnknownLoc::get(ctx));
  return m.getAsOpaquePointer();
}

const void *CreateTFSelectOp(brt::ir::ByREBuilder &byre_builder,
                             DTypeEnum dtype,
                             const std::vector<int64_t> &cond_shape,
                             const std::vector<int64_t> &input_shape) {
  mlir::ModuleOp m = byre_builder.GetModuleOp();
  auto ctx = byre_builder.GetMLIRContext();
  auto op_builder = OpBuilder(ctx);
  ctx->loadDialect<ace::AceDialect>();

  auto cond_mlir_type = ConvertDTypeToMLIRType(DTypeEnum::Bool, ctx);
  auto input_mlir_type = ConvertDTypeToMLIRType(dtype, ctx);
  auto cond_type = MemRefType::get(cond_shape, cond_mlir_type);
  auto input_type = MemRefType::get(input_shape, input_mlir_type);
  auto output_type = MemRefType::get(input_shape, input_mlir_type);

  // create an entry func
  func::FuncOp func_op = byre_builder.CreateEntryPointFuncSignature(
      "test", {{cond_type, AT::Input, "Cond"},
               {input_type, AT::Input, "A"},
               {input_type, AT::Input, "B"},
               {output_type, AT::Output, "C"}});

  // add entry function body
  mlir::Block *entry_block = func_op.addEntryBlock();
  op_builder.setInsertionPointToStart(entry_block);
  op_builder.create<byre::ComputeOp>(UnknownLoc::get(ctx), "tf.Select",
                                     ValueRange{entry_block->getArgument(0),
                                                entry_block->getArgument(1),
                                                entry_block->getArgument(2)},
                                     ValueRange{entry_block->getArgument(3)});
  op_builder.create<mlir::func::ReturnOp>(UnknownLoc::get(ctx));
  return m.getAsOpaquePointer();
}

const void *CreateTFStringToNumberOp(brt::ir::ByREBuilder &byre_builder,
                                     DTypeEnum InType, DTypeEnum OutType,
                                     const std::vector<int64_t> &input_shape) {
  mlir::ModuleOp m = byre_builder.GetModuleOp();
  auto ctx = byre_builder.GetMLIRContext();
  auto op_builder = OpBuilder(ctx);
  ctx->loadDialect<ace::AceDialect>();

  auto input_mlir_type = ConvertDTypeToMLIRType(InType, ctx);
  auto output_mlir_type = ConvertDTypeToMLIRType(OutType, ctx);
  auto input_type = MemRefType::get(input_shape, input_mlir_type);
  auto output_type = MemRefType::get(input_shape, output_mlir_type);

  // create an entry func
  func::FuncOp func_op = byre_builder.CreateEntryPointFuncSignature(
      "test", {{input_type, AT::Input, "A"}, {output_type, AT::Output, "B"}});

  // add entry function body
  mlir::Block *entry_block = func_op.addEntryBlock();
  op_builder.setInsertionPointToStart(entry_block);
  auto stringToNumberOp = op_builder.create<byre::ComputeOp>(
      UnknownLoc::get(ctx), "tf.StringToNumber",
      ValueRange{entry_block->getArgument(0)},
      ValueRange{entry_block->getArgument(1)});
  Type outTypeType;
  switch (OutType) {
  case DTypeEnum::Int32: {
    outTypeType = op_builder.getI32Type();
    break;
  }
  case DTypeEnum::Int64: {
    outTypeType = op_builder.getI64Type();
    break;
  }
  case DTypeEnum::Float32: {
    outTypeType = op_builder.getF32Type();
    break;
  }
  case DTypeEnum::Float64: {
    outTypeType = op_builder.getF64Type();
    break;
  }
  default: {
    BRT_THROW("tf.StringToNumber Op get an unsupport type.");
    break;
  }
  }
  stringToNumberOp->setAttr("out_type", TypeAttr::get(outTypeType));
  op_builder.create<mlir::func::ReturnOp>(UnknownLoc::get(ctx));
  return m.getAsOpaquePointer();
}

const void *
CreateWithEntryAttrs(brt::ir::ByREBuilder &byre_builder, DTypeEnum input_dtype,
                     const std::vector<int64_t> &shape,
                     const std::vector<std::string> &inputs,
                     const std::vector<std::string> &outputs,
                     const std::vector<std::string> &original_inputs) {
  mlir::ModuleOp m = byre_builder.GetModuleOp();
  auto ctx = byre_builder.GetMLIRContext();
  auto op_builder = OpBuilder(ctx);
  auto mlir_type = ConvertDTypeToMLIRType(input_dtype, ctx);
  auto input_type = MemRefType::get(shape, mlir_type);

  // create an entry func
  func::FuncOp func_op = byre_builder.CreateEntryPointFuncSignature(
      "test", {{input_type, AT::Input, "A"},
               {input_type, AT::Input, "B"},
               {input_type, AT::Output, "C"},
               {input_type, AT::Output, "D"}});

  // add attributes
  const auto convert_to_array_attr =
      [&ctx](const std::vector<std::string> &vec) {
        llvm::SmallVector<Attribute> attrs_vec;
        std::transform(vec.begin(), vec.end(), std::back_inserter(attrs_vec),
                       [&ctx](const std::string &ele) {
                         return StringAttr::get(ctx, ele);
                       });
        return ArrayAttr::get(ctx, attrs_vec);
      };
  std::vector<NamedAttribute> byteir_entry_point_attrs;
  byteir_entry_point_attrs.emplace_back(StringAttr::get(ctx, "inputs"),
                                        convert_to_array_attr(inputs));
  byteir_entry_point_attrs.emplace_back(StringAttr::get(ctx, "outputs"),
                                        convert_to_array_attr(outputs));
  auto byteir_entry_attrs = DictionaryAttr::get(ctx, byteir_entry_point_attrs);
  func_op->setAttr("tf.original_input_names",
                   convert_to_array_attr(original_inputs));
  func_op->setAttr("byteir.entry_point", byteir_entry_attrs);

  // add entry function body
  mlir::Block *entry_block = func_op.addEntryBlock();
  op_builder.setInsertionPointToStart(entry_block);
  //  insert ReturnOp
  op_builder.create<mlir::func::ReturnOp>(UnknownLoc::get(ctx));
  return m.getAsOpaquePointer();
}

} // namespace test
} // namespace brt
