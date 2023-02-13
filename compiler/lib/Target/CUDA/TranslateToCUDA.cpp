//===- TranslateToCUDA.cpp ------------------------------------*--- C++ -*-===//
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
// Some code comes from CppEmitter.h and TranslateToCpp.cpp in LLVM project
// Original license:
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "byteir/Target/CUDA/CUDAEmitter.h"
#include "byteir/Target/CUDA/ToCUDA.h"

#include "byteir/Target/Common/EmitUtil.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/IndentedOstream.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"

#define DEBUG_TYPE "translate-to-cuda"

using namespace byteir;
using namespace mlir;
using namespace mlir::emitc;
using namespace mlir::gpu;
using llvm::formatv;

#define RETURN_IF_FAILED(call)                                                 \
  if (failed(call)) {                                                          \
    return failure();                                                          \
  }

namespace {
static LogicalResult printOperation(CUDAEmitter &emitter,
                                    GPUFuncOp functionOp) {
  // We need to declare variables at top if the function has multiple blocks.
  if (!emitter.shouldDeclareVariablesAtTop() &&
      functionOp.getBlocks().size() > 1) {
    return functionOp.emitOpError(
        "with multiple blocks needs variables declared at top");
  }

  CppEmitter::Scope scope(emitter);
  raw_indented_ostream &os = emitter.ostream();

  if (emitter.shouldEmitExternC()) {
    os << "extern \"C\" ";
  }

  if (!functionOp.empty())
    os << "__global__ ";

  if (failed(emitter.emitTypes(functionOp.getLoc(),
                               functionOp.getFunctionType().getResults())))
    return failure();
  os << " " << functionOp.getName();

  if (functionOp.empty()) {
    os << "(";
    if (failed(interleaveCommaWithError(
            functionOp.getFunctionType().getInputs(), os,
            [&](Type type) -> LogicalResult {
              if (failed(emitter.emitType(functionOp.getLoc(), type)))
                return failure();
              return success();
            }))) {
      return failure();
    }
    os << ");\n";
    return success();
  } else {
    os << "(";
    if (failed(interleaveCommaWithError(
            functionOp.getArguments(), os,
            [&](BlockArgument arg) -> LogicalResult {
              if (failed(emitter.emitType(functionOp.getLoc(), arg.getType())))
                return failure();
              os << " " << emitter.getOrCreateName(arg);
              return success();
            }))) {
      return failure();
    }
    os << ") {\n";
  }
  os.indent();
  if (emitter.shouldDeclareVariablesAtTop()) {
    // Declare all variables that hold op results including those from nested
    // regions.
    WalkResult result =
        functionOp.walk<WalkOrder::PreOrder>([&](Operation *op) -> WalkResult {
          for (OpResult result : op->getResults()) {
            if (failed(emitter.emitVariableDeclaration(
                    result, /*trailingSemicolon=*/true))) {
              return WalkResult(
                  op->emitError("unable to declare result variable for op"));
            }
          }
          return WalkResult::advance();
        });
    if (result.wasInterrupted())
      return failure();
  }

  Region::BlockListType &blocks = functionOp.getBlocks();
  // Create label names for basic blocks.
  for (Block &block : blocks) {
    emitter.getOrCreateName(block);
  }

  // Declare variables for basic block arguments.
  for (auto it = std::next(blocks.begin()); it != blocks.end(); ++it) {
    Block &block = *it;
    for (BlockArgument &arg : block.getArguments()) {
      if (emitter.hasValueInScope(arg))
        return functionOp.emitOpError(" block argument #")
               << arg.getArgNumber() << " is out of scope";
      if (failed(
              emitter.emitType(block.getParentOp()->getLoc(), arg.getType()))) {
        return failure();
      }
      os << " " << emitter.getOrCreateName(arg) << ";\n";
    }
  }

  for (Block &block : blocks) {
    // Only print a label if there is more than one block.
    if (blocks.size() > 1) {
      if (failed(emitter.emitLabel(block)))
        return failure();
    }
    for (Operation &op : block.getOperations()) {
      // When generating code for an scf.if or std.cond_br op no semicolon needs
      // to be printed after the closing brace.
      // When generating code for an scf.for op, printing a trailing semicolon
      // is handled within the printOperation function.
      bool trailingSemicolon =
          !isa<scf::IfOp, scf::ForOp, cf::CondBranchOp>(op);

      if (failed(emitter.emitOperation(
              op, /*trailingSemicolon=*/trailingSemicolon)))
        return failure();
    }
  }
  os.unindent() << "}\n";
  return success();
}

static LogicalResult printOperation(CUDAEmitter &emitter, ModuleOp moduleOp) {
  CppEmitter::Scope scope(emitter);

  if (emitter.shouldEmitKernelOnly()) {
    for (Operation &op : moduleOp) {
      if (!isa<GPUModuleOp>(op))
        continue;
      if (failed(emitter.emitOperation(op, /*trailingSemicolon=*/false))) {
        return failure();
      }
    }
    return success();
  }

  for (Operation &op : moduleOp) {
    if (failed(emitter.emitOperation(op, /*trailingSemicolon=*/false))) {
      return failure();
    }
  }
  return success();
}

static LogicalResult printOperation(CUDAEmitter &emitter,
                                    GPUModuleOp moduleOp) {
  CppEmitter::Scope scope(emitter);

  if (emitter.shouldEmitKernelOnly()) {
    for (Operation &op : moduleOp) {
      if (!isa<GPUFuncOp>(op))
        continue;
      if (failed(emitter.emitOperation(op, /*trailingSemicolon=*/false))) {
        return failure();
      }
    }
    return success();
  }

  for (Operation &op : moduleOp) {
    if (failed(emitter.emitOperation(op, /*trailingSemicolon=*/false))) {
      return failure();
    }
  }
  return success();
}

static LogicalResult printOperation(CUDAEmitter &emitter, GridDimOp gdimOp) {
  RETURN_IF_FAILED(emitter.emitAssignPrefix(*gdimOp.getOperation()));
  raw_ostream &os = emitter.ostream();
  os << "gridDim." << mlir::gpu::stringifyDimension(gdimOp.getDimension());
  return success();
}

static LogicalResult printOperation(CUDAEmitter &emitter, BlockDimOp bdimOp) {
  RETURN_IF_FAILED(emitter.emitAssignPrefix(*bdimOp.getOperation()));
  raw_ostream &os = emitter.ostream();
  os << "blockDim." << mlir::gpu::stringifyDimension(bdimOp.getDimension());
  return success();
}

static LogicalResult printOperation(CUDAEmitter &emitter, BlockIdOp bidOp) {
  RETURN_IF_FAILED(emitter.emitAssignPrefix(*bidOp.getOperation()));
  raw_ostream &os = emitter.ostream();
  os << "blockIdx." << mlir::gpu::stringifyDimension(bidOp.getDimension());
  return success();
}

static LogicalResult printOperation(CUDAEmitter &emitter, ThreadIdOp tidOp) {
  RETURN_IF_FAILED(emitter.emitAssignPrefix(*tidOp.getOperation()));
  raw_ostream &os = emitter.ostream();
  os << "threadIdx." << mlir::gpu::stringifyDimension(tidOp.getDimension());
  return success();
}

static LogicalResult printOperation(CUDAEmitter &emitter,
                                    gpu::ReturnOp returnOp) {
  raw_ostream &os = emitter.ostream();
  os << "return";
  switch (returnOp.getNumOperands()) {
  case 0:
    return success();
  case 1:
    os << " " << emitter.getOrCreateName(returnOp.getOperand(0));
    return success(emitter.hasValueInScope(returnOp.getOperand(0)));
  default:
    os << " std::make_tuple(";
    if (failed(emitter.emitOperandsAndAttributes(*returnOp.getOperation())))
      return failure();
    os << ")";
    return success();
  }
}

static LogicalResult printOperation(CUDAEmitter &emitter, BarrierOp barrierOp) {
  raw_ostream &os = emitter.ostream();
  os << "__syncthreads()";
  return success();
}

} // namespace

CUDAEmitter::CUDAEmitter(raw_ostream &os, bool declareVariablesAtTop,
                         bool kernelOnly, bool externC)
    : CppEmitter(os, declareVariablesAtTop), kernelOnly(kernelOnly),
      externC(externC) {}

LogicalResult CUDAEmitter::emitOperation(Operation &op,
                                         bool trailingSemicolon) {

  bool callEmitter = false;
  bool earlyTerminate = false;
  LogicalResult status =
      llvm::TypeSwitch<Operation *, LogicalResult>(&op)
          // ModuleOp
          .Case<ModuleOp>([&](auto op) { return printOperation(*this, op); })
          // GPU ops.
          .Case<GPUModuleOp, GPUFuncOp, GridDimOp, BlockDimOp, BlockIdOp,
                ThreadIdOp, BarrierOp, gpu::ReturnOp>(
              [&](auto op) { return printOperation(*this, op); })
          // bubble ops
          .Case<gpu::ModuleEndOp>([&](auto) {
            earlyTerminate = true;
            return success();
          })
          .Default([&](Operation *op) {
            callEmitter = true;
            return success();
          });

  RETURN_IF_FAILED(status)
  if (earlyTerminate)
    return success();
  if (callEmitter)
    return CppEmitter::emitOperation(op, trailingSemicolon);
  os << (trailingSemicolon ? ";\n" : "\n");
  return success();
}

LogicalResult byteir::translateToCUDA(Operation *op, raw_ostream &os,
                                      bool declareVariablesAtTop,
                                      bool kernelOnly, bool externC) {

  CUDAEmitter emitter(os, declareVariablesAtTop, kernelOnly, externC);
  return emitter.emitOperation(*op, /*trailingSemicolon=*/false);
}
