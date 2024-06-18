//===- GPUToNVVM.cpp ------------------------------------------*--- C++ -*-===//
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
// Some code comes from LowerGpuOpsToNVVMOps.cpp in LLVM project
// Original license:
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to generate NVVMIR operations for higher-level
// GPU operations.
//
//===----------------------------------------------------------------------===//

#include "byteir/Conversion/GPUToNVVM/GPUToNVVM.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/FormatVariadic.h"

#include "../PassDetail.h"

using namespace mlir;
using namespace mlir::gpu;
using namespace mlir::LLVM;
using namespace mlir::NVVM;

namespace {

template <typename SourceOp>
struct OpToFuncCallLowering : public ConvertOpToLLVMPattern<SourceOp> {
public:
  explicit OpToFuncCallLowering(LLVMTypeConverter &lowering_, StringRef f32Func,
                                StringRef f64Func)
      : ConvertOpToLLVMPattern<SourceOp>(lowering_), f32Func(f32Func),
        f64Func(f64Func) {}

  LogicalResult
  matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    using LLVM::LLVMFuncOp;

    static_assert(
        std::is_base_of<OpTrait::OneResult<SourceOp>, SourceOp>::value,
        "expected single result op");

    static_assert(std::is_base_of<OpTrait::SameOperandsAndResultType<SourceOp>,
                                  SourceOp>::value,
                  "expected op with same operand and result types");

    SmallVector<mlir::Value, 1> castedOperands;
    for (mlir::Value operand : adaptor.getOperands())
      castedOperands.push_back(maybeCast(operand, rewriter));

    mlir::Type resultType = castedOperands.front().getType();
    mlir::Type funcType = getFunctionType(resultType, castedOperands);
    StringRef funcName =
        getFunctionName(cast<LLVM::LLVMFunctionType>(funcType).getReturnType());
    if (funcName.empty())
      return failure();

    LLVMFuncOp funcOp = appendOrGetFuncOp(funcName, funcType, op);
    auto callOp = rewriter.create<LLVM::CallOp>(
        op->getLoc(), resultType, SymbolRefAttr::get(funcOp), castedOperands);

    if (resultType == adaptor.getOperands().front().getType()) {
      rewriter.replaceOp(op, {callOp.getResult()});
      return success();
    }

    mlir::Value truncated = rewriter.create<LLVM::FPTruncOp>(
        op->getLoc(), adaptor.getOperands().front().getType(),
        callOp.getResult());
    rewriter.replaceOp(op, {truncated});
    return success();
  }

private:
  mlir::Value maybeCast(mlir::Value operand, PatternRewriter &rewriter) const {
    mlir::Type type = operand.getType();
    if (!isa<Float16Type>(type))
      return operand;

    return rewriter.create<LLVM::FPExtOp>(
        operand.getLoc(), Float32Type::get(rewriter.getContext()), operand);
  }

  mlir::Type getFunctionType(mlir::Type resultType, ValueRange operands) const {
    SmallVector<mlir::Type> operandTypes(operands.getTypes());
    return LLVM::LLVMFunctionType::get(resultType, operandTypes);
  }

  StringRef getFunctionName(mlir::Type type) const {
    if (isa<Float32Type>(type))
      return f32Func;
    if (isa<Float64Type>(type))
      return f64Func;
    return "";
  }

  LLVM::LLVMFuncOp appendOrGetFuncOp(StringRef funcName, mlir::Type funcType,
                                     Operation *op) const {
    using LLVM::LLVMFuncOp;

    auto funcAttr = StringAttr::get(op->getContext(), funcName);
    Operation *funcOp = SymbolTable::lookupNearestSymbolFrom(op, funcAttr);
    if (funcOp)
      return cast<LLVMFuncOp>(*funcOp);

    mlir::OpBuilder b(op->getParentOfType<LLVMFuncOp>());
    return b.create<LLVMFuncOp>(op->getLoc(), funcName, funcType);
  }

  const std::string f32Func;
  const std::string f64Func;
};

struct TanhOpLowering : public ConvertOpToLLVMPattern<math::TanhOp> {
public:
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(math::TanhOp op, typename math::TanhOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto asmDialectAttr = LLVM::AsmDialectAttr::get(rewriter.getContext(),
                                                    LLVM::AsmDialect::AD_ATT);

    rewriter.replaceOpWithNewOp<LLVM::InlineAsmOp>(
        op,
        /*resultTypes=*/getTypeConverter()->convertType(op.getType()),
        /*operands=*/op->getOperands(),
        /*asm_string=*/"tanh.approx.f32 $0, $1;",
        /*constraints=*/"=f,f",
        /*has_side_effects=*/false,
        /*is_align_stack=*/false,
        /*asm_dialect=*/asmDialectAttr,
        /*operand_attrs=*/ArrayAttr());
    return success();
  }
};

void populateOptionalGpuToNVVMExtConversionPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns) {

  patterns.add<OpToFuncCallLowering<arith::MaxNumFOp>>(converter, "__nv_fmaxf",
                                                       "__nv_fmax");
  patterns.add<OpToFuncCallLowering<arith::MinNumFOp>>(converter, "__nv_fminf",
                                                       "__nv_fmin");
}

void populateTempGpuToNVVMExtConversionPatterns(LLVMTypeConverter &converter,
                                                RewritePatternSet &patterns) {

  patterns.add<OpToFuncCallLowering<arith::MaximumFOp>>(converter, "__nv_fmaxf",
                                                        "__nv_fmax");
  patterns.add<OpToFuncCallLowering<arith::MinimumFOp>>(converter, "__nv_fminf",
                                                        "__nv_fmin");
}

static IntegerAttr wrapNumericMemorySpace(MLIRContext *ctx, unsigned space) {
  return IntegerAttr::get(IntegerType::get(ctx, 64), space);
}

/// A function that maps a MemorySpace enum to a target-specific integer value.
using MemorySpaceMapping =
    std::function<unsigned(gpu::AddressSpace gpuAddressSpace)>;
void populateGpuMemorySpaceAttributeConversions(
    TypeConverter &typeConverter, const MemorySpaceMapping &mapping) {
  typeConverter.addTypeAttributeConversion(
      [mapping](BaseMemRefType type, gpu::AddressSpaceAttr memorySpaceAttr) {
        gpu::AddressSpace memorySpace = memorySpaceAttr.getValue();
        unsigned addressSpace = mapping(memorySpace);
        return wrapNumericMemorySpace(memorySpaceAttr.getContext(),
                                      addressSpace);
      });
}

bool compare_nvgpu_arch_lt(const std::string &lhs, const std::string &rhs) {
  if (lhs.size() < 4 || rhs.size() < 4)
    return false;

  auto larch = std::stoi(lhs.substr(3));
  auto rarch = std::stoi(rhs.substr(3));
  return larch < rarch;
}

// Note: this pass is an externsion pass of upstream pass:
// https://github.com/llvm/llvm-project/blob/main/mlir/lib/Conversion/GPUToNVVM/LowerGpuOpsToNVVMOps.cpp
struct GPUToNVVMExtPass : public GPUToNVVMExtBase<GPUToNVVMExtPass> {
  GPUToNVVMExtPass() = default;
  GPUToNVVMExtPass(bool useBarePtrCallConv, unsigned indexBitwidth,
                   const std::string &gpuArch) {
    this->useBarePtrCallConv = useBarePtrCallConv;
    this->indexBitwidth = indexBitwidth;
    this->gpuArch = gpuArch;
  }

  void runOnOperation() override {
    gpu::GPUModuleOp m = getOperation();

    // Request C wrapper emission.
    for (auto func : m.getOps<func::FuncOp>()) {
      func->setAttr(LLVM::LLVMDialect::getEmitCWrapperAttrName(),
                    UnitAttr::get(&getContext()));
    }

    // Customize the bitwidth used for the device side index computations.
    LowerToLLVMOptions options(
        m.getContext(),
        DataLayout(cast<DataLayoutOpInterface>(m.getOperation())));
    if (this->useBarePtrCallConv)
      options.useBarePtrCallConv = true;
    if (indexBitwidth != kDeriveIndexBitwidthFromDataLayout)
      options.overrideIndexBitwidth(indexBitwidth);

    // Apply in-dialect lowering. In-dialect lowering will replace
    // ops which need to be lowered further, which is not supported by a
    // single conversion pass.
    {
      RewritePatternSet patterns(m.getContext());
      populateGpuRewritePatterns(patterns);
      if (failed(applyPatternsAndFoldGreedily(m, std::move(patterns))))
        return signalPassFailure();
    }
    {
      RewritePatternSet patterns(m.getContext());
      populateMathAlgebraicSimplificationPatterns(patterns);
      (void)applyPatternsAndFoldGreedily(m, std::move(patterns));
    }

    LLVMTypeConverter converter(m.getContext(), options);
    // NVVM uses alloca in the default address space to represent private
    // memory allocations, so drop private annotations. NVVM uses address
    // space 3 for shared memory. NVVM uses the default address space to
    // represent global memory.
    populateGpuMemorySpaceAttributeConversions(
        converter, [](gpu::AddressSpace space) -> unsigned {
          switch (space) {
          case gpu::AddressSpace::Global:
            return static_cast<unsigned>(
                NVVM::NVVMMemorySpace::kGlobalMemorySpace);
          case gpu::AddressSpace::Workgroup:
            return static_cast<unsigned>(
                NVVM::NVVMMemorySpace::kSharedMemorySpace);
          case gpu::AddressSpace::Private:
            return 0;
          }
          llvm_unreachable("unknown address space enum value");
          return 0;
        });
    // Lowering for MMAMatrixType.
    converter.addConversion([&](gpu::MMAMatrixType type) -> Type {
      return convertMMAToLLVMType(type);
    });
    RewritePatternSet llvmPatterns(m.getContext());

    arith::populateArithToLLVMConversionPatterns(converter, llvmPatterns);
    cf::populateControlFlowToLLVMConversionPatterns(converter, llvmPatterns);
    populateFuncToLLVMConversionPatterns(converter, llvmPatterns);
    populateFinalizeMemRefToLLVMConversionPatterns(converter, llvmPatterns);
    populateGpuToNVVMConversionPatterns(converter, llvmPatterns);
    populateGpuWMMAToNVVMConversionPatterns(converter, llvmPatterns);
#if 0
    // FIXME: enable if gpu arch >= sm_75
    llvmPatterns.add<TanhOpLowering>(converter, 10);
#endif
    // our extension fixing
    // populateOptionalGpuToNVVMExtConversionPatterns(converter, llvmPatterns);

    // TODO: remove this rewrite pattern while upgrading llvm higher than
    // `ee54c86ef`
    if (compare_nvgpu_arch_lt(this->gpuArch, "sm_80")) {
      populateTempGpuToNVVMExtConversionPatterns(converter, llvmPatterns);
    }

    LLVMConversionTarget target(getContext());
    configureGpuToNVVMConversionLegality(target);
    FrozenRewritePatternSet frozenLLVMPatterns(std::move(llvmPatterns));
    if (failed(applyPartialConversion(m, target, frozenLLVMPatterns)))
      signalPassFailure();

    // TODO: retrieve attribute from memref arg when convert func to llvm
    m.walk([&](LLVM::LLVMFuncOp func) {
      for (auto &&iter : llvm::enumerate(func.getArguments())) {
        if (llvm::isa<LLVM::LLVMPointerType>(iter.value().getType())) {
          func.setArgAttr(iter.index(), LLVMDialect::getNoAliasAttrName(),
                          UnitAttr::get(m->getContext()));
        }
      }
    });
  }
};

} // anonymous namespace

std::unique_ptr<OperationPass<gpu::GPUModuleOp>>
mlir::createGPUToNVVMExtPass(bool useBarePtrCallConv, unsigned indexBitwidth,
                             const std::string &gpuArch) {
  return std::make_unique<GPUToNVVMExtPass>(useBarePtrCallConv, indexBitwidth,
                                            gpuArch);
}
