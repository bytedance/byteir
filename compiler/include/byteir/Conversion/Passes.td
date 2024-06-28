//===- Passes.td - Conversion pass definition file -------*--- tablegen -*-===//
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

#ifndef BYTEIR_CONVERSION_PASSES
#define BYTEIR_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// GPUToNVVM
//===----------------------------------------------------------------------===//

def GPUToNVVMExt : Pass<"gpu-to-nvvm-ext", "gpu::GPUModuleOp"> {
  let summary = "Generate NVVM operations for gpu operations";
  let constructor = "mlir::createGPUToNVVMExtPass()";
  let dependentDialects = [
    "NVVM::NVVMDialect",
    "memref::MemRefDialect",
    "cf::ControlFlowDialect"
  ];
  let options = [
    Option<"useBarePtrCallConv", "use-bare-ptr-memref-call-conv", "bool",
           /*default=*/"false",
           "Replace memref arguments in GPU functions with bare pointers."
           "All memrefs must have static shape">,
    Option<"indexBitwidth", "index-bitwidth", "unsigned",
           /*default=kDeriveIndexBitwidthFromDataLayout*/"0",
           "Bitwidth of the index type, 0 to use size of machine word">,
    Option<"gpuArch", "gpu-arch", "std::string", /*default*/ "\"sm_80\"",
           "Specificy the target nvgpu arch">
  ];
}

//===----------------------------------------------------------------------===//
// ToLinalg
//===----------------------------------------------------------------------===//

def HloFusionToLinalg : Pass<"hlo-fusion-to-linalg", "func::FuncOp"> {
  let summary = "Convert from HLO fusion to Linalg dialect.";
  let constructor = "mlir::createHloFusionToLinalgPass()";
  let dependentDialects = [
    "mlir::linalg::LinalgDialect",
    "mlir::scf::SCFDialect",
  ];
  let options = [
    Option<"anchorTag", "anchor-tag", "std::string",
            /*default=*/"",
            "Optional unitAttr tag be applied the fused ops">,
    Option<"enablePrimitiveOps", "enable-primitive-ops", "bool",
            /*default=*/"false",
            "Lower to primitive Linalg ops (map, reduce and "
            "transpose) when possible, instead of linalg.generic">
  ];
}

def UnrealizedCastToLinalg : Pass<"unrealized-cast-to-linalg", "func::FuncOp"> {
  let summary = "Convert UnrealizedCast op generated from hlo lowering to Linalg";
  let constructor = "mlir::createUnrealizedCastToLinalgPass()";
  let dependentDialects = [
    "mlir::linalg::LinalgDialect",
  ];
}

def TensorToLinalg : Pass<"tensor-to-linalg", "func::FuncOp"> {
  let summary = "Convert Tensor ops to Linalg";
  let constructor = "mlir::createTensorToLinalgPass()";
  let dependentDialects = [
    "mlir::linalg::LinalgDialect",
  ];
}

def LinalgExtToLinalg : Pass<"linalg-ext-to-linalg", "func::FuncOp"> {
  let summary = "Convert LinalgExt ops to Linalg";
  let constructor = "mlir::createLinalgExtToLinalgPass()";
}

//===----------------------------------------------------------------------===//
// ToGPU
//===----------------------------------------------------------------------===//

def CoalescedForToGPULaunch : Pass<"coalesced-for-to-gpu", "func::FuncOp"> {
  let summary = "Convert from a coalesced affine loop to gpu launch";
  let constructor = "mlir::createCoalescedForToGPULaunchPass()";
  let dependentDialects = [
    "mlir::gpu::GPUDialect",
    "mlir::scf::SCFDialect",
  ];
  let options = [
      Option<"blockSize", "bsize", "int64_t", /*default=*/"32",
             "block size for GPU">,
  ];
}

def ConvertFuncToGPU : Pass<"convert-func-to-gpu", "ModuleOp"> {
  let summary = "Cconvert func to GPU func";
  let constructor = "mlir::createConvertFuncToGPUPass()";
  let dependentDialects = [
    "mlir::gpu::GPUDialect",
  ];
  let options = [
    ListOption<"blockSizes", "bsizes", "int64_t",
               "block sizes (up to 3 numbers) for GPU",
               "llvm::cl::ZeroOrMore">,
    ListOption<"gridSizes", "gsizes", "int64_t",
               "grid sizes (up to 3 numbers) for GPU",
               "llvm::cl::ZeroOrMore">,
    Option<"moduleName", "module-name", "std::string",
            /*default=*/"\"unified\"",
            "Optional name for GPUModule to put all gpu kernels">,
  ];
}

//===----------------------------------------------------------------------===//
// ToAce
//===----------------------------------------------------------------------===//

def ConvertMhloToAce : Pass<"convert-mhlo-to-ace", "func::FuncOp"> {
  let summary = "Fuse mhlo ops to coarse ace ops.";
  let constructor = "mlir::createConvertMhloToAcePass()";
  let dependentDialects = [
    "mlir::ace::AceDialect",
  ];
}

//===----------------------------------------------------------------------===//
// ToAIT
//===----------------------------------------------------------------------===//

def GenAITConfig : Pass<"gen-ait-config", "func::FuncOp"> {
  let summary = "Generate AIT configuration";
  let constructor = "mlir::createGenAITConfigPass()";
  let options = [
    ListOption<"funcNames", "func-names", "std::string",
               "names of all cat func for AIT backends.">,
    ListOption<"aitLibPaths", "ait-lib-paths", "std::string",
               "paths to all AIT-generated .so libs">,
  ];
}

//===----------------------------------------------------------------------===//
// ToByre
//===----------------------------------------------------------------------===//

def ConvertToByre : Pass<"convert-to-byre", "ModuleOp"> {
  let summary = "Convert op to Byre Op.";
  let constructor = "mlir::createConvertToByrePass()";
  let dependentDialects = [
    "mlir::byre::ByreDialect",
  ];
  let options = [
    Option<"appendArgTypes", "append-arg-types", "bool", /*default=*/"false",
           "append arg types to Byre">,
  ];

}

def ConvertFuncAndCallToByre : Pass<"convert-func-and-call-to-byre", "ModuleOp"> {
  let summary = "Convert func and call to Byre";
  let constructor = "mlir::createConvertFuncAndCallToByrePass()";
  let dependentDialects = [
    "mlir::memref::MemRefDialect",
    "mlir::byre::ByreDialect",
  ];
  let options = [
    Option<"appendArgTypes", "append-arg-types", "bool", /*default=*/"false",
           "append arg types to Byre">,
    Option<"removeDupOutputs", "remove-dup-outputs", "bool", /*default=*/"false",
            "remove dup function outputs to avoid inserting copy">,
  ];
}

//===----------------------------------------------------------------------===//
// ToPTX or CUDA
//===----------------------------------------------------------------------===//

def GenPTXConfig : Pass<"gen-ptx-config", "func::FuncOp"> {
  let summary = "Generate PTX configuration";
  let constructor = "mlir::createGenPTXConfigPass()";
  let options = [
    Option<"useBarePtrCallConv", "use-bare-ptr-memref-call-conv", "bool",
           /*default=*/"false",
           "Replace memref arguments in GPU functions with bare pointers."
           "All memrefs must have static shape">
  ];
}

def CollectGPUKernel : Pass<"collect-gpu-kernel", "ModuleOp"> {
  let summary = "Collect GPU kernels for generating device code";
  let constructor = "mlir::createCollectGPUKernelPass()";
  let options = [
    Option<"moduleName", "module-name", "std::string",
            /*default=*/"\"unified\"",
            "Optional name for GPUModule to put all gpu kernels">,
    Option<"removeHost", "remove-host", "bool",
            /*default=*/"true",
            "Whether to remove host part">,
  ];
}

//===----------------------------------------------------------------------===//
// ToMhlo
//===----------------------------------------------------------------------===//

def ConvertArithToMhlo : Pass<"convert-arith-to-mhlo", "func::FuncOp"> {
  let summary = "Convert Arith op to Mhlo op";
  let constructor = "mlir::createConvertArithToMhloPass()";
  let dependentDialects = ["mlir::mhlo::MhloDialect"];
}

//===----------------------------------------------------------------------===//
// ToLLVM
//===----------------------------------------------------------------------===//

def GenLLVMConfig : Pass<"gen-llvm-config", "func::FuncOp"> {
  let summary = "Generate LLVM configuration";
  let constructor = "mlir::createGenLLVMConfigPass()";
  let options = [
    Option<"fileName", "file-name", "std::string",
           /*default=*/"\"host_kernels.ll\"",
           "To specify where the generated llvm kernel will be writed to">,
  ];
}

def CollectFuncToLLVM : Pass<"collect-func-to-llvm", "mlir::ModuleOp"> {
  let summary =
      "Collect functions to submodule which will be converted to llvmir";
  let constructor = "mlir::createCollectFuncToLLVMPass()";
}

//===----------------------------------------------------------------------===//
// ConvertHloToCat
//===----------------------------------------------------------------------===//

def ConvertHloToCat : Pass<"convert-hlo-to-cat", "func::FuncOp"> {
  let summary = "convert hlo ops to cat op.";
  let constructor = "mlir::createConvertHloToCatPass(/*validCatOps=*/{})";
  let options = [
    ListOption<"validCatOps", "valid-cat-ops", "std::string",
               "List of cat ops could be converted to">,
  ];
  let dependentDialects = [
    "mlir::cat::CatDialect",
  ];
}

//===----------------------------------------------------------------------===//
// FuseMhloToCat
//===----------------------------------------------------------------------===//

def FuseMhloToCat : Pass<"fuse-mhlo-to-cat", "func::FuncOp"> {
  let summary = "Fuse mhlo ops to Cat Op.";
  let constructor = "mlir::createFuseMhloToCatPass()";
  let dependentDialects = [
    "mlir::cat::CatDialect",
  ];
}

//===----------------------------------------------------------------------===//
// ToCat
//===----------------------------------------------------------------------===//

def MhloToCat : Pass<"mhlo-to-cat", "func::FuncOp"> {
  let summary = "Convert mhlo op to Cat Op.";
  let constructor = "mlir::createMhloToCatPass()";
  let dependentDialects = [
    "mlir::cat::CatDialect",
  ];
}

//===----------------------------------------------------------------------===//
// HloToByreTensor
//===----------------------------------------------------------------------===//

def ConvertHloToByreTensor : Pass<"hlo-to-byre-tensor", "func::FuncOp"> {
  let summary = "Convert hlo op to byre tensor op.";
  let constructor = "mlir::createConvertHloToByreTensorPass()";
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::byre::ByreDialect",
    "mlir::tensor::TensorDialect",
    "mlir::shape::ShapeDialect",
  ];

  let options = [
    Option<"appendArgTypes", "append-arg-types", "bool", /*default=*/"false",
           "append arg types to Byre">,
    Option<"enableTF32", "enable-tf32", "bool", /*default=*/"false",
           "enable 1xTF32 on fp32 gemm/bmm">,
  ];
}


//===----------------------------------------------------------------------===//
// FuncToByreTensor
//===----------------------------------------------------------------------===//

def ConvertFuncToByreTensor : Pass<"func-to-byre-tensor", "ModuleOp"> {
  let summary = "Convert func op to byre tensor compute op.";
  let constructor = "mlir::createConvertFuncToByreTensorPass()";
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::byre::ByreDialect",
    "mlir::tensor::TensorDialect",
    "mlir::shape::ShapeDialect",
  ];

  let options = [
    Option<"appendArgTypes", "append-arg-types", "bool", /*default=*/"false",
           "append arg types to Byre">,
  ];
}


//===----------------------------------------------------------------------===//
// FuncToByreTensor
//===----------------------------------------------------------------------===//

def ConvertGPULaunchFuncToByre : Pass<"gpu-launch-func-to-byre"> {
  let summary = "Convert gpu.launch_func op to byre compute op.";
  let constructor = "mlir::createConvertGPULaunchFuncToByrePass()";
  let dependentDialects = [
    "mlir::byre::ByreDialect",
    "mlir::gpu::GPUDialect"
  ];

  let options = [
    Option<"useBarePtrCallConv", "use-bare-ptr-memref-call-conv", "bool",
          /*default=*/"false",
          "Replace memref arguments in GPU functions with bare pointers."
          "All memrefs must have static shape">,
  ];
}

//===----------------------------------------------------------------------===//
// MemrefToByre
//===----------------------------------------------------------------------===//

def ConvertMemrefToByre : Pass<"memref-to-byre", "func::FuncOp"> {
  let summary = "Convert memref op to byre op.";
  let constructor = "mlir::createConvertMemrefToByrePass()";
  let dependentDialects = [
    "mlir::byre::ByreDialect",
    "mlir::memref::MemRefDialect"
  ];
}

//===----------------------------------------------------------------------===//
// FuncToByre
//===----------------------------------------------------------------------===//

// def ConvertFuncToByreTensor : Pass<"func-to-byre", "ModuleOp"> {
//   let summary = "Convert func op to byre compute op.";
//   let constructor = "mlir::createConvertFuncToByrePass()";
//   let dependentDialects = [
//     "mlir::memref::MemRefDialect",
//     "mlir::byre::ByreDialect",
//   ];
//   let options = [
//     Option<"appendArgTypes", "append-arg-types", "bool", /*default=*/"false",
//            "append arg types to Byre">,
//     Option<"removeDupOutputs", "remove-dup-outputs", "bool", /*default=*/"false",
//             "remove dup function outputs to avoid inserting copy">,
//   ];
// }


def MemrefCopyToLinalgPass :
    Pass<"memrefcopy-to-linalg", "ModuleOp"> {
  let summary = "Convert memref.copy to linalg op";
  let constructor = "mlir::createMemrefCopyToLinalgPass()";
  let dependentDialects = [
    "mlir::linalg::LinalgDialect",
    "mlir::memref::MemRefDialect",
  ];

  let options = [
    Option<"anchorTag", "anchor-tag", "std::string",
            /*default=*/"",
            "Optional unitAttr anchored tag to apply this pass">,
    Option<"attachAttr", "attach-attr", "std::string",
            /*default=*/"",
            "An optional unit attribute attaching on target functions: <attr_name>">,
    Option<"outlining", "outlining", "bool",
            /*default=*/"true",
            "Whether to outline the copy op to a new function">,
  ];
}

//===----------------------------------------------------------------------===//
// LcclToByre
//===----------------------------------------------------------------------===//

def ConvertLcclToByre : Pass<"lccl-to-byre", "func::FuncOp"> {
  let summary = "Convert lccl op to byre op.";
  let constructor = "mlir::createConvertLcclToByrePass()";
  let dependentDialects = [
    "mlir::byre::ByreDialect",
    "mlir::lccl::LcclDialect"
  ];
}

#endif // BYTEIR_CONVERSION_PASSES
