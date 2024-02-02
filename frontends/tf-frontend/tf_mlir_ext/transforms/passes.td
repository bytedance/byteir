//===- passes.td - Transforms pass definition file -------*--- tablegen -*-===//
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


#ifndef TF_MLIR_EXT_PASSES
#define TF_MLIR_EXT_PASSES

include "mlir/Pass/PassBase.td"

def ConstantFolding : Pass<"constant-folding", "func::FuncOp"> {
  let summary = "extent constant folding for tf dialect ops";
  let constructor = "mlir::tfext::createConstantFoldingPass()";
}

def ProcessDynamicStitchAsStatic : Pass<"process-dynamic-stitch-as-static", "func::FuncOp"> {
  let summary = "Convert dynamic partition and dynamic stitch related subgraph to a "
                 "equivalated one with more compulation";

  let constructor = "mlir::tfext::createProcessDynamicStitchAsStaticPass()";
}

def RemoveControlFlow : Pass<"remove-control-flow", "func::FuncOp"> {
  let summary = "remove control flow";

  let constructor = "mlir::tfext::createRemoveControlFlowPass()";
}

def RewriteToIf : Pass<"rewrite-to-if", "ModuleOp"> {
  let summary = "rewrite merge/switch to TF.If";
  let constructor = "mlir::tfext::createRewriteToIfPass()";
}

def RewriteToCustomCallOps: Pass<"rewrite-to-custom-call", "ModuleOp"> {
  let summary = "convert some ops to mhlo custom_call";
  let constructor = "mlir::tfext::createRewriteToCustomCallOpsPass()";
  let options = [
    ListOption<"ops", "ops", "std::string",
               "Operations to be wrapped",
               "llvm::cl::ZeroOrMore">,
    Option<"keepBody", "keep-body", "bool", /*default=*/"false",
           "keep func body">,	
  ];
  let dependentDialects = [
    "mlir::TF::TensorFlowDialect",
    "mlir::mhlo::MhloDialect",
  ];
}

def MhloLegalizeTfExt: Pass<"mhlo-legalize-tf-ext", "func::FuncOp"> {
  let summary = "Legalized ops that failed to be lowered to mhlo in xla-legalize-tf pass";
  let constructor = "mlir::tfext::createMhloLegalizeTfExtPass()";
  let dependentDialects = [
    "mlir::TF::TensorFlowDialect",
    "mlir::mhlo::MhloDialect",
    "mlir::tensor::TensorDialect",
  ];
}

def FuseTFOps : Pass<"fuse-tf-ops", "func::FuncOp"> {
  let summary = "Fuse TF Ops, ex. SpaceToBatch + Conv3D + BatchToSpace to Dilated Conv3D";
  let constructor = "mlir::tfext::createFuseTFOpsPass()";
  let dependentDialects = [
    "mlir::TF::TensorFlowDialect",
  ];
}


def ReshapeMovedownString : Pass<"host-string-graph-refine", "func::FuncOp"> {
  let summary = "Host graphs that related to string refine pass";
  let constructor = "mlir::tfext::createReshapeMovedownStringPass()";
  let dependentDialects = [
    "mlir::TF::TensorFlowDialect",
  ];
}

def TfFallbackToCustomCall: Pass<"tf-fallback-to-custom-call", "func::FuncOp"> {
  let summary = "convert ops remaining in tf dialect to custom call";
  let constructor = "mlir::tfext::createTfFallbackToCustomCallPass()";
  let dependentDialects = [
    "mlir::TF::TensorFlowDialect",
    "mlir::mhlo::MhloDialect",
    "mlir::ace::AceDialect",
  ];
}

def RewriteFuncAttrToByteIR: Pass<"rewrite-func-attr-to-byteir", "func::FuncOp"> {
  let summary = "rewrite function attributes to byteir function attributes";
  let constructor = "mlir::tfext::createRewriteFuncAttrToByteIRPass()";
  let dependentDialects = [];
}

def TFSwitchMergeToIf : Pass<"tf-switch-merge-to-if", "func::FuncOp"> {
  let summary = "transform merge and switch to TF.If";
  let constructor = "mlir::tfext::createTFSwitchMergeToIfPass()";
}

def RemoveCstrReshapable : Pass<"remove-cstr-reshapable", "func::FuncOp"> {
  let summary = "replace cstr_reshapable to true";
  let constructor = "mlir::tfext::createRemoveCstrReshapablePass()";
}

#endif // TF_MLIR_EXT_PASSES
