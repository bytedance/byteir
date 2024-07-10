//===- Passes.td - Transforms pass definition file -------*--- tablegen -*-===//
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


#ifndef BYTEIR_TRANSFORMS_PASSES
#define BYTEIR_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"
include "mlir/Rewrite/PassUtil.td"

//===----------------------------------------------------------------------===//
// Anchored Pipeline
//===----------------------------------------------------------------------===//

def AnchoredPipeline : Pass<"anchored-pipeline"> {
  let summary = "Apply a given pipeline when an Operation having anchorAttr";
  let constructor = "mlir::createAnchoredPipelinePass()";
  let options = [
    Option<"anchorAttr", "anchor-attr", "std::string",
           /*default=*/"",
           "An optional Unit attribute anchoring on target functions.">,
  ];
}

//===----------------------------------------------------------------------===//
// ApplyPDLPatterns
//===----------------------------------------------------------------------===//

def ApplyPDLPatterns : Pass<"apply-pdl-patterns"> {
  let summary = "Apply pdl patterns on given module";
  let constructor = "mlir::createApplyPDLPatternsPass()";
  let options = [
    Option<"pdlFile", "pdl-file", "std::string",
           /*default=*/"", "To specify the pdl module file.">,
  ];
  let dependentDialects = [
    "::mlir::pdl::PDLDialect",
    "::mlir::pdl_interp::PDLInterpDialect"
  ];
}

//===----------------------------------------------------------------------===//
// TryCatch Module Pipeline
//===----------------------------------------------------------------------===//

def TryCatchModulePipeline : Pass<"try-catch-module-pipeline", "ModuleOp"> {
  let summary =
      "Apply a given pipeline with try-catch method in loop when there are "
      "some limitation";
  let constructor = "mlir::createTryCatchModulePipelinePass()";
  let options = [
    Option<"anchorAttr", "anchor-attr", "std::string",
           /*default=*/"",
           "An optional Unit attribute anchoring on target functions.">,
    Option<"maxTries", "max-tries", "int64_t",
           /*default=*/"10",
           "The limit times of the try-catch loop">
  ];
}

//===----------------------------------------------------------------------===//
// CanonicalizeExt
//===----------------------------------------------------------------------===//

def CanonicalizeExt : Pass<"canonicalize-ext"> {
  let summary = "Perform as an extension of upstream canonicalize pass.";
  let description = [{
    This pass performs as an extension of upstream canonicalize pass, with 
    additional canonicalization pattens, and dce especially for mhlo.custom_call. 
  }];
  let constructor = "mlir::createCanonicalizeExtPass()";
  let options = [
    Option<"topDownProcessingEnabled", "top-down", "bool",
           /*default=*/"true",
           "Seed the worklist in general top-down order">,
    Option<"enableRegionSimplification", "region-simplify", "bool",
           /*default=*/"true",
           "Seed the worklist in general top-down order">,
    Option<"maxIterations", "max-iterations", "int64_t",
           /*default=*/"10",
           "Seed the worklist in general top-down order">,
    Option<"foldLimit", "fold-limit", "int64_t",
           /*default=*/"0",
           "Fold constants which size less equal than foldLimit, -1 means no limit">,
    Option<"blindFold", "blind-fold", "bool",
           /*default=*/"false",
           "Fold constants even if it may finally bloat the compilation result">
  ] # RewritePassUtils.options;
}

//===----------------------------------------------------------------------===//
// CMAE (Common Memory Access Elimination)
//===----------------------------------------------------------------------===//

def CMAE : Pass<"cmae", "func::FuncOp"> {
  let summary = "Eliminate common memory access within a block";
  let constructor = "mlir::createCMAEPass()";
  let options = [
    Option<"skipAttr", "skip-attr", "std::string",
            /*default=*/"",
            "An optional attribute to skip this pass for specific function."
            "Use it for functions with non-restricted args.">
  ];
}

//===----------------------------------------------------------------------===//
// Condition Canonicalization
//===----------------------------------------------------------------------===//

def CondCanonicalize : Pass<"cond-canonicalize", "func::FuncOp"> {
  let summary = "canonicalization based on condition";
  let constructor = "mlir::createCondCanonicalizePass()";
}

//===----------------------------------------------------------------------===//
// CollectFunc
//===----------------------------------------------------------------------===//

def CollectFunc : Pass<"collect-func", "ModuleOp"> {
  let summary = "collect all function based on anchor";
  let constructor = "mlir::createCollectFuncPass()";
  let options = [
    Option<"anchorAttr", "anchor-attr", "std::string",
            /*default=*/"",
            "An optional Unit attribute anchoring on target functions.">
  ];
}

//===----------------------------------------------------------------------===//
// GenericDeviceConfig
//===----------------------------------------------------------------------===//

def GenericDeviceConfig : Pass<"generic-device-config", "mlir::func::FuncOp"> {
  let summary = "Generate Generic Device configuration";
  let constructor = "mlir::createGenericDeviceConfigPass()";
  let options = [
    Option<"anchorAttr", "anchor-attr", "std::string",
            /*default=*/"",
            "An attribute anchoring on target functions.">,
    Option<"computeName", "compute-name", "std::string",
            /*default=*/"",
            "An attribute setting compute name.">,
  ];
}

//===----------------------------------------------------------------------===//
// GraphCanonicalizeExt
//===----------------------------------------------------------------------===//

def GraphCanonicalize : Pass<"graph-canonicalize"> {
  let summary = "Perform a graph-level canonicalize pass.";
  let description = [{
    This pass performs canonicalization for graph-level only. 
    Graph-level includes mhlo, shape and function only.
  }];
  let constructor = "mlir::createGraphCanonicalizePass()";
  let options = [
    Option<"topDownProcessingEnabled", "top-down", "bool",
           /*default=*/"true",
           "Seed the worklist in general top-down order">,
    Option<"enableRegionSimplification", "region-simplify", "bool",
           /*default=*/"true",
           "Seed the worklist in general top-down order">,
    Option<"maxIterations", "max-iterations", "int64_t",
           /*default=*/"10",
           "Seed the worklist in general top-down order">,
    Option<"foldLimit", "fold-limit", "int64_t",
           /*default=*/"0",
           "Fold constants which size less equal than foldLimit, -1 means no limit">,
    Option<"blindFold", "blind-fold", "bool",
           /*default=*/"false",
           "Fold constants even if it may finally bloat the compilation result">
  ] # RewritePassUtils.options;
}

//===----------------------------------------------------------------------===//
// FuncTag
//===----------------------------------------------------------------------===//

def FuncTag : Pass<"func-tag", "ModuleOp"> {
  let summary = "Attach Tag based on func name or attr";
  let constructor = "mlir::createFuncTagPass()";
  let options = [
    Option<"anchorAttr", "anchor-attr", "std::string",
            /*default=*/"",
            "An optional Unit attribute anchoring on target functions.">,
    Option<"attachAttr", "attach-attr", "std::string",
            /*default=*/"",
            "An optional attribute attaching on target functions: <attr_name>:<type>:<value>">,
    Option<"funcName", "func-name", "std::string",
            /*default=*/"",
            "An optional name to specify target functions.">
  ];
}

//===----------------------------------------------------------------------===//
// RemoveFuncTag
//===----------------------------------------------------------------------===//

def RemoveFuncTag : Pass<"remove-func-tag", "ModuleOp"> {
  let summary = "Remove Tag based on func name and attr";
  let constructor = "mlir::createRemoveFuncTagPass()";
  let options = [
    Option<"attrName", "attr-name", "std::string",
            /*default=*/"",
            "An optional attribute removing from target functions:">,
    Option<"funcName", "func-name", "std::string",
            /*default=*/"",
            "An optional name to specify target functions.">
  ];
}

//===----------------------------------------------------------------------===//
// GraphClusteringByDevice
//===----------------------------------------------------------------------===//
// This pass is currently a naive one and only works correct on tensor level Dialects
def GraphClusteringByDevice : Pass<"graph-clustering-by-device", "ModuleOp"> {
  let summary = "Clustering each function by device.";
  let constructor = "mlir::createGraphClusteringByDevicePass()";
  let options = [
    Option<"attrName", "attr-name", "std::string", /*default=*/"\"device\"",
           "The attr name used for clustering.">,
    Option<"device", "device", "std::string", /*default=*/"\"test\"",
           "The concrete device name for newly created device functions.">,
    Option<"deviceAnchorName", "device-anchor-name", "std::string", /*default=*/"\"test_device\"",
           "An anchor used to trigger specific pipeline.">,
    Option<"dupNonSplat", "dup-non-splat", "bool", /*default=*/"false",
           "duplicate a non-splat const if it is used by both host and device side.">,
    Option<"dupOutputs", "dup-outputs", "bool", /*default=*/"false",
           "duplicate a return value in the internal function if it is returned multiple times in main.">,
    Option<"clusterAlgo", "cluster-algo", "::mlir::GraphClusteringAlgo", /*default=*/"::mlir::GraphClusteringAlgo::kFallback",
           "The clustering algorithm",
           [{llvm::cl::values(
              clEnumValN(mlir::GraphClusteringAlgo::kFallback, "Fallback", "Fallback clustering algorithm"),
              clEnumValN(mlir::GraphClusteringAlgo::kTopDown, "TopDown", "Create and merge device cluster progressively from top to bottom"),
              clEnumValN(mlir::GraphClusteringAlgo::kBottomUp, "BottomUp", "Create and merge device cluster progressively from bottom to top"),
              clEnumValN(mlir::GraphClusteringAlgo::kGreedy, "Greedy", "Choose largest subgraph from BottomUp or TopDown strategy")
            )}]>,
    Option<"enableMultiGraph", "enable-multi-graph", "bool", /*default=*/"false",
           "create multi device func as possible.">,
  ];
}

//===----------------------------------------------------------------------===//
// LoopTag
//===----------------------------------------------------------------------===//

def LoopTag : Pass<"loop-tag", "func::FuncOp"> {
  let summary = "Attach Tag based on func name";
  let constructor = "mlir::createLoopTagPass()";
  let options = [
    Option<"anchorAttr", "anchor-attr", "std::string",
            /*default=*/"",
            "An optional Unit attribute anchoring on target functions.">,
    Option<"attachAttr", "attach-attr", "std::string",
            /*default=*/"",
            "An optional attribute attaching on target functions: <attr_name>:<type>:<value>">,
    Option<"depth", "depth", "int", /*default=*/"1",
           "Use this depth factor to specify loop, -1 is last">,
    Option<"loopType", "loop-type", "std::string",
            /*default=*/"\"scf.for\"",
            "An optional name to specify loop type.">
  ];
}

//===----------------------------------------------------------------------===//
// Memory planning
//===----------------------------------------------------------------------===//
def MemoryPlanning: InterfacePass<"memory-planning", "mlir::FunctionOpInterface"> {
  let summary = "Pass to perform static memory planning";
  let constructor = "mlir::createMemoryPlanningPass()";
  let dependentDialects = [
    "memref::MemRefDialect"
  ];
  let options = [
    Option<"alignment", "alignment", "size_t", /*default=*/"1",
           "Memory address alignment, must be power of 2">,
    Option<"alloca", "alloca", "bool", /*default=*/"false",
           "Whether to planning on memref.alloca ops">,
    Option<"memSpace", "mem-space", "size_t", /*default=*/"0",
           "Memory space considered in planning. 0 indicates all spaces.">,
  ];
}

//===----------------------------------------------------------------------===//
// MemrefOpToStdCall
//===----------------------------------------------------------------------===//

def RewriteOpToStdCall : Pass<"rewrite-op-to-std-call", "ModuleOp"> {
  let summary = "Rewrite any op of memref to external std call.";
  let constructor = "mlir::createRewriteOpToStdCallPass()";
  let dependentDialects = [
    "mlir::memref::MemRefDialect",
    "mlir::func::FuncDialect"
  ];
  let options = [
    ListOption<"callTable", "call-table", "std::string",
               "Replace op with call function name: call-table=<op name>:<call name>,...",
               "llvm::cl::ZeroOrMore">
  ];
}

//===----------------------------------------------------------------------===//
// OneShotBufferize
//===----------------------------------------------------------------------===//

def OneShotBufferize : Pass<"byteir-one-shot-bufferize", "ModuleOp"> {
  let summary = "One shot bufferization pass.";
  let constructor = "byteir::createOneShotBufferizePass()";
}

//===----------------------------------------------------------------------===//
// LoopUnroll
//===----------------------------------------------------------------------===//

def LoopUnroll : Pass<"unroll", "mlir::func::FuncOp"> {
  let summary = "Unroll scf/affine loops for anchored loop or satisfying depth";
  let constructor = "mlir::createByteIRLoopUnrollPass()";
  let options = [
    Option<"unrollFactor", "unroll-factor", "unsigned", /*default=*/"2",
           "Use this unroll factor for all loops being unrolled">,
    Option<"unrollUpToFactor", "unroll-up-to-factor", "bool", /*default=*/"false", 
           "Allow unrolling up to the factor specified">,
    Option<"unrollFull", "unroll-full", "bool", /*default=*/"false",
           "Fully unroll loops">,
    Option<"depth", "depth", "int", /*default=*/"-1",
           "Loop depth, a negative number implies disabling">,
    Option<"unrollAll", "unroll-all", "bool", /*default=*/"false",
           "Unroll all loops">,
    Option<"annotateIdx", "annotate-idx", "bool", /*default=*/"false",
           "Annotate ops with step and total steps when unrolling. Annotation works with multiple unrollings if loop structures remains unchanged.">,
  ];
}

//===----------------------------------------------------------------------===//
// InsertUniqueId
//===----------------------------------------------------------------------===//

def InsertUniqueId : Pass<"insert-unique-id", "ModuleOp"> {
  let summary = "Insert unique ids (operation name + id) for operations.";
  let constructor = "mlir::createByteIRInsertUniqueIdPass()";
  let options = [
    Option<"anchorAttr", "anchor-attr", "std::string",
            /*default=*/"",
            "An optional attribute anchoring on target functions.">,
    Option<"eraseId", "erase-id", "bool",
            /*default=*/"false",
            "Remove unique ids.">
  ];
}

//===----------------------------------------------------------------------===//
// Remove func body
//===----------------------------------------------------------------------===//
def RemoveFuncBody: Pass<"remove-func-body", "mlir::func::FuncOp"> {
  let summary = "Remove func body for func having specific attr";
  let constructor = "mlir::createRemoveFuncBodyPass()";
  let options = [
    Option<"anchorAttr", "anchor-attr", "std::string",
            /*default=*/"",
            "An optional attribute anchoring on target functions.">,
    Option<"disableForcePrivate", "disable-force-private", "bool", /*default=*/"false",
           "Disable force public functon to private function"> 
  ];
}

//===----------------------------------------------------------------------===//
// SetArgShape
//===----------------------------------------------------------------------===//

def SetArgShape: Pass<"set-arg-shape", "ModuleOp"> {
  let summary = "Set arg shape for the function specified.";
  let constructor = "mlir::createSetArgShapePass()";
  let options = [
    Option<"dim", "dim", "int", /*default=*/"0",
           "The dim to be set in all qulified args.">,
    Option<"size", "size", "int", 
            /*default=*/"1",
           "The size to be used in all qulified args for the specified dim.">,
    Option<"entryFuncName", "entry-func-name", "std::string",
            /*default=*/"\"main\"",
            "Used to specify the entry-function.">,
    Option<"argAttrName", "arg-attr-name", "std::string",
            /*default=*/"",
            "Specify the arg(s) to be acted on if it contains the attribute name.">
  ];
}

//===----------------------------------------------------------------------===//
// SetSpace Passes
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// SetAllSpace (mainly use for all in the same device)
// This pass is soft-deprecated.
// Use SetOpSpace + SetArgSpace instead
//===----------------------------------------------------------------------===//

def SetAllSpace: Pass<"set-all-space", "ModuleOp"> {
  let summary = "Set all memSpace including intermediate for the entry function specified.";
  let constructor = "mlir::createSetAllSpacePass()";
  let options = [
    Option<"entryFunc", "entry-func", "std::string",
            /*default=*/"\"main\"",
            "Used to specify the entry-function.">,
    Option<"space", "space", "std::string",
            /*default=*/"",
            "Specified space for all value">,
  ];
  let dependentDialects = [
    "mlir::memref::MemRefDialect",
  ];
}

//===----------------------------------------------------------------------===//
// SetArgSpace: Set argument spaces
//===----------------------------------------------------------------------===//

def SetArgSpace: Pass<"set-arg-space", "ModuleOp"> {
  let summary = "Set arg memSpace for the entry function specified.";
  let constructor = "mlir::createSetArgSpacePass()";
  let options = [
    Option<"entryFunc", "entry-func", "std::string",
            /*default=*/"\"main\"",
            "Used to specify the entry-function.">,
    Option<"allSpace", "all-space", "std::string",
            /*default=*/"",
            "Specified space for all args">,
    Option<"allowArgWritable", "allow-arg-writable",  "bool", /*default=*/"false",
            "allow arg (not return) be writable">,
    Option<"autoDeduce", "auto-deduce", "bool", /*default=*/"false",
            "whether to deduce arg space">,
  ];
  let dependentDialects = [
    "mlir::memref::MemRefDialect",
  ];
}

//===----------------------------------------------------------------------===//
// SetOpSpace: Set Op Space, only attach attribute
//===----------------------------------------------------------------------===//

def SetOpSpace: Pass<"set-op-space", "func::FuncOp"> {
  let summary = "Set op device attribute for the entry function specified.";
  let constructor = "mlir::createSetOpSpacePass()";
  let options = [
    Option<"entryFunc", "entry-func", "std::string",
            /*default=*/"\"main\"",
            "Used to specify the entry-function.">,
    Option<"space", "space", "std::string",
            /*default=*/"",
            "Specified space for all ops">,
  ];
}

//===----------------------------------------------------------------------===//
// Shape func outlining
//===----------------------------------------------------------------------===//

def ShapeFuncOutlining : Pass<"shape-func-outlining", "ModuleOp"> {
  let summary = "Outline all shape computation in entry func to a shape func";
  let constructor = "mlir::createShapeFuncOutliningPass()";
  let options = [
    Option<"entryFuncName", "entry-func-name", "std::string",
            /*default=*/"\"main\"",
            "Used to specify the entry-function.">
  ];
}

#endif // BYTEIR_TRANSFORMS_PASSES
