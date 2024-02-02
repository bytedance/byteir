//===- passes.td - Pipeline pass definition file ---------*--- tablegen -*-===//
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

#ifndef TF_MLIR_EXT_PIPELINES_PASSES
#define TF_MLIR_EXT_PIPELINES_PASSES

include "mlir/Pass/PassBase.td"

def CustomizedTfToMhloPipeline : Pass<"customized-tf-to-mhlo", "mlir::ModuleOp"> {
  let summary = "Customized TF dialect to MHLO dialect";
  let constructor = "mlir::tfext::createCustomizedTfToMhloPipelinePass()";
  let dependentDialects = [
    "mlir::TF::TensorFlowDialect",
    "mlir::mhlo::MhloDialect",
    "mlir::chlo::ChloDialect",
    "mlir::stablehlo::StablehloDialect",
    "mlir::shape::ShapeDialect",
    "mlir::ace::AceDialect",
  ];
  let options = [
    ListOption<"customCallOps", "customcall-ops", "std::string",
               "customCall ops: customcall-ops=softmax,argmax,...",
               "llvm::cl::ZeroOrMore">,
    Option<"removeControlFlow", "remove-control-flow", "bool", /*default=*/"false",
           "remove control flow or not">,
    Option<"staticalizeDynamicShape", "staticalize-dynamic-shape", "bool",
          /*default*/"false", "Aggresively and experimentally try to rewrite "
          "the dynamic graph to a equivalent static graph">,
    Option<"stopAfterRewriteCustomCall", "stop-after-rewrite-customcall", "bool", /*default=*/"false",
           "stop after rewrite customcall ops">,
    Option<"setAssumingToBeTrue", "set-assuming-to-be-true", "bool", /*default=*/"true",
           "remove cstr_reshapable,cstr_broadcastable, and set assuming to be true">
  ];
}

#endif // TF_MLIR_EXT_PIPELINES_PASSES
