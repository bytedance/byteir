//===- OFPasses.td --------------------------------------------------------===//
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

#ifndef ONNX_FRONTEND_SRC_CONVERSION_OFPASSES_TD
#define ONNX_FRONTEND_SRC_CONVERSION_OFPASSES_TD

include "mlir/Pass/PassBase.td"

def OFCanonicalizer : Pass<"of-canonicalize", "mlir::ModuleOp"> {
  let summary = "canonicalization and func dialect type conversion";
  let constructor = "onnx_frontend::createOFCanonicalizerPass()";
  let options = [
    Option<"testConvergence", "test-convergence", "bool", /*default=*/"false",
           "Test only: Fail pass on non-convergence to detect cyclic pattern">
  ];
}

def OFRewriteToCustomCall: Pass<"rewrite-to-custom-call", "mlir::func::FuncOp"> {
  let summary = "convert some ops to mhlo.custom_call";
  let constructor = "onnx_frontend::createOFRewriteToCustomCallPass()";
  let options = [
    ListOption<"customCallOps", "ops", "std::string",
               "Operations to be rewritten to mhlo.custom_call",
               "llvm::cl::ZeroOrMore">,
  ];
  let dependentDialects = [
    "mlir::mhlo::MhloDialect",
  ];
}

def OFModifyEntryPoint: Pass<"of-modify-entry-point", "mlir::ModuleOp"> {
  let summary = "modify the attributes of entry point function";
  let constructor = "onnx_frontend::createOFModifyEntryPointPass()";
}

#endif // ONNX_FRONTEND_SRC_CONVERSION_OFPASSES_TD
