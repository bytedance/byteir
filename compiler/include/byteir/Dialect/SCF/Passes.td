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


#ifndef BYTEIR_DIALECT_SCF_PASSES
#define BYTEIR_DIALECT_SCF_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// InsertTrivialSCFLoop
//===----------------------------------------------------------------------===//

def InsertTrivialSCFLoop : Pass<"insert-trivial-scf-loop", "mlir::func::FuncOp"> {
  let summary = "Insert trivial scf Loop for scalar computation";
  let constructor = "mlir::createInsertTrivialSCFLoopPass()";
  let dependentDialects = [
    "scf::SCFDialect"
  ];
  let options = [
    Option<"anchorTag", "anchor-tag", "std::string",
            /*default=*/"",
            "Optional unitAttr anchored tag to apply this pass">
  ];
}

//===----------------------------------------------------------------------===//
// FuseNestedForall
//===----------------------------------------------------------------------===//

def FuseNestedForall : Pass<"fuse-nested-forall", "mlir::func::FuncOp"> {
  let summary = "Fuse nested forall if possible";
  let constructor = "mlir::createFuseNestedForallPass()";
  let dependentDialects = [
    "scf::SCFDialect"
  ];
  let options = [
    Option<"anchorTag", "anchor-tag", "std::string",
            /*default=*/"",
            "Optional unitAttr anchored tag to apply this pass">
  ];
}

//===----------------------------------------------------------------------===//
// ForallCollapsing
//===----------------------------------------------------------------------===//

def ForallCollapsing : Pass<"forall-collapsing", "mlir::func::FuncOp"> {
  let summary = "collapse forall";
  let constructor = "mlir::createForallCollapsingPass()";
  let dependentDialects = [
    "scf::SCFDialect"
  ];
  let options = [
    Option<"anchorTag", "anchor-tag", "std::string",
            /*default=*/"",
            "Optional unitAttr anchored tag to apply this pass">
  ];
}

#endif // BYTEIR_DIALECT_SCF_PASSES