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

#ifndef BYTEIR_DIALECT_TRANSFORM_PASSES
#define BYTEIR_DIALECT_TRANSFORM_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// TransformDialectInterpreter
//===----------------------------------------------------------------------===//

def TransformDialectInterpreter : Pass<"transform-dialect-interpreter", "ModuleOp"> {
  let summary = "Apply transform dialect operations one by one";
  let constructor = "mlir::createTransformDialectInterpreter()";
  let options = [
    Option<"eraseAfter", "erase-after", "bool",
            /*default=*/"false",
            "Erase after applied">
  ];
}

//===----------------------------------------------------------------------===//
// DetensorizeTransformationInsertion
//===----------------------------------------------------------------------===//

def DetensorizeTransformInsertion : Pass<"insert-detensorize-transform", "ModuleOp"> {
  let summary = "Insert detensorize transform IR to functions.";
  let constructor = "mlir::createDetensorizeTransformInsertionPass()";
  let options = [
    Option<"funcAnchorAttr", "func-anchor", "std::string",
            /*default=*/"",
            "An optional Unit attribute anchoring on target functions.">,
    Option<"matchPrefix", "match-prefix", "std::string",
            /*default=*/"\"__byteir_detensorize\"",
            "An optional match prefix attribute on target ops.">,
  ];
}


//===----------------------------------------------------------------------===//
// FuseExtTransformInsertion
//===----------------------------------------------------------------------===//

def FuseExtTransformInsertion : Pass<"insert-fuse-ext-transform", "ModuleOp"> {
  let summary = "Insert fuse-ext transform IR to device functions.";
  let constructor = "mlir::createFuseExtTransformInsertionPass()";
  let options = [
    Option<"funcAnchorAttr", "func-anchor", "std::string",
            /*default=*/"",
            "An optional Unit attribute anchoring on target functions.">,
    Option<"matchPrefix", "match-prefix", "std::string",
            /*default=*/"",
            "An optional match prefix attribute on target ops.">,
    Option<"tileSizeAttrName", "tile-size-attr-name", "std::string",
            /*default=*/"",
            "Tile size attribute name.">,
    Option<"tileInterchangeAttrName", "tile-interchange-attr-name", "std::string",
            /*default=*/"",
            "Tile interchange attribute name.">,
    Option<"keepIntermediates", "keep-intermediates-attr-name", "bool",
            /*default=*/"false",
            "keep intermediates attribute name.">
  ];
}

//===----------------------------------------------------------------------===//
// RewriteInDPSTransformInsertion
//===----------------------------------------------------------------------===//

def RewriteInDPSTransformInsertion : Pass<"insert-rewrite-in-dps-transform", "ModuleOp"> {
  let summary = "Insert rewrite in destination-passing-style transform IR to functions.";
  let constructor = "mlir::createRewriteInDPSTransformInsertionPass()";
  let options = [
    Option<"funcAnchorAttr", "func-anchor", "std::string",
            /*default=*/"",
            "An optional Unit attribute anchoring on target functions.">,
    Option<"matchPrefix", "match-prefix", "std::string",
            /*default=*/"\"__byteir_detensorize\"",
            "An optional match prefix attribute on target ops.">,
  ];
}


#endif // BYTEIR_DIALECT_TRANSFORM_PASSES
