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


#ifndef BYTEIR_DIALECT_BYRE_PASSES
#define BYTEIR_DIALECT_BYRE_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// ByreFold
//===----------------------------------------------------------------------===//

def ByreFold : Pass<"byre-fold", "func::FuncOp"> {
  let summary = "Fold Byre Ops";
  let constructor = "mlir::createByreFoldPass()";
}

//===----------------------------------------------------------------------===//
// DumpByre
//===----------------------------------------------------------------------===//

def DumpByre : Pass<"dump-byre", "ModuleOp"> {
  let summary = "Dump byre module to bytecode";
  let constructor = "mlir::createDumpByrePass()";
  let dependentDialects = [
    "mlir::byre::ByreDialect",
    "mlir::byre::serialization::ByreSerialDialect",
  ];
  let options = [
    Option<"fileName", "file-name", "std::string",
            /*default=*/"",
            "The filename to dump the byre bytecode">,
    Option<"version", "version", "std::string",
            /*default=*/"\"current\"",
            "Target version of serializable byre module">,
  ];
}

//===----------------------------------------------------------------------===//
// LoadByre
//===----------------------------------------------------------------------===//

def LoadByre : Pass<"load-byre", "ModuleOp"> {
  let summary = "Load byre module";
  let constructor = "mlir::createLoadByrePass()";
  let dependentDialects = [
    "mlir::byre::ByreDialect",
    "mlir::byre::serialization::ByreSerialDialect",
    "mlir::func::FuncDialect",
    "mlir::memref::MemRefDialect",
    "mlir::ace::AceDialect",
  ];
}


//===----------------------------------------------------------------------===//
// ByreToByreSerial
//===----------------------------------------------------------------------===//

def ByreToByreSerial : Pass<"byre-to-byre-serial", "ModuleOp"> {
  let summary = "Convert byre to serializable byre dialect";
  let constructor = "mlir::createByreToByreSerialPass()";
  let dependentDialects = [
    "mlir::byre::ByreDialect",
    "mlir::byre::serialization::ByreSerialDialect",
  ];
}

//===----------------------------------------------------------------------===//
// ByreSerialToByre
//===----------------------------------------------------------------------===//

def ByreSerialToByre : Pass<"byre-serial-to-byre", "ModuleOp"> {
  let summary = "Convert serializable byre to byre dialect";
  let constructor = "mlir::createByreSerialToByrePass()";
  let dependentDialects = [
    "mlir::byre::ByreDialect",
    "mlir::byre::serialization::ByreSerialDialect",
    "mlir::func::FuncDialect",
    "mlir::memref::MemRefDialect",
    "mlir::ace::AceDialect",
  ];
}

#endif // BYTEIR_DIALECT_BYRE_PASSES