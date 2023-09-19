//===-- CatBase.td - Cat dialect definitions ---------*------- tablegen -*-===//
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
//
// Defines the Cat dialect
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_DIALECT_CAT_IR_CAT_BASE
#define BYTEIR_DIALECT_CAT_IR_CAT_BASE

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

//===----------------------------------------------------------------------===//
// Cat Dialect.
//===----------------------------------------------------------------------===//

def Cat_Dialect : Dialect {
  let name = "cat";
  let cppNamespace = "::mlir::cat";
}

// The 'CatOpInterface' provides access to the 'CatOp' interface.
def Cat_CatOpInterface : OpInterface<"CatOpInterface"> {
  let cppNamespace = "::mlir::cat";

  let extraClassDeclaration = [{
    // Retrieve the inputs of the cat op, return its operands directly if op is
    // not inherited from MemoryEffectOpInterface or with Pure, else
    // only the operand with MemoryEffect::Read would be treated as input
    SmallVector<Value> getInputs();

    // Retrieve the outputs of the cat op, return its results directly if op is
    // not inherited from MemoryEffectOpInterface or with Pure, else the
    // operand with MemoryEffect::Write would be treated as output additionally
    SmallVector<Value> getOutputs();
  }];
}

#endif // BYTEIR_DIALECT_CAT_IR_CAT_BASE
