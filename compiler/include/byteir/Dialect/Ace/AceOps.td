//===-- AceOps.td - Ace dialect operation definitions --*- tablegen -*---===//
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


#ifndef BYTEIR_DIALECT_ACE_ACE_OPS
#define BYTEIR_DIALECT_ACE_ACE_OPS

include "byteir/Dialect/Ace/AceBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"


//===----------------------------------------------------------------------===//
// Ace Dialect operations.
//===----------------------------------------------------------------------===//

class Ace_Op<string mnemonic, list<Trait> traits = []> :
    Op<Ace_Dialect, mnemonic, traits> {
}

def Ace_ConstOp : Ace_Op<"constant", [
    ConstantLike, Pure, AllTypesMatch<["value", "output"]>]> {
  let summary = "Constant operator";
  let description = [{
    Represents a constant value.
  }];
  let arguments = (ins
    ElementsAttr:$value
  );

  let results = (outs
    Ace_StaticShapeTensor:$output
  );

  let builders = [
    OpBuilder<(ins "Attribute":$value)>];

  // FIXME: let this op only has generic format for golden ref calculate.
  // let assemblyFormat = "attr-dict $value";
  let hasFolder = 1;
  let hasVerifier = 0;
}


def Ace_ActivateOp : Ace_Op<"activate", [Pure]> {
  let summary = "Activate operation";

  let arguments = (ins
    AnyTensor:$input,
    StrAttr:$act_func
  );

  let results = (
    outs AnyTensor:$output
  );

  let builders = [
    OpBuilder<(ins "::mlir::Type":$output, "::mlir::Value":$input,
      "::mlir::StringAttr":$actType,
      "llvm::ArrayRef<NamedAttribute>":$attrs),
      [{
        $_state.addOperands(input);
        $_state.addAttribute(getActFuncAttrName(odsState.name), actType);
        $_state.addTypes(output);
        $_state.addAttributes(attrs);
      }]>,
  ];
}


def Ace_CustomCallOp : Ace_Op<"custom_call", [Pure]> {
  let summary = "CustomCall operator";

  let arguments = (ins
    Variadic<AnyTensor>:$inputs,
    StrAttr:$call_target_name
  );

  let results = (outs Variadic<AnyTensor>:$results);
}


#endif // BYTEIR_DIALECT_ACE_ACE_OPS
