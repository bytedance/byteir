//===-- AceBase.td - Ace dialect definitions ---------*------- tablegen -*-===//
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
// Defines the Ace dialect
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_DIALECT_ACE_ACE_BASE
#define BYTEIR_DIALECT_ACE_ACE_BASE

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

//===----------------------------------------------------------------------===//
// Ace Dialect.
//===----------------------------------------------------------------------===//

// Ace Dialect works as an extension to mhlo dialect, these two dialects work
// together as high level IR in byteir.
def Ace_Dialect : Dialect {
  let name = "ace";
  let cppNamespace = "::mlir::ace";

  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// Ace Type.
//===----------------------------------------------------------------------===//

class Ace_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Ace_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def Ace_StringType : Ace_Type<"String", "string", [MemRefElementTypeInterface]> {
  let description = [{`ace.string` represents string type}];
}

def Ace_ResourceType : Ace_Type<"Resource", "resource"> {
  let description = [{`ace.resource` represents resource type from tensorflow}];
}

def Ace_CustomType : AnyTypeOf<[Ace_StringType, Ace_ResourceType]>;

def Ace_StaticShapeTensor : StaticShapeTensorOf<[Ace_CustomType]>;

//===----------------------------------------------------------------------===//
// Ace Attribute.
//===----------------------------------------------------------------------===//

def TensorEncoding : AttrDef<Ace_Dialect, "TensorEncoding"> {
  let mnemonic = "tensor_encoding";

  let parameters = (ins
    ArrayRefParameter<"int64_t">:$isDynamic
  );

  let summary = "Attribute that extends tensor type.";

  let description = [{
    This attribute is used to extend tensor type with tensor specific
    properties. These properties aren't modeled in the MLIR type. This
    attribute is set in the `encoding` field of the tensor type.

    `is_dynamic` array parameter represents that which dim of tensor is dynamic.
    Sample: tensor<?x16xf32, #ace.tensor_encoding<is_dynamic = [1, 0]>>,
    1 represent the first dimension is dynamic, 0 represent the second
    dimension is not dynamic.

    // TODO(lyq): verify the `is_dynamic` array length is same as shape rank.
  }];
  let assemblyFormat = "`<` `is_dynamic` `=` `[` $isDynamic `]` `>`";

}

#endif // BYTEIR_DIALECT_ACE_ACE_BASE
