//===-- LaceBase.td - Lace dialect definitions ---------*----- tablegen -*-===//
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
// Defines the Lace dialect
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_DIALECT_LACE_LACE_BASE
#define BYTEIR_DIALECT_LACE_LACE_BASE

include "byteir/Dialect/Ace/AceBase.td" // include Ace types
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Lace Dialect.
//===----------------------------------------------------------------------===//

// Lace Dialect works as an extension to lmhlo dialect, these two dialects work
// together as high level IR in byteir.
def Lace_Dialect : Dialect {
  let name = "lace";
  let cppNamespace = "::mlir::lace";
}

//===----------------------------------------------------------------------===//
// Lace Types.
//===----------------------------------------------------------------------===//

def Lace_StringType : TypeAlias<Ace_StringType, "string type in lace dialect">;

def Lace_BufferType : MemRefOf<[AnyFloat, AnyInteger, Lace_StringType]>;

//===----------------------------------------------------------------------===//
// Lace OpInterfaces.
//===----------------------------------------------------------------------===//

def Lace_AliasLikeOpInterface : OpInterface<"AliasLikeOpInterface"> {
  let cppNamespace = "::mlir::lace";

  let methods = [
    InterfaceMethod<[{
        Query the offset between the target memref and source memref
      }], "int64_t", "getOffsetElem"
    >,
  ];
}

#endif // BYTEIR_DIALECT_LACE_LACE_BASE
