//===-- ByreOps.td - Byre dialect operation definitions ----*- tablegen -*-===//
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
// Defines some operations of the Byre dialect.
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_DIALECT_BYRE_BYRE_OPS
#define BYTEIR_DIALECT_BYRE_BYRE_OPS

include "mlir/Dialect/DLTI/DLTIBase.td"
include "byteir/Dialect/Byre/ByreBase.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/DataLayoutInterfaces.td"
include "mlir/Interfaces/CopyOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"

//===----------------------------------------------------------------------===//
// Byre Dialect operations.
//===----------------------------------------------------------------------===//

// Base class
class Byre_Op<string mnemonic, list<Trait> traits = []> :
    Op<Byre_Dialect, mnemonic, traits>;

def Byre_ComputeOp : Byre_Op<"compute",
    [HasParent<"func::FuncOp">, ByreInterface, MemRefsNormalizable,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "compute operation defines a kernel in runtime";
  let description = [{ 
    Example:
    ```mlir
    %2 = byre.compute @my_add(%0, %1) : (f32, f32) -> f32
    ```
  }];

  let arguments = (ins
    FlatSymbolRefAttr:$callee,
    Variadic<AnyType>:$operands,
    OptionalAttr<ArrayAttr>:$memory_effects
  );

  let results = (outs
    Variadic<AnyType>:$results
  );

  let builders = [
     OpBuilder<(ins "StringRef":$callee,
                    "ValueRange":$inputs,
                    "ValueRange":$outputs)>
  ];

  let extraClassDeclaration = [{
    FunctionType getType();
	
    /// Get the argument operands to the called function.
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }

    operand_iterator arg_operand_begin() { return operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }
  }];

  // TODO change assemblyFormat to real parser and printer later
  // for more complicated support
  let assemblyFormat = [{
    $callee `(` $operands `)` attr-dict (`:` type($operands)^)? (`->` type($results)^)?
  }];

  let hasVerifier = 1;
  //let skipDefaultBuilders = 1;
}

def Byre_ComputeShapeOp : Byre_Op<"compute_shape",
    [HasParent<"func::FuncOp">,
     DeclareOpInterfaceMethods<ByreInterface, ["getCalleeName"]>,
     MemRefsNormalizable, Pure]> {
  let summary = "operation which references a shape computation fn";
  let description = [{
    ComputeShape op references a shape computation function in runtime,
    unlike normal computing task, shape computation task works on the
    tensor meta(e.g. tensor shape) and doesn't aware any actual tensor data,
    so they can be scheduled at the very first even before intermediate
    memory allocations.
  }];

  let arguments = (ins
    StrAttr:$shape_fn,
    Variadic<AnyMemRef>:$operands
  );

  let results = (outs
    Variadic<Index>
  );

  let hasVerifier = 1;
}

def Byre_CopyOp : Byre_Op<"copy",
    [HasParent<"func::FuncOp">, ByreInterface, CopyOpInterface]> {
  let summary = "copy operation";
  let description = [{ 
    Copies the data to from the source memref to the target memref .

    Usage:

    ```mlir
    byre.copy(%arg0, %arg1) : memref<?xf32, stride_specification>,
                              memref<?xf32, stride_specification>
    ```
    Note %arg0 is source, %arg1 is destination.
  }];

  let arguments = (ins
    Arg<AnyStridedMemRef, "", [MemRead]>:$source,
    Arg<AnyStridedMemRef, "", [MemWrite]>:$target);

  let assemblyFormat = [{
    `(` $source `,` $target `)` attr-dict `:`
        type($source) `,` type($target)
  }];

  let hasCanonicalizer = 1;
  let hasFolder = 1;
  let hasVerifier = 1;
}

def Byre_GroupCopyOp : Byre_Op<"group_copy",
    [HasParent<"func::FuncOp">, ByreInterface, SameVariadicOperandSize]> {
  let summary = "group copy operation";
  let description = [{ 
    Copies data to from collective source memref to target memref .
  }];

  let arguments = (ins
    Arg<Variadic<AnyStridedMemRef>, "", [MemRead]>:$source,
    Arg<Variadic<AnyStridedMemRef>, "", [MemWrite]>:$target);

  let hasCanonicalizer = 1;
  let hasVerifier = 1;
}

def Byre_AliasOp
    : Byre_Op<"alias",
              [HasParent<"func::FuncOp">,
               DeclareOpInterfaceMethods<ByreInterface, ["getCalleeName"]>,
               DeclareOpInterfaceMethods<ViewLikeOpInterface>,
               Pure]> {
  let summary = "alias operation";

  let arguments = (ins
    AnyStridedMemRef:$source,
    I64Attr:$offset
  );

  let results = (outs
    AnyStridedMemRef:$target
  );

  let hasCanonicalizer = 1;
  let hasVerifier = 1;
}

def Byre_CustomOp : Byre_Op<"custom",
    [HasParent<"func::FuncOp">,
    DeclareOpInterfaceMethods<ByreInterface, ["getCalleeName"]>]> {
  let summary = "compute custom operation passed by library path and api name. ";
  let description = [{ 
    Example:
    ```mlir
    %2 = byre.custom(%0, %1) { lib_path = "xxx.so", api_name = "add", version = "1.0.0", extra_args = [0 : i64, 1 : i64, 2.0 : f32] } : (f32, f32) -> f32
    ```
    During execution, "xxx.so" will be loaded, and "add" function will be called.
  }];

  let arguments = (ins
    StrAttr:$lib_path,
    StrAttr:$api_name,
    StrAttr:$version,
    Variadic<AnyType>:$operands,
    ArrayAttr:$extra_args,
    OptionalAttr<ArrayAttr>:$memory_effects
  );

  let results = (outs
    Variadic<AnyType>:$results
  );

  let builders = [
     OpBuilder<(ins "StringRef":$lib_path,
                    "StringRef":$api_name,
                    "StringRef":$version,
                    "ValueRange":$inputs,
                    "ValueRange":$outputs,
                    "ArrayAttr":$extra_args)>
  ];

  let extraClassDeclaration = [{
    FunctionType getType();
	
    /// Get the argument operands to the called function.
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }

    operand_iterator arg_operand_begin() { return operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }
  }];

  let hasVerifier = 1;
}

#endif // BYTEIR_DIALECT_BYRE_BYRE_OPS
