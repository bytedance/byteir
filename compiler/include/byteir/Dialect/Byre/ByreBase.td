//===-- ByreBase.td - Byre dialect definitions -------------*- tablegen -*-===//
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
// Defines the Byre dialect
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_DIALECT_BYRE_BYRE_BASE
#define BYTEIR_DIALECT_BYRE_BYRE_BASE

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

//===----------------------------------------------------------------------===//
// Byre Dialect.
//===----------------------------------------------------------------------===//

def Byre_Dialect : Dialect {
  let name = "byre";
  let cppNamespace = "::mlir::byre";
  let hasOperationAttrVerify = 1;
  let useDefaultTypePrinterParser = 0;

  let extraClassDeclaration = [{
    /// Get the name of the attribute used to annotate the modules that contain
    /// kernel modules.
    static StringRef getContainerModuleAttrName() {
      return "byre.container_module";
    }

    static StringRef getModuleMemorySpaceAttrName() {
      return "byre.memory_space";
    }

    static StringRef getEntryPointFunctionAttrName() {
      return "byre.entry_point";
    }

    static StringRef getEntryPointFuncArgTypeAttrName() {
      return "byre.argtype";
    }

    static StringRef getEntryPointFuncArgNameAttrName() {
      return "byre.argname";
    }

    static StringRef getEntryPointFuncArgWeightValueAttrName() {
      return "byre.weight_value";
    }

    static StringRef getEntryPointFuncArgAliasIndexAttrName() {
      return "byre.arg_alias_index";
    }

    static StringRef getEntryPointFuncResultAttrsName() {
      return "byre.result_attrs";
    }

    ::mlir::Type parseType(::mlir::DialectAsmParser &parser) const override;

    void printType(::mlir::Type type,
                    ::mlir::DialectAsmPrinter &printer) const override;
  }];
}

//===----------------------------------------------------------------------===//
// Byre Types.
//===----------------------------------------------------------------------===//

def Byre_AsyncToken : DialectType<
  Byre_Dialect, CPred<"$_self.isa<::mlir::byre::AsyncTokenType>()">, "async token type">,
             BuildableType<"mlir::byre::AsyncTokenType::get($_builder.getContext())">;

//===----------------------------------------------------------------------===//
// Byre Attributes.
//===----------------------------------------------------------------------===//

def Byre_ArgType_None : I32BitEnumAttrCaseNone<"None">;            // 0
def Byre_ArgType_Input : I32BitEnumAttrCaseBit<"Input", 0x0000>;   // 1
def Byre_ArgType_Output : I32BitEnumAttrCaseBit<"Output", 0x0001>; // 2
def Byre_ArgType_Weight : I32BitEnumAttrCaseBit<"Weight", 0x0002>; // 4

def Byre_ArgTypeAttr : I32BitEnumAttr<
    "EntryFuncArgType",
    "entry function argument type", [
        Byre_ArgType_None,
        Byre_ArgType_Input,
        Byre_ArgType_Output,
        Byre_ArgType_Weight
    ]> {
  let cppNamespace = "::mlir::byre";
}

def Byre_MemoryEffect_None : I32BitEnumAttrCaseNone<"None">;           // 0
def Byre_MemoryEffect_Read : I32BitEnumAttrCaseBit<"Read", 0x0000>;    // 1
def Byre_MemoryEffect_Write : I32BitEnumAttrCaseBit<"Write", 0x0001>;  // 2

def Byre_MemoryEffectAttr : I32BitEnumAttr<
    "MemoryEffect",
    "memory effect", [
      Byre_MemoryEffect_None,
      Byre_MemoryEffect_Read,
      Byre_MemoryEffect_Write
    ]> {
  let cppNamespace = "::mlir::byre";
}

//===----------------------------------------------------------------------===//
// Byre OpInterfaces.
//===----------------------------------------------------------------------===//

def Byre_AsyncOpInterface : OpInterface<"AsyncOpInterface"> {
  let description = [{
    Interface for Runtime operations that execute asynchronously on the device.

    The op doesn't start executing until all depent ops producing the async
    dependency tokens have finished executing.

    If the op returns a token, the op merely schedules the execution on the
    device and returns immediately, without waiting for the execution to
    complete. On the hand, if the op does not return a token, the op will wait
    for the execution to complete.
  }];
  let cppNamespace = "::mlir::byre";

  let methods = [
    InterfaceMethod<[{
        Query the operands that represent async dependency tokens.
      }],
      "OperandRange", "getAsyncDependencies", (ins), [{}], [{
        ConcreteOp op = cast<ConcreteOp>(this->getOperation());
        return op.asyncDependencies();
      }]
    >,
    InterfaceMethod<[{
        Adds a new token to the list of async dependencies.
      }],
      "void", "addAsyncDependency", (ins "Value":$token),
      [{}], [{
        ::mlir::byre::addAsyncDependency(this->getOperation(), token);
      }]
    >,
    InterfaceMethod<[{
        Query the result that represents the async token to depend on.
      }],
      "OpResult", "getAsyncToken", (ins), [{}], [{
        ConcreteOp op = cast<ConcreteOp>(this->getOperation());
        return op.asyncToken().template dyn_cast_or_null<OpResult>();
      }]
    >
  ];
}

// The 'ByreInterface' provides access to the 'ByreOp' interface.
def ByreInterface : OpInterface<"ByreOp"> {
  let cppNamespace = "::mlir::byre";
  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Return the name of callee for this op.
      }],
      /*retTy=*/"std::string",
      /*methodName=*/"getCalleeName",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        Operation* op = $_op.getOperation();
        if (auto flatSymAttr = op->getAttrOfType<::mlir::FlatSymbolRefAttr>("callee")) {
          return flatSymAttr.getValue().str();
        } else if (auto strAttr = op->getAttrOfType<::mlir::StringAttr>("callee")) {
          return strAttr.str();
        }
        return "";
      }]
    >
  ];

  let extraClassDeclaration = [{
    // Retrieve the inputs of the byre op, return its operands directly if op is
    // not inherited from MemoryEffectOpInterface or with Pure, else
    // only the operand with MemoryEffect::Read would be treated as input
    SmallVector<Value> getInputs();

    // Retrieve the outputs of the byre op, return its results directly if op is
    // not inherited from MemoryEffectOpInterface or with Pure, else the
    // operand with MemoryEffect::Write would be treated as output additionally
    SmallVector<Value> getOutputs();
  }];
}

#endif // BYTEIR_DIALECT_BYRE_BYRE_BASE
