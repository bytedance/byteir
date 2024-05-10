//===-- ByreSerial.td - Byre dialect serializations ---------*- tablegen -*-===//
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
// Some code comes from openxla/stablehlo project, the original license:
// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
// Copyright 2022 The StableHLO Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_DIALECT_BYRE_BYRE_SERIAL
#define BYTEIR_DIALECT_BYRE_BYRE_SERIAL

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

// ----- byre serialization dialect ------------------------------------------//

def ByreSerial_Dialect : Dialect {
  let name = "byre_serial";
  let cppNamespace = "::mlir::byre::serialization";

  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;
  let usePropertiesForAttributes = 1;
}

// ----- interfaces and helper class ----------------------------------------//

def Byre_SerializableTypeInterface : TypeInterface<"SerializableTypeInterface"> {
  let cppNamespace = "::mlir::byre::serialization";
  let methods = [
    InterfaceMethod<
      "Returns the minimum version of the Byre dialect an attribute is supported in.",
      "mlir::byre::serialization::Version", "getMinVersion">,
    InterfaceMethod<
      "Returns the maximum version (inclusive) of the Byre dialect an attribute is supported in.",
      "mlir::byre::serialization::Version", "getMaxVersion">,
  ];
}

class Byre_SerializableType<string cppName, string name, string minVersion, string maxVersion>
  : TypeDef<ByreSerial_Dialect, cppName, [Byre_SerializableTypeInterface]> {
  let mnemonic = name;
  let extraClassDeclaration = [{
    mlir::byre::serialization::Version getMinVersion() {
      return mlir::byre::serialization::Version(}] # !subst(".", ", ", minVersion) # [{);
    }
    mlir::byre::serialization::Version getMaxVersion() {
      }] # !if(
        !eq(maxVersion, "current"),
        [{ return mlir::byre::serialization::Version::getCurrentVersion(); }],
        [{ return mlir::byre::serialization::Version("}] # !subst(".", ", ", maxVersion) # [{"); }]
      ) # [{
    }
  }];
}

def Byre_SerializableAttrInterface : AttrInterface<"SerializableAttrInterface"> {
  let cppNamespace = "::mlir::byre::serialization";
  let methods = [
    InterfaceMethod<
      "Returns the minimum version of the Byre dialect an attribute is supported in.",
      "mlir::byre::serialization::Version", "getMinVersion">,
    InterfaceMethod<
      "Returns the maximum version (inclusive) of the Byre dialect an attribute is supported in.",
      "mlir::byre::serialization::Version", "getMaxVersion">,
  ];
}

class Byre_SerializableAttr<string cppName, string minVersion, string maxVersion>
  : AttrDef<ByreSerial_Dialect, cppName, [Byre_SerializableAttrInterface]> {
  let extraClassDeclaration = [{
    mlir::byre::serialization::Version getMinVersion() {
      return mlir::byre::serialization::Version(}] # !subst(".", ", ", minVersion) # [{);
    }
    mlir::byre::serialization::Version getMaxVersion() {
      }] # !if(
        !eq(maxVersion, "current"),
        [{ return mlir::byre::serialization::Version::getCurrentVersion(); }],
        [{ return mlir::byre::serialization::Version("}] # !subst(".", ", ", maxVersion) # [{"); }]
      ) # [{
    }
  }];
}

class Byre_SerializableI32Enum<string name, string summary,
                            list<I32EnumAttrCase> cases> :
    I32EnumAttr<name, summary, cases> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::byre::serialization";
}

class Byre_SerializableBitEnum<string name, string summary,
                               list<BitEnumAttrCaseBase> cases> :
    I32BitEnumAttr<name, summary, cases> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::byre::serialization";
}

class Byre_SerializableEnumAttr<EnumAttrInfo enumInfo, string mnemonic, string minVersion, string maxVersion>
  : EnumAttr<ByreSerial_Dialect, enumInfo, mnemonic, [Byre_SerializableAttrInterface]> {
  let extraClassDeclaration = [{
    mlir::byre::serialization::Version getMinVersion() {
      return mlir::byre::serialization::Version(}] # !subst(".", ", ", minVersion) # [{);
    }
    mlir::byre::serialization::Version getMaxVersion() {
      }] # !if(
        !eq(maxVersion, "current"),
        [{ return mlir::byre::serialization::Version::getCurrentVersion(); }],
        [{ return mlir::byre::serialization::Version("}] # !subst(".", ", ", maxVersion) # [{"); }]
      ) # [{
    }
  }];
}

class Byre_SerializableI32EnumAttr<string name, string mnemonic, string minVersion, string maxVersion, list<I32EnumAttrCase> cases>:
  Byre_SerializableEnumAttr<Byre_SerializableI32Enum<name, name, cases>, mnemonic, minVersion, maxVersion>;

class Byre_SerializableBitEnumAttr<string name, string mnemonic, string minVersion, string maxVersion, list<BitEnumAttrCaseBase> cases>:
  Byre_SerializableEnumAttr<Byre_SerializableBitEnum<name, name, cases>, mnemonic, minVersion, maxVersion>;

def Byre_SerializableOpInterface : OpInterface<"SerializableOpInterface"> {
  let cppNamespace = "::mlir::byre::serialization";
  let methods = [
    InterfaceMethod<
      "Returns the minimum version of the Byre dialect an op is supported in.",
      "mlir::byre::serialization::Version", "getMinVersion">,
    InterfaceMethod<
      "Returns the maximum version (inclusive) of the Byre dialect an op is supported in.",
      "mlir::byre::serialization::Version", "getMaxVersion">,
  ];
}

class Byre_SerializableOp<string mnemonic, string minVersion, string maxVersion, list<Trait> traits = []> :
    Op<ByreSerial_Dialect, mnemonic,
      [DeclareOpInterfaceMethods<Byre_SerializableOpInterface>] # traits> {
  let extraClassDefinition = [{
    mlir::byre::serialization::Version $cppClass::getMinVersion() {
      return mlir::byre::serialization::Version(}] # !subst(".", ", ", minVersion) # [{);
    }
    mlir::byre::serialization::Version $cppClass::getMaxVersion() {
      }] # !if(
        !eq(maxVersion, "current"),
        [{ return mlir::byre::serialization::Version::getCurrentVersion(); }],
        [{ return mlir::byre::serialization::Version("}] # !subst(".", ", ", maxVersion) # [{"); }]
      ) # [{
    }
  }];
}

// ----- Byre dialect serializable types ------------------------------------//

def Byre_BooleanTypeV1 : Byre_SerializableType<"BooleanV1", "bool_v1", "1.0.0", "current">;

def Byre_FloatBF16TypeV1 : Byre_SerializableType<"FloatBF16V1", "bf16_v1", "1.0.0", "current">;

def Byre_FloatF16TypeV1 : Byre_SerializableType<"FloatF16V1", "f16_v1", "1.0.0", "current">;

def Byre_FloatF32TypeV1 : Byre_SerializableType<"FloatF32V1", "f32_v1", "1.0.0", "current">;

def Byre_FloatF64TypeV1 : Byre_SerializableType<"FloatF64V1","f64_v1", "1.0.0", "current">;

def Byre_FloatF8E4M3FNTypeV1 : Byre_SerializableType<"FloatF8E4M3FNV1", "f8E4M3FN_v1", "1.0.0", "current">;

def Byre_FloatF8E5M2TypeV1 : Byre_SerializableType<"FloatF8E5M2V1", "f8E5M2_v1", "1.0.0", "current">;

def Byre_FloatF8E4M3FNUZTypeV1 : Byre_SerializableType<"FloatF8E4M3FNUZV1", "f8E4M3FNUZ_v1", "1.0.0", "current">;

def Byre_FloatF8E4M3B11FNUZTypeV1 : Byre_SerializableType<"FloatF8E4M3B11FNUZV1", "f8E4M3B11FNUZ_v1", "1.0.0", "current">;

def Byre_FloatF8E5M2FNUZTypeV1 : Byre_SerializableType<"FloatF8E5M2FNUZV1", "f8E5M2FNUZ_v1", "1.0.0", "current">;

def Byre_IndexTypeV1 : Byre_SerializableType<"IndexV1", "index_v1", "1.0.0", "current">;

def Byre_IntegerI4TypeV1 : Byre_SerializableType<"IntegerI4V1", "i4_v1", "1.0.0", "current">;

def Byre_IntegerI8TypeV1 : Byre_SerializableType<"IntegerI8V1", "i8_v1", "1.0.0", "current">;

def Byre_IntegerI16TypeV1 : Byre_SerializableType<"IntegerI16V1", "i16_v1", "1.0.0", "current">;

def Byre_IntegerI32TypeV1 : Byre_SerializableType<"IntegerI32V1", "i32_v1", "1.0.0", "current">;

def Byre_IntegerI64TypeV1 : Byre_SerializableType<"IntegerI64V1", "i64_v1", "1.0.0", "current">;

def Byre_IntegerUI4TypeV1 : Byre_SerializableType<"IntegerUI4V1", "ui4_v1", "1.0.0", "current">;

def Byre_IntegerUI8TypeV1 : Byre_SerializableType<"IntegerUI8V1", "ui8_v1", "1.0.0", "current">;

def Byre_IntegerUI16TypeV1 : Byre_SerializableType<"IntegerUI16V1", "ui16_v1", "1.0.0", "current">;

def Byre_IntegerUI32TypeV1 : Byre_SerializableType<"IntegerUI32V1", "ui32_v1", "1.0.0", "current">;

def Byre_IntegerUI64TypeV1 : Byre_SerializableType<"IntegerUI64V1", "ui64_v1", "1.0.0", "current">;

def Byre_StringTypeV1 : Byre_SerializableType<"StringV1", "string_v1", "1.0.0", "current">;

// TODO: support `"MemRefLayoutAttrInterface":$layout` if need.
def Byre_MemrefTypeV1 : Byre_SerializableType<"MemrefV1", "memref_v1", "1.0.0", "current"> {
  let parameters = (ins
    ArrayRefParameter<"int64_t">:$shape,
    "mlir::Type":$elementType,
    "mlir::Attribute":$memorySpace
  );
  let assemblyFormat = "`<` custom<Shape>($shape) `` $elementType `` custom<MemorySpace>($memorySpace) `>`";
}

def Byre_RankedTensorV1 : Byre_SerializableType<"RankedTensorV1", "rankedtensor_V1", "1.0.0", "current"> {
  let parameters = (ins
    ArrayRefParameter<"int64_t">:$shape,
    "mlir::Type":$elementType,
    "mlir::Attribute":$encoding
  );
  let assemblyFormat = "`<` custom<Shape>($shape) `` $elementType `` custom<Encoding>($encoding) `>`"; 
}

def Byre_FunctionTypeV1 : Byre_SerializableType<"FunctionV1", "functiontype_v1", "1.0.0", "current"> {
  let parameters = (ins
    ArrayRefParameter<"mlir::Type">:$inputs,
    ArrayRefParameter<"mlir::Type">:$outputs
  );
  let assemblyFormat = "`<` `(` custom<TypeArray>($inputs) `)` `->` custom<TypeArray>($outputs) `>`";
}

// ----- Byre dialect serializable attrs ------------------------------------//

def Byre_IntegerAttrV1 : Byre_SerializableAttr<"IntegerV1", "1.0.0", "current"> {
  let mnemonic = "integer_v1";
  let parameters = (ins "mlir::Type":$type, "APInt":$value);
  let assemblyFormat = "`<` custom<IntegerAttrV1>($value, $type) `>`";
}

def Byre_FloatAttrV1 : Byre_SerializableAttr<"FloatV1", "1.0.0", "current"> {
  let mnemonic = "float_v1";
  let parameters = (ins "mlir::Type":$type, "APFloat":$value);
  let assemblyFormat = "`<` custom<FloatAttrV1>($value, $type) `>`";
}

def Byre_UnitAttrV1 : Byre_SerializableAttr<"UnitV1", "1.0.0", "current"> {
  let mnemonic = "unit_v1";
}

def Byre_ArrayAttrV1 : Byre_SerializableAttr<"ArrayV1", "1.0.0", "current"> {
  let mnemonic = "array_v1";
  let parameters = (ins ArrayRefParameter<"mlir::Attribute">:$value);
  let assemblyFormat = "`<` custom<ArrayAttrV1>($value) `>`";
}

def Byre_DictionaryAttrV1 : Byre_SerializableAttr<"DictionaryV1", "1.0.0", "current"> {
  let mnemonic = "dict_v1";
  let parameters = (ins ArrayRefParameter<"std::pair<mlir::Attribute, mlir::Attribute>", "">:$value);
  let assemblyFormat = "`<` custom<DictionaryAttrV1>($value) `>`";
}

def Byre_StringAttrV1 : Byre_SerializableAttr<"StringV1", "1.0.0", "current"> {
  let mnemonic = "string_v1";
  let parameters = (ins StringRefParameter<"">:$value);
  let assemblyFormat = "`<` $value `>`";
}

def Byre_TypeAttrV1 : Byre_SerializableAttr<"TypeV1", "1.0.0", "current"> {
  let mnemonic = "type_v1";
  let parameters = (ins "::mlir::Type":$value);
  let assemblyFormat = "`<` $value `>`";
}

def Byre_SymbolRefAttrV1 : Byre_SerializableAttr<"SymbolRefV1", "1.0.0", "current"> {
  let mnemonic = "symbolref_v1";
  let parameters = (ins "mlir::Attribute":$rootReference, ArrayRefParameter<"mlir::Attribute">:$nestedReferences);
  let assemblyFormat = "`<` $rootReference `` custom<NestedReferences>($nestedReferences) `>`";
}

def Byre_BlobDataV1 : AttrParameter<"::llvm::ArrayRef<char>", ""> {
  let allocator = "$_dst = $_allocator.copyInto($_self);";
}

def Byre_DenseIntOrFPElementsAttrV1 : Byre_SerializableAttr<"DenseIntOrFPElementsV1", "1.0.0", "current"> {
  let mnemonic = "dense_int_or_fp_elements_v1";
  let parameters = (ins "::mlir::Type":$type, Byre_BlobDataV1:$data);
  let hasCustomAssemblyFormat = 1;
}

def Byre_DenseArrayAttrV1 : Byre_SerializableAttr<"DenseArrayV1", "1.0.0", "current"> {
  let mnemonic = "dense_array_v1";
  let parameters = (ins "mlir::Type":$elementType, "int64_t":$size, Byre_BlobDataV1:$data);
  let hasCustomAssemblyFormat = 1;
}

def Byre_DenseStringElementsAttrV1 : Byre_SerializableAttr<"DenseStringElementsV1", "1.0.0", "current"> {
  let mnemonic = "dense_string_elements_v1";
  let parameters = (ins "mlir::Type":$type, "llvm::ArrayRef<llvm::StringRef>":$value);
  let builders = [
    AttrBuilderWithInferredContext<(ins "mlir::Type":$type, "llvm::ArrayRef<llvm::StringRef>":$value), [{
      bool isSplat = (value.size() == 1);
      return $_get(type.getContext(), type, value, isSplat);
    }]>,
  ];
  let hasCustomAssemblyFormat = 1;
  let genStorageClass = 0;
  let skipDefaultBuilders = 1;
}

// ----- Byre dialect serializable enums ------------------------------------//

def Byre_ArgTypeV1_None : I32BitEnumAttrCaseNone<"None">;            // 0
def Byre_ArgTypeV1_Input : I32BitEnumAttrCaseBit<"Input", 0x0000>;   // 1
def Byre_ArgTypeV1_Output : I32BitEnumAttrCaseBit<"Output", 0x0001>; // 2
def Byre_ArgTypeV1_Weight : I32BitEnumAttrCaseBit<"Weight", 0x0002>; // 4

def Byre_ArgTypeAttrV1 : Byre_SerializableBitEnumAttr<"ArgTypeV1", "arg_type_v1", "1.0.0", "current", [
        Byre_ArgTypeV1_None,
        Byre_ArgTypeV1_Input,
        Byre_ArgTypeV1_Output,
        Byre_ArgTypeV1_Weight
    ]>;

def Byre_MemoryEffectV1_None : I32BitEnumAttrCaseNone<"None">;           // 0
def Byre_MemoryEffectV1_Read : I32BitEnumAttrCaseBit<"Read", 0x0000>;    // 1
def Byre_MemoryEffectV1_Write : I32BitEnumAttrCaseBit<"Write", 0x0001>;  // 2

def Byre_MemoryEffectAttrV1 : Byre_SerializableBitEnumAttr<"MemoryEffectV1", "memory_effect_v1", "1.0.0", "current", [
      Byre_MemoryEffectV1_None,
      Byre_MemoryEffectV1_Read,
      Byre_MemoryEffectV1_Write
    ]>;

// ----- Byre dialect serializable ops --------------------------------------//

def Byre_ModuleOpV1 : Byre_SerializableOp<"module_v1", "1.0.0", "current", [NoTerminator]> {
  let arguments = (ins AnyAttr:$container_module,
                       OptionalAttr<AnyAttr>:$sym_name,
                       OptionalAttr<AnyAttr>:$sym_visibility);
  let regions = (region AnyRegion:$bodyRegion);
}

def Byre_FuncOpV1 : Byre_SerializableOp<"func_v1", "1.0.0", "current"> {
  let arguments = (ins AnyAttr:$sym_name,
                       AnyAttr:$function_type,
                       AnyAttr:$byre_entry_point,
                       AnyAttr:$byre_arg_types,
                       AnyAttr:$byre_arg_names,
                       OptionalAttr<AnyAttr>:$byre_arg_alias_indexs,
                       OptionalAttr<AnyAttr>:$byteir_entry_point,
                       OptionalAttr<AnyAttr>:$tf_original_input_names,
                       OptionalAttr<AnyAttr>:$device_file_name,
                       OptionalAttr<AnyAttr>:$sym_visibility);
  let regions = (region AnyRegion:$body);
}

def Byre_ReturnOpV1 : Byre_SerializableOp<"return_v1", "1.0.0", "current", [Terminator]> {
  let arguments = (ins Variadic<AnyType>:$operands);
}

def Byre_AllocOpV1 : Byre_SerializableOp<"alloc_v1", "1.0.0", "current"> {
  let arguments = (ins Variadic<AnyType>:$operands,
                       AnyAttr:$operand_segment_sizes,
                       OptionalAttr<AnyAttr>:$alignment);
  let results = (outs AnyType:$memref);
}

def Byre_ComputeOpV1 : Byre_SerializableOp<"compute_v1", "1.0.0", "current"> {
  let arguments = (ins
    AnyAttr:$callee,
    Variadic<AnyType>:$operands,
    OptionalAttr<AnyAttr>:$memory_effects,
    OptionalAttr<AnyAttr>:$extra_args
  );

  let results = (outs
    Variadic<AnyType>:$results
  );
}

def Byre_CopyOpV1 : Byre_SerializableOp<"copy_v1", "1.0.0", "current"> {
  let arguments = (ins
    AnyType:$source,
    AnyType:$target,
    AnyAttr:$callee
  );
}

def Byre_GroupCopyOpV1 : Byre_SerializableOp<"group_copy_v1", "1.0.0", "current", [SameVariadicOperandSize]> {
  let arguments = (ins
    Variadic<AnyType>:$source,
    Variadic<AnyType>:$target,
    AnyAttr:$callee
  );
}

def Byre_AliasOpV1 : Byre_SerializableOp<"alias", "1.0.0", "current"> {
  let arguments = (ins
    AnyType:$source,
    AnyAttr:$offset
  );
  let results = (outs
    AnyType:$target
  );
}

#endif // BYTEIR_DIALECT_BYRE_BYRE_SERIAL
