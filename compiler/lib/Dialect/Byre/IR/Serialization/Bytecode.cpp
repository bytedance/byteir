//===- Bytecode.cpp -------------------------------------------------------===//
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

#include "byteir/Dialect/Byre/Serialization/ByreSerialOps.h"
#include "byteir/Dialect/Byre/Serialization/Versioning.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "llvm/ADT/TypeSwitch.h"

#include "./Bytecode.h"

using namespace mlir;
using namespace mlir::byre;
using namespace mlir::byre::serialization;

namespace {
enum class TypeKind : uint32_t {
  BooleanV1 = 0,
  FloatBF16V1 = 1,
  FloatF16V1 = 2,
  FloatF32V1 = 3,
  FloatF64V1 = 4,
  FloatF8E4M3FNV1 = 5,
  FloatF8E5M2V1 = 6,
  FloatF8E4M3FNUZV1 = 7,
  FloatF8E4M3B11FNUZV1 = 8,
  FloatF8E5M2FNUZV1 = 9,
  IndexV1 = 10,
  IntegerI4V1 = 11,
  IntegerI8V1 = 12,
  IntegerI16V1 = 13,
  IntegerI32V1 = 14,
  IntegerI64V1 = 15,
  IntegerUI4V1 = 16,
  IntegerUI8V1 = 17,
  IntegerUI16V1 = 18,
  IntegerUI32V1 = 19,
  IntegerUI64V1 = 20,
  StringV1 = 21,
  MemrefV1 = 22,
  RankedTensorV1 = 23,
  FunctionV1 = 24
};

enum class AttrKind : uint32_t {
  IntegerV1 = 0,
  FloatV1 = 1,
  UnitV1 = 2,
  ArrayV1 = 3,
  DictionaryV1 = 4,
  StringV1 = 5,
  TypeV1 = 6,
  SymbolRefV1 = 7,
  DenseIntOrFPElementsV1 = 8,
  DenseStringElementsV1 = 9,
  DenseArrayV1 = 10,
  ArgTypeV1 = 11,
  MemoryEffectV1 = 12
};

const llvm::fltSemantics &getFloatSemantics(Type type) {
  if (type.isa<FloatBF16V1Type>())
    return APFloat::BFloat();
  if (type.isa<FloatF16V1Type>())
    return APFloat::IEEEhalf();
  if (type.isa<FloatF32V1Type>())
    return APFloat::IEEEsingle();
  if (type.isa<FloatF64V1Type>())
    return APFloat::IEEEdouble();
  if (type.isa<FloatF8E4M3FNUZV1Type>())
    return APFloat::Float8E4M3FNUZ();
  if (type.isa<FloatF8E4M3B11FNUZV1Type>())
    return APFloat::Float8E4M3B11FNUZ();
  if (type.isa<FloatF8E4M3FNV1Type>())
    return APFloat::Float8E4M3FN();
  if (type.isa<FloatF8E5M2FNUZV1Type>())
    return APFloat::Float8E5M2FNUZ();
  if (type.isa<FloatF8E5M2V1Type>())
    return APFloat::Float8E5M2();
  llvm::report_fatal_error("unsupported floating-point type");
}

unsigned getBitWidthForIntegerType(Type type) {
  static_assert(IndexType::kInternalStorageBitWidth == 64,
                "unexpected bit width when resolving index type");
  if (type.isa<IndexV1Type>())
    return IndexType::kInternalStorageBitWidth;
  if (type.isa<BooleanV1Type>())
    return 1;
  if (type.isa<IntegerI4V1Type>() || type.isa<IntegerUI4V1Type>())
    return 4;
  if (type.isa<IntegerI8V1Type>() || type.isa<IntegerUI8V1Type>())
    return 8;
  if (type.isa<IntegerI16V1Type>() || type.isa<IntegerUI16V1Type>())
    return 16;
  if (type.isa<IntegerI32V1Type>() || type.isa<IntegerUI32V1Type>())
    return 32;
  if (type.isa<IntegerI64V1Type>() || type.isa<IntegerUI64V1Type>())
    return 64;
  llvm::report_fatal_error("unsupported integer type");
}

//----- type reader and writer

// clang-format off
#define FOR_EACH_TRIVIAL_TYPE(cb) \
  cb(BooleanV1)                   \
  cb(FloatBF16V1)                 \
  cb(FloatF16V1)                  \
  cb(FloatF32V1)                  \
  cb(FloatF64V1)                  \
  cb(FloatF8E4M3FNV1)             \
  cb(FloatF8E5M2V1)               \
  cb(FloatF8E4M3FNUZV1)           \
  cb(FloatF8E4M3B11FNUZV1)        \
  cb(FloatF8E5M2FNUZV1)           \
  cb(IndexV1)                     \
  cb(IntegerI4V1)                 \
  cb(IntegerI8V1)                 \
  cb(IntegerI16V1)                \
  cb(IntegerI32V1)                \
  cb(IntegerI64V1)                \
  cb(IntegerUI4V1)                \
  cb(IntegerUI8V1)                \
  cb(IntegerUI16V1)               \
  cb(IntegerUI32V1)               \
  cb(IntegerUI64V1)               \
  cb(StringV1)
// clang-format on

#define DEF_TRIVIAL_TYPE_WRITER(name)                                          \
  static LogicalResult write(name##Type, DialectBytecodeWriter &writer) {      \
    writer.writeVarInt(static_cast<uint32_t>(TypeKind::name));                 \
    return success();                                                          \
  }

#define DEF_TRIVIAL_TYPE_READER(name)                                          \
  static Type read##name##Type(MLIRContext *ctx, DialectBytecodeReader &) {    \
    return name##Type::get(ctx);                                               \
  }

FOR_EACH_TRIVIAL_TYPE(DEF_TRIVIAL_TYPE_WRITER)
FOR_EACH_TRIVIAL_TYPE(DEF_TRIVIAL_TYPE_READER)

#undef FOR_EACH_TRIVIAL_TYPE
#undef DEF_TRIVIAL_TYPE_WRITER
#undef DEF_TRIVIAL_TYPE_READER

static LogicalResult write(MemrefV1Type t, DialectBytecodeWriter &writer) {
  writer.writeVarInt(static_cast<uint32_t>(TypeKind::MemrefV1));
  writer.writeSignedVarInts(t.getShape());
  writer.writeType(t.getElementType());
  writer.writeOptionalAttribute(t.getMemorySpace());
  return success();
}

static LogicalResult write(RankedTensorV1Type t,
                           DialectBytecodeWriter &writer) {
  writer.writeVarInt(static_cast<uint32_t>(TypeKind::RankedTensorV1));
  writer.writeSignedVarInts(t.getShape());
  writer.writeType(t.getElementType());
  writer.writeOptionalAttribute(t.getEncoding());
  return success();
}

static LogicalResult write(FunctionV1Type t, DialectBytecodeWriter &writer) {
  writer.writeVarInt(static_cast<uint32_t>(TypeKind::FunctionV1));
  writer.writeTypes(t.getInputs());
  writer.writeTypes(t.getOutputs());
  return success();
}

static Type readMemrefV1Type(MLIRContext *ctx, DialectBytecodeReader &reader) {
  SmallVector<int64_t> shape;
  Type elementType;
  Attribute memSpace;
  if (succeeded(reader.readSignedVarInts(shape)) &&
      succeeded(reader.readType(elementType)) &&
      succeeded(reader.readOptionalAttribute(memSpace))) {
    return MemrefV1Type::get(ctx, shape, elementType, memSpace);
  }
  return Type();
}

static Type readRankedTensorV1Type(MLIRContext *ctx,
                                   DialectBytecodeReader &reader) {
  SmallVector<int64_t> shape;
  Type elementType;
  Attribute encoding;
  if (succeeded(reader.readSignedVarInts(shape)) &&
      succeeded(reader.readType(elementType)) &&
      succeeded(reader.readOptionalAttribute(encoding))) {
    return RankedTensorV1Type::get(ctx, shape, elementType, encoding);
  }
  return Type();
}

static Type readFunctionV1Type(MLIRContext *ctx,
                               DialectBytecodeReader &reader) {
  SmallVector<Type> inputs;
  SmallVector<Type> outputs;
  if (succeeded(reader.readTypes(inputs)) &&
      succeeded(reader.readTypes(outputs))) {
    return FunctionV1Type::get(ctx, inputs, outputs);
  }
  return Type();
}

//----- attribute reader and writer

#define FOR_EACH_ENUM_ATTR(cb) cb(ArgTypeV1) cb(MemoryEffectV1)

#define DEF_ENUM_ATTR_WRITER(name)                                             \
  static LogicalResult write(name##Attr attr, DialectBytecodeWriter &writer) { \
    writer.writeVarInt(static_cast<uint32_t>(AttrKind::name));                 \
    writer.writeVarInt(                                                        \
        static_cast<std::underlying_type_t<name>>(attr.getValue()));           \
    return success();                                                          \
  }

#define DEF_ENUM_ATTR_READER(name)                                             \
  static Attribute read##name##Attr(MLIRContext *ctx,                          \
                                    DialectBytecodeReader &reader) {           \
    uint64_t value;                                                            \
    if (succeeded(reader.readVarInt(value))) {                                 \
      auto enumValue = symbolize##name(value);                                 \
      if (enumValue.has_value()) {                                             \
        return name##Attr::get(ctx, *enumValue);                               \
      }                                                                        \
    }                                                                          \
    return Attribute();                                                        \
  }

FOR_EACH_ENUM_ATTR(DEF_ENUM_ATTR_WRITER)
FOR_EACH_ENUM_ATTR(DEF_ENUM_ATTR_READER)

#undef DEF_ENUM_ATTR_WRITER
#undef DEF_ENUM_ATTR_READER
#undef FOR_EACH_ENUM_ATTR

static LogicalResult write(IntegerV1Attr attr, DialectBytecodeWriter &writer) {
  writer.writeVarInt(static_cast<uint32_t>(AttrKind::IntegerV1));
  writer.writeType(attr.getType());
  writer.writeAPIntWithKnownWidth(attr.getValue());
  return success();
}

static LogicalResult write(FloatV1Attr attr, DialectBytecodeWriter &writer) {
  writer.writeVarInt(static_cast<uint32_t>(AttrKind::FloatV1));
  writer.writeType(attr.getType());
  writer.writeAPFloatWithKnownSemantics(attr.getValue());
  return success();
}

static LogicalResult write(UnitV1Attr attr, DialectBytecodeWriter &writer) {
  writer.writeVarInt(static_cast<uint32_t>(AttrKind::UnitV1));
  return success();
}

static LogicalResult write(ArrayV1Attr attr, DialectBytecodeWriter &writer) {
  writer.writeVarInt(static_cast<uint32_t>(AttrKind::ArrayV1));
  writer.writeAttributes(attr.getValue());
  return success();
}

static LogicalResult write(DictionaryV1Attr attr,
                           DialectBytecodeWriter &writer) {
  writer.writeVarInt(static_cast<uint32_t>(AttrKind::DictionaryV1));
  writer.writeList(attr.getValue(), [&](auto attr) {
    writer.writeAttribute(attr.first);
    writer.writeAttribute(attr.second);
  });
  return success();
}

static LogicalResult write(StringV1Attr attr, DialectBytecodeWriter &writer) {
  writer.writeVarInt(static_cast<uint32_t>(AttrKind::StringV1));
  writer.writeOwnedString(attr.getValue());
  return success();
}

static LogicalResult write(TypeV1Attr attr, DialectBytecodeWriter &writer) {
  writer.writeVarInt(static_cast<uint32_t>(AttrKind::TypeV1));
  writer.writeType(attr.getValue());
  return success();
}

static LogicalResult write(SymbolRefV1Attr attr,
                           DialectBytecodeWriter &writer) {
  writer.writeVarInt(static_cast<uint32_t>(AttrKind::SymbolRefV1));
  writer.writeAttribute(attr.getRootReference());
  writer.writeAttributes(attr.getNestedReferences());
  return success();
}

static LogicalResult write(DenseIntOrFPElementsV1Attr attr,
                           DialectBytecodeWriter &writer) {
  writer.writeVarInt(static_cast<uint32_t>(AttrKind::DenseIntOrFPElementsV1));
  writer.writeType(attr.getType());
  writer.writeOwnedBlob(attr.getData());
  return success();
}

static LogicalResult write(DenseStringElementsV1Attr attr,
                           DialectBytecodeWriter &writer) {
  writer.writeVarInt(static_cast<uint32_t>(AttrKind::DenseStringElementsV1));
  writer.writeType(attr.getType());
  writer.writeList(attr.getValue(), [&](llvm::StringRef value) {
    writer.writeOwnedString(value);
  });
  return success();
}

static LogicalResult write(DenseArrayV1Attr attr,
                           DialectBytecodeWriter &writer) {
  writer.writeVarInt(static_cast<uint32_t>(AttrKind::DenseArrayV1));
  writer.writeType(attr.getElementType());
  writer.writeSignedVarInt(attr.getSize());
  writer.writeOwnedBlob(attr.getData());
  return success();
}

static Attribute readIntegerV1Attr(MLIRContext *ctx,
                                   DialectBytecodeReader &reader) {
  Type type;
  if (succeeded(reader.readType(type))) {
    FailureOr<APInt> value =
        reader.readAPIntWithKnownWidth(getBitWidthForIntegerType(type));
    if (succeeded(value)) {
      return IntegerV1Attr::get(ctx, type, *value);
    }
  }
  return Attribute();
}

static Attribute readFloatV1Attr(MLIRContext *ctx,
                                 DialectBytecodeReader &reader) {
  Type type;
  if (succeeded(reader.readType(type))) {
    FailureOr<APFloat> value =
        reader.readAPFloatWithKnownSemantics(getFloatSemantics(type));
    if (succeeded(value)) {
      return FloatV1Attr::get(ctx, type, *value);
    }
  }
  return Attribute();
}

static Attribute readUnitV1Attr(MLIRContext *ctx,
                                DialectBytecodeReader &reader) {
  return UnitV1Attr::get(ctx);
}

static Attribute readArrayV1Attr(MLIRContext *ctx,
                                 DialectBytecodeReader &reader) {
  SmallVector<Attribute> attrs;
  if (succeeded(reader.readAttributes(attrs))) {
    return ArrayV1Attr::get(ctx, attrs);
  }
  return Attribute();
}

static Attribute readDictionaryV1Attr(MLIRContext *ctx,
                                      DialectBytecodeReader &reader) {
  SmallVector<std::pair<Attribute, Attribute>> attrs;
  auto readValue = [&]() -> FailureOr<std::pair<Attribute, Attribute>> {
    Attribute name;
    Attribute value;
    if (succeeded(reader.readAttribute(name)) &&
        succeeded(reader.readAttribute(value))) {
      return std::pair(name, value);
    }
    return failure();
  };
  if (succeeded(reader.readList(attrs, readValue))) {
    return DictionaryV1Attr::get(ctx, attrs);
  }
  return Attribute();
}

static Attribute readStringV1Attr(MLIRContext *ctx,
                                  DialectBytecodeReader &reader) {
  StringRef value;
  if (succeeded(reader.readString(value))) {
    return StringV1Attr::get(ctx, value);
  }
  return Attribute();
}

static Attribute readTypeV1Attr(MLIRContext *ctx,
                                DialectBytecodeReader &reader) {
  Type type;
  if (succeeded(reader.readType(type))) {
    return TypeV1Attr::get(ctx, type);
  }
  return Attribute();
}

static Attribute readSymbolRefV1Attr(MLIRContext *ctx,
                                     DialectBytecodeReader &reader) {
  Attribute root;
  SmallVector<Attribute> nested;
  if (succeeded(reader.readAttribute(root)) &&
      succeeded(reader.readAttributes(nested))) {
    return SymbolRefV1Attr::get(ctx, root, nested);
  }
  return Attribute();
}

static Attribute readDenseIntOrFPElementsV1Attr(MLIRContext *ctx,
                                                DialectBytecodeReader &reader) {
  Type type;
  ArrayRef<char> data;
  if (succeeded(reader.readType(type)) && succeeded(reader.readBlob(data))) {
    return DenseIntOrFPElementsV1Attr::get(ctx, type, data);
  }
  return Attribute();
}

static Attribute readDenseStringElementsV1Attr(MLIRContext *ctx,
                                               DialectBytecodeReader &reader) {
  Type type;
  SmallVector<StringRef> value;
  if (succeeded(reader.readType(type)) &&
      succeeded(reader.readList(
          value, [&](StringRef &e) { return reader.readString(e); }))) {
    return DenseStringElementsV1Attr::get(type, value);
  }
  return Attribute();
}

static Attribute readDenseArrayV1Attr(MLIRContext *ctx,
                                      DialectBytecodeReader &reader) {
  Type type;
  int64_t size;
  ArrayRef<char> data;
  if (succeeded(reader.readType(type)) &&
      succeeded(reader.readSignedVarInt(size)) &&
      succeeded(reader.readBlob(data))) {
    return DenseArrayV1Attr::get(ctx, type, size, data);
  }
  return Attribute();
}

struct ByreDialectBytecodeInterface : public BytecodeDialectInterface {
  using BytecodeDialectInterface::BytecodeDialectInterface;
  ByreDialectBytecodeInterface(Dialect *dialect)
      : BytecodeDialectInterface(dialect) {}

  LogicalResult writeType(Type type,
                          DialectBytecodeWriter &writer) const final {
    return TypeSwitch<Type, LogicalResult>(type)
        .Case<BooleanV1Type, FloatBF16V1Type, FloatF16V1Type, FloatF32V1Type,
              FloatF64V1Type, FloatF8E4M3FNV1Type, FloatF8E5M2V1Type,
              FloatF8E4M3FNUZV1Type, FloatF8E4M3B11FNUZV1Type,
              FloatF8E5M2FNUZV1Type, IndexV1Type, IntegerI4V1Type,
              IntegerI8V1Type, IntegerI16V1Type, IntegerI32V1Type,
              IntegerI64V1Type, IntegerUI4V1Type, IntegerUI8V1Type,
              IntegerUI16V1Type, IntegerUI32V1Type, IntegerUI64V1Type,
              StringV1Type, MemrefV1Type, RankedTensorV1Type, FunctionV1Type>(
            [&](auto type) { return write(type, writer); })
        .Default([](Type) { return failure(); });
  }

  Type readType(DialectBytecodeReader &reader) const final {
    uint64_t kind;
    if (failed(reader.readVarInt(kind)))
      return Type();

    MLIRContext *ctx = getContext();

    switch (static_cast<TypeKind>(kind)) {
#define Case(name)                                                             \
  case TypeKind::name:                                                         \
    return read##name##Type(ctx, reader)

      Case(BooleanV1);
      Case(FloatBF16V1);
      Case(FloatF16V1);
      Case(FloatF32V1);
      Case(FloatF64V1);
      Case(FloatF8E4M3FNV1);
      Case(FloatF8E5M2V1);
      Case(FloatF8E4M3FNUZV1);
      Case(FloatF8E4M3B11FNUZV1);
      Case(FloatF8E5M2FNUZV1);
      Case(IndexV1);
      Case(IntegerI4V1);
      Case(IntegerI8V1);
      Case(IntegerI16V1);
      Case(IntegerI32V1);
      Case(IntegerI64V1);
      Case(IntegerUI4V1);
      Case(IntegerUI8V1);
      Case(IntegerUI16V1);
      Case(IntegerUI32V1);
      Case(IntegerUI64V1);
      Case(StringV1);
      Case(MemrefV1);
      Case(FunctionV1);
      Case(RankedTensorV1);

#undef Case
    default:
      return Type();
    }
  }

  LogicalResult writeAttribute(Attribute attr,
                               DialectBytecodeWriter &writer) const final {
    return TypeSwitch<Attribute, LogicalResult>(attr)
        .Case<IntegerV1Attr, FloatV1Attr, UnitV1Attr, ArrayV1Attr,
              DictionaryV1Attr, StringV1Attr, TypeV1Attr, SymbolRefV1Attr,
              DenseIntOrFPElementsV1Attr, DenseStringElementsV1Attr,
              DenseArrayV1Attr, ArgTypeV1Attr, MemoryEffectV1Attr>(
            [&](auto attr) { return write(attr, writer); })
        .Default([](Attribute) { return failure(); });
  }

  Attribute readAttribute(DialectBytecodeReader &reader) const final {
    uint64_t kind;
    if (failed(reader.readVarInt(kind)))
      return Attribute();

    MLIRContext *ctx = getContext();
    switch (static_cast<AttrKind>(kind)) {
#define Case(name)                                                             \
  case AttrKind::name:                                                         \
    return read##name##Attr(ctx, reader)

      Case(IntegerV1);
      Case(FloatV1);
      Case(UnitV1);
      Case(ArrayV1);
      Case(DictionaryV1);
      Case(StringV1);
      Case(TypeV1);
      Case(SymbolRefV1);
      Case(DenseIntOrFPElementsV1);
      Case(DenseStringElementsV1);
      Case(DenseArrayV1);
      Case(ArgTypeV1);
      Case(MemoryEffectV1);

#undef Case

    default:
      return Attribute();
    }
  }

  // TODO: Version information was added to each type/attribute/op separately
  // and the IR version was not yet attached to the dialect version. It still
  // write the dialect version here for future expasion and to trigger IR
  // upgrade after loading.
  struct ByreDialectVersion : public DialectVersion {
    static constexpr int64_t kVersionNumber = 0;
  };

  void writeVersion(DialectBytecodeWriter &writer) const final {
    writer.writeVarInt(ByreDialectVersion::kVersionNumber);
  }

  std::unique_ptr<DialectVersion>
  readVersion(DialectBytecodeReader &reader) const final {
    uint64_t versionNumber;
    if (failed(reader.readVarInt(versionNumber)))
      return nullptr;

    auto version = std::make_unique<ByreDialectVersion>();
    return version;
  }

  LogicalResult upgradeFromVersion(Operation *topLevelOp,
                                   const DialectVersion &) const final {
    return convertToVersion(topLevelOp, Version::getCurrentVersion());
  }
};
} // namespace

void byre::serialization::addBytecodeInterface(ByreSerialDialect *dialect) {
  dialect->addInterfaces<ByreDialectBytecodeInterface>();
}
