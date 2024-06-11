//===- ByreSerialOps.cpp -------------------------------------------------===//
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
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

#include "byteir/Dialect/Byre/Serialization/ByreSerialOps.h"
#include "byteir/Dialect/Byre/Serialization.h"
#include "byteir/Dialect/Byre/Serialization/Versioning.h"
#include "mlir/IR/AttributeSupport.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"

#include <string>

#include "./Bytecode.h"

using namespace mlir;
using namespace mlir::byre;
using namespace mlir::byre::serialization;

void ByreSerialDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "byteir/Dialect/Byre/Serialization/ByreSerialTypes.cpp.inc"
      >();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "byteir/Dialect/Byre/Serialization/ByreSerialAttrs.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "byteir/Dialect/Byre/Serialization/ByreSerialOps.cpp.inc"
      >();

  addBytecodeInterface(this);
}

namespace mlir {
namespace byre {
namespace serialization {
namespace detail {

struct DenseElementsAttributeStorage : public AttributeStorage {
public:
  DenseElementsAttributeStorage(mlir::Type type, bool isSplat)
      : type(type), isSplat(isSplat) {}

  mlir::Type type;
  bool isSplat;
};

/// An attribute representing a reference to a dense vector or tensor object
/// containing strings.
struct DenseStringElementsV1AttrStorage : public DenseElementsAttributeStorage {
  DenseStringElementsV1AttrStorage(mlir::Type ty,
                                   llvm::ArrayRef<llvm::StringRef> data,
                                   bool isSplat = false)
      : DenseElementsAttributeStorage(ty, isSplat), data(data) {}

  struct KeyTy {
    KeyTy(mlir::Type type, llvm::ArrayRef<llvm::StringRef> data,
          llvm::hash_code hashCode, bool isSplat = false)
        : type(type), data(data), hashCode(hashCode), isSplat(isSplat) {}

    /// The type of the dense elements.
    mlir::Type type;

    /// The raw buffer for the data storage.
    llvm::ArrayRef<llvm::StringRef> data;

    /// The computed hash code for the storage data.
    llvm::hash_code hashCode;

    /// A boolean that indicates if this data is a splat or not.
    bool isSplat;
  };

  /// Compare this storage instance with the provided key.
  bool operator==(const KeyTy &key) const {
    if (key.type != type)
      return false;

    // Otherwise, we can default to just checking the data. StringRefs compare
    // by contents.
    return key.data == data;
  }

  /// Construct a key from a shaped type, StringRef data buffer, and a flag that
  /// signals if the data is already known to be a splat. Callers to this
  /// function are expected to tag preknown splat values when possible, e.g. one
  /// element shapes.
  static KeyTy getKey(mlir::Type ty, llvm::ArrayRef<llvm::StringRef> data,
                      bool isKnownSplat) {
    // Handle an empty storage instance.
    if (data.empty())
      return KeyTy(ty, data, 0);

    // If the data is already known to be a splat, the key hash value is
    // directly the data buffer.
    if (isKnownSplat)
      return KeyTy(ty, data, llvm::hash_value(data.front()), isKnownSplat);

    // Handle the simple case of only one element.
    // assert(ty.getNumElements() != 1 &&
    //        "splat of 1 element should already be detected");

    // Create the initial hash value with just the first element.
    const auto &firstElt = data.front();
    auto hashVal = llvm::hash_value(firstElt);

    // Check to see if this storage represents a splat. If it doesn't then
    // combine the hash for the data starting with the first non splat element.
    for (size_t i = 1, e = data.size(); i != e; i++)
      if (!firstElt.equals(data[i]))
        return KeyTy(ty, data, llvm::hash_combine(hashVal, data.drop_front(i)));

    // Otherwise, this is a splat so just return the hash of the first element.
    return KeyTy(ty, data.take_front(), hashVal, /*isSplat=*/true);
  }

  /// Hash the key for the storage.
  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_combine(key.type, key.hashCode);
  }

  /// Construct a new storage instance.
  static DenseStringElementsV1AttrStorage *
  construct(AttributeStorageAllocator &allocator, KeyTy key) {
    // If the data buffer is non-empty, we copy it into the allocator with a
    // 64-bit alignment.
    llvm::ArrayRef<llvm::StringRef> copy, data = key.data;
    if (data.empty()) {
      return new (allocator.allocate<DenseStringElementsV1AttrStorage>())
          DenseStringElementsV1AttrStorage(key.type, copy, key.isSplat);
    }

    int numEntries = key.isSplat ? 1 : data.size();

    // Compute the amount data needed to store the ArrayRef and StringRef
    // contents.
    size_t dataSize = sizeof(llvm::StringRef) * numEntries;
    for (int i = 0; i < numEntries; i++)
      dataSize += data[i].size();

    char *rawData = reinterpret_cast<char *>(
        allocator.allocate(dataSize, alignof(uint64_t)));

    // Setup a mutable array ref of our string refs so that we can update their
    // contents.
    auto mutableCopy = MutableArrayRef<llvm::StringRef>(
        reinterpret_cast<llvm::StringRef *>(rawData), numEntries);
    auto *stringData = rawData + numEntries * sizeof(llvm::StringRef);

    for (int i = 0; i < numEntries; i++) {
      memcpy(stringData, data[i].data(), data[i].size());
      mutableCopy[i] = llvm::StringRef(stringData, data[i].size());
      stringData += data[i].size();
    }

    copy = llvm::ArrayRef<llvm::StringRef>(
        reinterpret_cast<llvm::StringRef *>(rawData), numEntries);

    return new (allocator.allocate<DenseStringElementsV1AttrStorage>())
        DenseStringElementsV1AttrStorage(key.type, copy, key.isSplat);
  }

  llvm::ArrayRef<llvm::StringRef> data;
};

} // namespace detail
} // namespace serialization
} // namespace byre
} // namespace mlir

void DenseIntOrFPElementsV1Attr::print(mlir::AsmPrinter &os) const {
  os << '<'
     << DenseIntOrFPElementsAttr::getFromRawBuffer(
            mappingTypeFrom(getType()).cast<ShapedType>(), getData())
     << '>';
}

// Parse tensor elements using DenseIntOrFPElementsAttr printing.
Attribute DenseIntOrFPElementsV1Attr::parse(AsmParser &parser, mlir::Type) {
  DenseIntOrFPElementsAttr attr;
  if (failed(parser.parseLess()) || failed(parser.parseAttribute(attr)) ||
      failed(parser.parseGreater())) {
    return DenseIntOrFPElementsV1Attr();
  }
  return DenseIntOrFPElementsV1Attr::get(
      parser.getContext(), mappingTypeTo(attr.getType()), attr.getRawData());
}

void DenseArrayV1Attr::print(mlir::AsmPrinter &os) const {
  os << '<'
     << DenseArrayAttr::get(getContext(), mappingTypeFrom(getElementType()),
                            getSize(), getData())
     << '>';
}

Attribute DenseArrayV1Attr::parse(AsmParser &parser, mlir::Type) {
  DenseArrayAttr attr;
  if (failed(parser.parseLess()) || failed(parser.parseAttribute(attr)) ||
      failed(parser.parseGreater())) {
    return DenseArrayV1Attr();
  }
  return DenseArrayV1Attr::get(parser.getContext(),
                               mappingTypeTo(attr.getElementType()),
                               attr.getSize(), attr.getRawData());
}

void DenseStringElementsV1Attr::print(mlir::AsmPrinter &os) const {
  os << '<'
     << DenseStringElementsAttr::get(
            mappingTypeFrom(getType()).cast<ShapedType>(), getValue())
     << '>';
}

Attribute DenseStringElementsV1Attr::parse(AsmParser &parser, mlir::Type) {
  DenseStringElementsAttr attr;
  if (failed(parser.parseLess()) || failed(parser.parseAttribute(attr)) ||
      failed(parser.parseGreater())) {
    return DenseStringElementsV1Attr();
  }
  llvm::SmallVector<llvm::StringRef> values =
      llvm::SmallVector<llvm::StringRef>(
          attr.getValues<llvm::StringRef>().begin(),
          attr.getValues<llvm::StringRef>().end());
  return DenseStringElementsV1Attr::get(mappingTypeTo(attr.getType()),
                                        llvm::ArrayRef(values));
}

mlir::Type DenseStringElementsV1Attr::getType() const {
  return this->getImpl()->type;
}

llvm::ArrayRef<llvm::StringRef> DenseStringElementsV1Attr::getValue() const {
  return this->getImpl()->data;
}

namespace {

std::string dimSizeToString(int64_t dimSize) {
  if (ShapedType::isDynamic(dimSize))
    return "?";
  return std::to_string(dimSize);
}

void printShape(AsmPrinter &os, ArrayRef<int64_t> dimSizes) {
  if (dimSizes.empty())
    return;
  for (int64_t dimSize : dimSizes) {
    os << dimSizeToString(dimSize) << 'x';
  }
}

ParseResult parseShape(AsmParser &parser, SmallVector<int64_t> &dimSizes) {
  if (failed(parser.parseDimensionList(dimSizes))) {
    return failure();
  }
  return success();
}

void printMemorySpace(AsmPrinter &os, Attribute memorySpace) {
  if (!memorySpace)
    return;
  os << ", " << memorySpace;
}

ParseResult parseMemorySpace(AsmParser &parser, Attribute &memorySpace) {
  if (failed(parser.parseOptionalComma())) {
    return success();
  }
  if (failed(parser.parseAttribute(memorySpace))) {
    return failure();
  }
  return success();
}

void printEncoding(AsmPrinter &os, Attribute encoding) {
  if (!encoding)
    return;
  os << ", " << encoding;
}

ParseResult parseEncoding(AsmParser &parser, Attribute &encoding) {
  if (failed(parser.parseOptionalComma())) {
    return success();
  }
  if (failed(parser.parseAttribute(encoding))) {
    return failure();
  }
  return success();
}

void printTypeArray(AsmPrinter &os, ArrayRef<Type> typeArray) {
  if (typeArray.empty())
    os << "()";
  os << typeArray;
}

ParseResult parseTypeArray(AsmParser &parser, SmallVector<Type> &typeArray) {
  if (succeeded(parser.parseOptionalLParen()) &&
      succeeded(parser.parseOptionalRParen())) {
    return success();
  }

  auto parseEle = [&]() { return parser.parseType(typeArray.emplace_back()); };
  if (failed(parser.parseCommaSeparatedList(parseEle))) {
    return failure();
  }
  return success();
}

void printIntegerAttrV1(AsmPrinter &os, APInt value, mlir::Type type) {
  os << value << " : " << type;
}

ParseResult parseIntegerAttrV1(AsmParser &parser, APInt &value,
                               mlir::Type &type) {
  OptionalParseResult parseResult = parser.parseOptionalInteger(value);
  if (!parseResult.has_value() || failed(*parseResult)) {
    return failure();
  }
  if (succeeded(parser.parseColon()) && succeeded(parser.parseType(type))) {
    auto type_ = mappingTypeFrom(type);
    auto width = cast<IntegerType>(type_).getWidth();
    if (cast<IntegerType>(type_).isSignless() ||
        cast<IntegerType>(type_).isSigned()) {
      value = value.sextOrTrunc(width);
    } else {
      value = value.zextOrTrunc(width);
    }
    return success();
  }
  return failure();
}

void printFloatAttrV1(AsmPrinter &os, APFloat value, mlir::Type type) {
  os << value << " : " << type;
}

ParseResult parseFloatAttrV1(AsmParser &parser, mlir::FailureOr<APFloat> &value,
                             mlir::Type &type) {
  double value_;
  if (succeeded(parser.parseFloat(value_)) && succeeded(parser.parseColon()) &&
      succeeded(parser.parseType(type))) {
    bool losesInfo;
    APFloat floatValue(value_);
    floatValue.convert(
        cast<FloatType>(mappingTypeFrom(type)).getFloatSemantics(),
        APFloat::rmNearestTiesToEven, &losesInfo);
    value = FailureOr(floatValue);
    return success();
  }
  return failure();
}

void printArrayAttrV1(AsmPrinter &os, ArrayRef<Attribute> arrayAttr) {
  os << '[' << arrayAttr << ']';
}

ParseResult parseArrayAttrV1(AsmParser &parser,
                             SmallVector<Attribute> &arrayAttr) {
  ArrayAttr array;
  if (failed(parser.parseAttribute(array))) {
    return failure();
  }
  arrayAttr.append(array.begin(), array.end());
  return success();
}

void printDictionaryAttrV1(
    AsmPrinter &os,
    ArrayRef<std::pair<mlir::Attribute, mlir::Attribute>> values) {
  os << '{';
  llvm::interleaveComma(
      values, os, [&](auto nvp) { os << nvp.first << " = " << nvp.second; });
  os << '}';
}

ParseResult parseDictionaryAttrV1(
    AsmParser &parser,
    SmallVector<std::pair<mlir::Attribute, mlir::Attribute>> &values) {
  auto parseEle = [&]() {
    Attribute name;
    Attribute value;
    if (failed(parser.parseAttribute(name)) || failed(parser.parseEqual()) ||
        failed(parser.parseAttribute(value))) {
      return failure();
    }
    values.push_back({name, value});
    return success();
  };
  if (failed(parser.parseCommaSeparatedList(AsmParser::Delimiter::Braces,
                                            parseEle))) {
    return failure();
  }
  return success();
}

void printNestedReferences(AsmPrinter &os,
                           ArrayRef<mlir::Attribute> nestedReferences) {
  for (auto value : nestedReferences) {
    os << "::" << value;
  }
}

ParseResult
parseNestedReferences(AsmParser &parser,
                      SmallVector<mlir::Attribute> &nestedReferences) {
  while (succeeded(parser.parseOptionalColon())) {
    Attribute ref;
    if (failed(parser.parseColon()) || failed(parser.parseAttribute(ref))) {
      return failure();
    }

    nestedReferences.push_back(ref);
  }
  return success();
}

} // namespace

#include "byteir/Dialect/Byre/Serialization/ByreSerialDialect.cpp.inc"
#include "byteir/Dialect/Byre/Serialization/ByreSerialEnums.cpp.inc"

#include "byteir/Dialect/Byre/Serialization/ByreSerialTypeInterfaces.cpp.inc"
#define GET_TYPEDEF_CLASSES
#include "byteir/Dialect/Byre/Serialization/ByreSerialTypes.cpp.inc"

#include "byteir/Dialect/Byre/Serialization/ByreSerialAttrInterfaces.cpp.inc"
#define GET_ATTRDEF_CLASSES
#include "byteir/Dialect/Byre/Serialization/ByreSerialAttrs.cpp.inc"

#include "byteir/Dialect/Byre/Serialization/ByreSerialOpInterfaces.cpp.inc"
#define GET_OP_CLASSES
#include "byteir/Dialect/Byre/Serialization/ByreSerialOps.cpp.inc"
