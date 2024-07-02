//===- op_accessor.cc -----------------------------------------*--- C++ -*-===//
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

#include "brt/core/framework/op_accessor.h"

#include "brt/core/context/execution_frame.h"
#include "brt/core/framework/op_kernel_info.h"
#include "brt/core/ir/ir.h"
#include "brt/core/ir/util.h"
#include "byteir/Dialect/Byre/ByreDialect.h"

using namespace mlir;
using namespace brt::ir;

namespace brt {

size_t OpAccessor::GetNumArgs() const { return GetOpArgNum(info_); }

size_t OpAccessor::GetNumResults() const { return GetOpResultNum(info_); }

AsyncValueRef OpAccessor::GetArgAsWeight(size_t arg_idx) const {
  auto weight = GetWeightFromOpArgIndex(info_, arg_idx);
  // FIXME: Tentatively disable this check
  // (XRZ) Please check it
  // BRT_ENFORCE(weight, "argument " + std::to_string(arg_idx) + " is not
  // weight");
  return weight;
}

Shape OpAccessor::GetArgShape(size_t arg_idx) const {
  if (!frame_) {
    auto tensor = GetMLIRValueFromOpArgIndex(info_, arg_idx);
    return GetStaticShape(tensor).value().vec();
  } else {
    auto tensor_idx = GetTensorIndexFromOpArgIndex(info_, arg_idx);
    return frame_->GetShape(tensor_idx);
  }
}

DTypeEnum OpAccessor::GetArgDTypeEnum(size_t arg_idx) const {
  auto tensor = GetMLIRValueFromOpArgIndex(info_, arg_idx);
  return GetElementDTypeEnum(tensor);
}

bool OpAccessor::HasAttr(const std::string &name) const {
  return info_.GetOperation()->hasAttr(name);
}

bool OpAccessor::GetAttrAsBool(const std::string &name) const {
  if (auto attr = info_.GetOperation()->getAttrOfType<BoolAttr>(name)) {
    return attr.getValue();
  }
  BRT_THROW("Attribute " + name + " is not set");
}

int64_t OpAccessor::GetAttrAsInt(const std::string &name) const {
  if (auto attr = info_.GetOperation()->getAttrOfType<IntegerAttr>(name)) {
    return attr.getInt();
  }
  BRT_THROW("Attribute " + name + " is not set");
}

float OpAccessor::GetAttrAsFloat(const std::string &name) const {
  if (auto attr = info_.GetOperation()->getAttrOfType<FloatAttr>(name)) {
    return attr.getValueAsDouble();
  }
  BRT_THROW("Attribute " + name + " is not set");
}

std::string OpAccessor::GetAttrAsString(const std::string &name) const {
  if (auto attr = info_.GetOperation()->getAttrOfType<StringAttr>(name)) {
    return attr.getValue().str();
  }
  BRT_THROW("Attribute " + name + " is not set");
}

DTypeEnum OpAccessor::GetAttrAsType(const std::string &name) const {
  if (auto attr = info_.GetOperation()->getAttrOfType<TypeAttr>(name)) {
    return ConvertMLIRTypeToDType(attr.getValue());
  }
  BRT_THROW("Attribute " + name + " is not set");
}

std::vector<int64_t>
OpAccessor::GetAttrAsIntArray(const std::string &name) const {
  if (auto attrArray = info_.GetOperation()->getAttrOfType<ArrayAttr>(name)) {
    // check if attribute is an array of IntegerAttr
    std::vector<int64_t> ret;
    for (auto &&i : attrArray.getValue()) {
      if (auto attr = dyn_cast<IntegerAttr>(i)) {
        ret.push_back(attr.getInt());
      } else {
        BRT_THROW("Cannot cast " + name + " to array of IntAttr");
      }
    }
    return ret;
  } else if (auto attrArray =
                 info_.GetOperation()->getAttrOfType<DenseIntElementsAttr>(
                     name)) {
    // check if attribute is a dense tensor with IntType
    std::vector<int64_t> ret;
    for (auto &&i : attrArray) {
      ret.push_back(i.getSExtValue());
    }
    return ret;
  }
  BRT_THROW("Attribute " + name + " is not set");
}

template <typename T>
bool OpAccessor::HasAttrOfSplatValue(const std::string &name) const {
  if constexpr (std::is_integral<T>::value) {
    if (auto attr =
            info_.GetOperation()->getAttrOfType<DenseIntElementsAttr>(name)) {
      return attr.isSplat();
    }
  } else if constexpr (std::is_floating_point<T>::value) {
    if (auto attr =
            info_.GetOperation()->getAttrOfType<DenseFPElementsAttr>(name)) {
      return attr.isSplat();
    }
  } else if constexpr (std::is_same_v<
                           T, DTypeTraits<DTypeEnum::StringView>::type_t>) {
    if (auto attr =
            info_.GetOperation()->getAttrOfType<DenseStringElementsAttr>(
                name)) {
      return attr.isSplat();
    }
  } else {
    static_assert(!std::is_same_v<T, T>, "unsupported type of splat value");
  }
  return false;
}

template <typename T>
T OpAccessor::GetAttrAsSplatValue(const std::string &name) const {
  if constexpr (std::is_integral<T>::value) {
    if (auto attr =
            info_.GetOperation()->getAttrOfType<DenseIntElementsAttr>(name)) {
      if (attr.isSplat()) {
        mlir::IntegerAttr splatValue = attr.getSplatValue<IntegerAttr>();
        mlir::Type type = splatValue.getType();
        if (type.getIntOrFloatBitWidth() == 1 || type.isUnsignedInteger()) {
          return splatValue.getValue().getZExtValue();
        } else {
          return splatValue.getValue().getSExtValue();
        }
      }
    }
  } else if constexpr (std::is_floating_point<T>::value) {
    if (auto attr =
            info_.GetOperation()->getAttrOfType<DenseFPElementsAttr>(name)) {
      if (attr.isSplat()) {
        return attr.getSplatValue<FloatAttr>().getValueAsDouble();
      }
    }
  } else if constexpr (std::is_same_v<
                           T, DTypeTraits<DTypeEnum::StringView>::type_t>) {
    if (auto attr =
            info_.GetOperation()->getAttrOfType<DenseStringElementsAttr>(
                name)) {
      if (attr.isSplat()) {
        llvm::StringRef str = attr.getSplatValue<StringAttr>().getValue();
        return StringView(str.data(), str.size());
      }
    }
  } else {
    static_assert(!std::is_same_v<T, T>, "unsupported type of splat value");
  }
  BRT_THROW("Attribute " + name + " is not set");
}

// GetDenseAttrAsVector will iterate every elements in dense attibutes.
// If you want to avoid iterating, consider use getRawData() but special handle
// for i1 ???
template <typename T>
std::vector<T> OpAccessor::GetAttrAsVector(const std::string &name) const {
  std::vector<T> results;
  if (auto attr =
          info_.GetOperation()->getAttrOfType<DenseIntElementsAttr>(name)) {
    results.reserve(attr.size());
    for (APInt &&i : attr) {
      results.push_back(static_cast<T>(i.getSExtValue()));
    }
    return results;
  } else if (auto attr =
                 info_.GetOperation()->getAttrOfType<DenseFPElementsAttr>(
                     name)) {
    results.reserve(attr.size());
    for (APFloat &&i : attr) {
      results.push_back(static_cast<T>(i.convertToDouble()));
    }
    return results;
  }
  BRT_THROW("Attribute " + name + " is not supported to get as vector");
}

void *OpAccessor::GetAttrAsVoidPtr(const std::string &name) const {
  if (auto attr = info_.GetOperation()->getAttrOfType<ArrayAttr>(name)) {
    size_t totalSize = 0;
    for (Attribute elementAttr : attr) {
      if (auto floatAttr = dyn_cast<FloatAttr>(elementAttr)) {
        totalSize += sizeof(float);
      } else if (auto intAttr = dyn_cast<IntegerAttr>(elementAttr)) {
        totalSize += sizeof(int64_t);
      } else {
        // TODO: support string
        BRT_THROW("Not all elements can be converted to void * for attribute" +
                  name);
      }
    }
    void *result = malloc(totalSize);
    int ptr = 0;
    for (Attribute elementAttr : attr) {
      if (auto floatAttr = dyn_cast<FloatAttr>(elementAttr)) {
        float val = floatAttr.getValueAsDouble();
        std::memcpy(static_cast<char *>(result) + ptr, &val, sizeof(float));
        ptr += sizeof(float);
      } else if (auto intAttr = dyn_cast<IntegerAttr>(elementAttr)) {
        int64_t val = intAttr.getInt();
        std::memcpy(static_cast<char *>(result) + ptr, &val, sizeof(int64_t));
        ptr += sizeof(int64_t);
      }
    }
    return result;
  }
  BRT_THROW("Attribute " + name + " is not supported to get as void *");
}

std::string OpAccessor::GetUID() const {
  auto byre_op = llvm::cast<byre::ByreOp>(info_.GetOperation());
  return ByREHandle::GetOpUID(byre_op);
}

int64_t OpAccessor::GetNumElementsOfShape(const Shape &shape) {
  return LinearizedStaticShape(shape).value();
}

AsyncValueRef OpAccessor::GetArgAsyncValueRef(size_t arg_idx) const {
  EnsureFrame("GetArgAsyncValueRef");
  auto tensor_idx = GetTensorIndexFromOpArgIndex(info_, arg_idx);
  return frame_->GetAsyncValueRef(tensor_idx);
}

template <typename T> T OpAccessor::GetArgScalar(size_t arg_idx) {
  auto scalar_idx = GetScalarIndexFromMLIRValue(
      info_, info_.GetOperation()->getOperand(arg_idx));
  return frame_->GetScalar<T>(scalar_idx);
}

template <typename T>
common::Status OpAccessor::SetResultScalar(size_t result_idx, const T &scalar) {
  auto scalar_idx = GetScalarIndexFromMLIRValue(
      info_, info_.GetOperation()->getResult(result_idx));
  return frame_->SetScalar(scalar_idx, scalar);
}

#define INST_ATTR_METH(T)                                                      \
  template bool OpAccessor::HasAttrOfSplatValue<T>(const std::string &) const; \
  template T OpAccessor::GetAttrAsSplatValue<T>(const std::string &) const;
INST_ATTR_METH(int64_t)
INST_ATTR_METH(double)
INST_ATTR_METH(StringView)
#undef INST_ATTR_METH

#define INST_DENSE_ATTR_METH(T)                                                \
  template std::vector<T> OpAccessor::GetAttrAsVector<T>(const std::string &)  \
      const;
INST_DENSE_ATTR_METH(float)
INST_DENSE_ATTR_METH(int32_t)
INST_DENSE_ATTR_METH(int64_t)
INST_DENSE_ATTR_METH(uint8_t)
INST_DENSE_ATTR_METH(uint32_t)
INST_DENSE_ATTR_METH(double)
INST_DENSE_ATTR_METH(half_float::half)
#undef INST_DENSE_ATTR_METH

#define INST_SCALAR_METH(T)                                                    \
  template T OpAccessor::GetArgScalar<T>(size_t);                              \
  template common::Status OpAccessor::SetResultScalar(size_t result_idx,       \
                                                      const T &scalar);
INST_SCALAR_METH(float)
INST_SCALAR_METH(int32_t)
INST_SCALAR_METH(int64_t)
INST_SCALAR_METH(uint8_t)
INST_SCALAR_METH(uint32_t)
INST_SCALAR_METH(double)
#undef INST_SCALAR_METH

} // namespace brt
