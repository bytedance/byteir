//===- ShapeAnalysis.cpp --------------------------------------------------===//
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

#include "byteir/Analysis/ShapeAnalysis.h"

using namespace mlir::dataflow;
using namespace mlir::shape_analysis;

namespace mlir {
namespace shape_analysis {

StaticShapeKnowledge StaticShapeKnowledge::getKnowledgeFromType(Type type) {
  StaticShapeKnowledge result = getPessimisticValueState();
  if (auto shapedType = dyn_cast_or_null<ShapedType>(type)) {
    if (shapedType.hasRank()) {
      result.hasRank = true;
      result.sizes.reserve(shapedType.getRank());
      for (auto dim : shapedType.getShape())
        result.sizes.push_back(dim);
    }
    result.dtype = shapedType.getElementType();
  }
  return result;
}

StaticShapeKnowledge StaticShapeKnowledge::getPessimisticValueState() {
  return StaticShapeKnowledge(false, {}, Type());
}

StaticShapeKnowledge
StaticShapeKnowledge::getPessimisticValueState(Value value) {
  if (value) {
    return getKnowledgeFromType(value.getType());
  }
  return getPessimisticValueState();
}

StaticShapeKnowledge
StaticShapeKnowledge::join(const StaticShapeKnowledge &lhs,
                           const StaticShapeKnowledge &rhs) {
  StaticShapeKnowledge result = getPessimisticValueState();
  result.hasError = true;

  // if ((lhs.dtype.has_value() && !lhs.dtype.value()) ||
  //     (rhs.dtype.has_value() && !rhs.dtype.value()) ||
  //     (lhs.dtype.has_value() && rhs.dtype.has_value() &&
  //      lhs.dtype.value() != rhs.dtype.value()))
  //   return result;
  if (lhs.isUninitialized())
    return rhs;
  if (rhs.isUninitialized())
    return lhs;
  // Early termination when dtype is nullptr
  // or not identical
  if (!(*lhs.dtype) || !(*rhs.dtype) || *lhs.dtype != *rhs.dtype)
    return result;

  result.hasError = false;
  result.dtype = lhs.dtype;

  if (!lhs.hasRank || !rhs.hasRank) {
    result.hasRank = false;
    return result;
  }

  if (lhs.sizes.size() != rhs.sizes.size()) {
    result.hasRank = false;
    return result;
  }

  result.hasRank = true;
  result.sizes.resize(lhs.sizes.size(), ShapedType::kDynamic);
  for (int i = 0, e = lhs.sizes.size(); i < e; i++) {
    if (lhs.sizes[i] == rhs.sizes[i]) {
      result.sizes[i] = lhs.sizes[i];
    }
  }

  return result;
}

StaticShapeKnowledge
StaticShapeKnowledge::meet(const StaticShapeKnowledge &lhs,
                           const StaticShapeKnowledge &rhs) {
  StaticShapeKnowledge result = getPessimisticValueState();
  result.hasError = true;

  // if ((lhs.dtype.has_value() && !lhs.dtype.value()) ||
  //     (rhs.dtype.has_value() && !rhs.dtype.value()) ||
  //     (lhs.dtype.has_value() && rhs.dtype.has_value() &&
  //      lhs.dtype.value() != rhs.dtype.value()))
  //   return result;
  if (lhs.isUninitialized())
    return rhs;
  if (rhs.isUninitialized())
    return lhs;
  // Early termination when dtype is nullptr
  // or not identical
  if (!(*lhs.dtype) || !(*rhs.dtype) || *lhs.dtype != *rhs.dtype)
    return result;

  result.hasError = false;
  result.dtype = lhs.dtype;

  if (!lhs.hasRank && !rhs.hasRank)
    return result;

  if (!rhs.hasRank) {
    result.hasRank = true;
    result.sizes = lhs.sizes;
    return result;
  }

  if (!lhs.hasRank) {
    result.hasRank = true;
    result.sizes = rhs.sizes;
    return result;
  }

  if (lhs.sizes.size() != rhs.sizes.size())
    return result;

  result.hasRank = true;
  result.sizes.resize(lhs.sizes.size(), ShapedType::kDynamic);
  for (auto i : llvm::seq<unsigned>(0, result.sizes.size())) {
    int64_t lhsSize = lhs.sizes[i];
    int64_t rhsSize = rhs.sizes[i];
    int64_t &resultSize = result.sizes[i];
    if (lhsSize == ShapedType::kDynamic) {
      resultSize = rhsSize;
    } else if (rhsSize == ShapedType::kDynamic) {
      resultSize = lhsSize;
    } else if (lhsSize == rhsSize) {
      resultSize = lhsSize;
    } else {
      result.hasError = true;
    }
  }

  return result;
}

void StaticShapeKnowledge::print(raw_ostream &os) const {
  if (hasError || !dtype) {
    os << "None\n";
  } else if (!(*dtype)) {
    os << "Unknown\n";
  } else {
    os << getType() << "\n";
  }
}

BoundedShapeKnowledge BoundedShapeKnowledge::getKnowledgeFromType(Type type) {
  BoundedShapeKnowledge result = getPessimisticValueState();
  if (auto shapedType = dyn_cast_or_null<ShapedType>(type)) {
    if (shapedType.hasRank()) {
      result.hasRank = true;
      result.sizes.reserve(shapedType.getRank());
      for (auto dim : shapedType.getShape())
        result.sizes.push_back(dim);
    }
    result.dtype = shapedType.getElementType();
  }
  return result;
}

BoundedShapeKnowledge BoundedShapeKnowledge::getPessimisticValueState() {
  return BoundedShapeKnowledge(false, {}, Type());
}

BoundedShapeKnowledge
BoundedShapeKnowledge::getPessimisticValueState(Value value) {
  if (value) {
    return getKnowledgeFromType(value.getType());
  }
  return getPessimisticValueState();
}

BoundedShapeKnowledge
BoundedShapeKnowledge::join(const BoundedShapeKnowledge &lhs,
                            const BoundedShapeKnowledge &rhs) {
  BoundedShapeKnowledge result = getPessimisticValueState();
  result.hasError = true;

  if (lhs.isUninitialized())
    return rhs;
  if (rhs.isUninitialized())
    return lhs;
  // Early termination when dtype is nullptr
  // or not identical
  if (!(*lhs.dtype) || !(*rhs.dtype) || *lhs.dtype != *rhs.dtype)
    return result;

  result.hasError = false;
  result.dtype = lhs.dtype;

  if (!lhs.hasRank || !rhs.hasRank) {
    result.hasRank = false;
    return result;
  }

  if (lhs.sizes.size() != rhs.sizes.size()) {
    result.hasRank = false;
    return result;
  }

  result.hasRank = true;
  result.sizes.resize(lhs.sizes.size(), ShapedType::kDynamic);
  for (int i = 0, e = lhs.sizes.size(); i < e; i++) {
    if (lhs.sizes[i] == rhs.sizes[i]) {
      result.sizes[i] = lhs.sizes[i];
    } else if (lhs.sizes[i] != ShapedType::kDynamic &&
               rhs.sizes[i] != ShapedType::kDynamic) {
      result.sizes[i] =
          (lhs.sizes[i] > rhs.sizes[i]) ? lhs.sizes[i] : rhs.sizes[i];
    }
  }

  return result;
}

BoundedShapeKnowledge
BoundedShapeKnowledge::meet(const BoundedShapeKnowledge &lhs,
                            const BoundedShapeKnowledge &rhs) {
  BoundedShapeKnowledge result = getPessimisticValueState();
  result.hasError = true;

  // if ((lhs.dtype.has_value() && !lhs.dtype.value()) ||
  //     (rhs.dtype.has_value() && !rhs.dtype.value()) ||
  //     (lhs.dtype.has_value() && rhs.dtype.has_value() &&
  //      lhs.dtype.value() != rhs.dtype.value()))
  //   return result;
  if (lhs.isUninitialized())
    return rhs;
  if (rhs.isUninitialized())
    return lhs;
  // Early termination when dtype is nullptr
  // or not identical
  if (!(*lhs.dtype) || !(*rhs.dtype) || *lhs.dtype != *rhs.dtype)
    return result;

  result.hasError = false;
  result.dtype = lhs.dtype;

  if (!lhs.hasRank && !rhs.hasRank)
    return result;

  if (!rhs.hasRank) {
    result.hasRank = true;
    result.sizes = lhs.sizes;
    return result;
  }

  if (!lhs.hasRank) {
    result.hasRank = true;
    result.sizes = rhs.sizes;
    return result;
  }

  if (lhs.sizes.size() != rhs.sizes.size())
    return result;

  result.hasRank = true;
  result.sizes.resize(lhs.sizes.size(), ShapedType::kDynamic);
  for (auto i : llvm::seq<unsigned>(0, result.sizes.size())) {
    int64_t lhsSize = lhs.sizes[i];
    int64_t rhsSize = rhs.sizes[i];
    int64_t &resultSize = result.sizes[i];
    if (lhsSize == ShapedType::kDynamic) {
      resultSize = rhsSize;
    } else if (rhsSize == ShapedType::kDynamic) {
      resultSize = lhsSize;
    } else if (lhsSize == rhsSize) {
      resultSize = lhsSize;
    } else {
      result.hasError = true;
    }
  }

  return result;
}
} // namespace shape_analysis
} // namespace mlir
