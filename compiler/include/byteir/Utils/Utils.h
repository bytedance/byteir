//===- Utils.h ------------------------------------------------------------===//
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

#ifndef BYTEIR_UTILS_UTILS_H
#define BYTEIR_UTILS_UTILS_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <functional>
#include <optional>
#include <string>
#include <type_traits>

namespace mlir {
class Location;
class Operation;
class Value;

namespace func {
class CallOp;
class FuncOp;
} // namespace func

// Return literal from a constant-like value
// return std::nullopt if not applicable
std::optional<int64_t> getLiteralFromConstantLike(Value);

std::optional<Attribute> getAttrFromConstantLike(Value);

// Return literal from a constant-like value,
// return defaultLit if not applicable
int64_t getLiteralFromConstantLike(Value, int64_t defaultLit);

// Return literals from a list of constant-like values
llvm::SmallVector<int64_t, 4>
getLiteralsFromConstantLikes(ArrayRef<Value> values, int64_t defaultLit);

// Create a vector with only the offset as 1, the rest as 0's.
// e.g. if offset == 1, size == 4, val == 3, return3 [0, 3, 0, 0]
llvm::SmallVector<int64_t, 4> createOneHot(unsigned size, unsigned offset,
                                           int64_t val = 1);

// Return all indices for non-zeros
llvm::SmallVector<unsigned, 4> getAllIndicesForNonZeros(ArrayRef<int64_t>);

// Return true when a value is a ConstantIndex with value of `lit`.
bool isConstantIndex(Value value, int64_t lit);

// Check whether an attribute is zero
// If an attribute contain multiple sub attributes,
// it will check all of sub attributes.
bool isZeroAttribute(Attribute value);

// Check whether an attribute is SmallestAttr
// If an attribute contain multiple sub attributes,
// it will check all of sub attributes.
bool isMinValueAttribute(Attribute value);

// Check whether an attribute is LargestAttr
// If an attribute contain multiple sub attributes,
// it will check all of sub attributes.
bool isMaxValueAttribute(Attribute value);

// Returns true if the given `attr` is a splat value and is `value`.
bool isSplatValue(DenseIntElementsAttr attr, int64_t value);

// Returns true if the given `attr` is a splat value as the given `value`.
bool isSplatValue(DenseFPElementsAttr attr, double value);

inline bool isSplatElementsAttribute(DenseIntOrFPElementsAttr attr,
                                     int64_t intValue, double doubleValue) {
  if (attr.isa<DenseIntElementsAttr>()) {
    return isSplatValue(cast<DenseIntElementsAttr>(attr), intValue);
  } else if (attr.isa<DenseFPElementsAttr>()) {
    return isSplatValue(cast<DenseFPElementsAttr>(attr), doubleValue);
  }
  assert(false && "attr must be DenseIntElementsAttr or DenseFPElementsAttr");
}

// Returns true if the given `attr` is a splat value and close to `value`.
bool isSplatCloseToValue(DenseFPElementsAttr attr, double value,
                         double EPSILON = 0.00001);

// extract values in `attr` to `arrayValues`
void getValuesFromDenseIntElementsAttr(DenseIntElementsAttr attr,
                                       SmallVector<int64_t> &arrayValues);

// Returns nD 64-bit dense elements attribute with the given values.
inline DenseIntElementsAttr getI64ElementsAttr(ArrayRef<int64_t> values,
                                               ArrayRef<int64_t> shape,
                                               Builder *builder) {
  RankedTensorType ty =
      RankedTensorType::get(shape, builder->getIntegerType(64));
  return DenseIntElementsAttr::get(ty, values);
}

// Return a placeholder name of an attribute
// to avoid breaking the verifier of the original attribute
// by adding some unique prefix or postfix
inline std::string getAttrPlaceholderName(StringRef name) {
  return "__placeholder__" + name.str();
}

// Remove placeholder of attribute names
// Note: it removes placeholder tag only.
// The attribute name will become the original name
// Note: the input is a list of original names
void removeAttrPlaceholders(mlir::Operation *op, ArrayRef<StringRef> Orignames);

// Remove placeholder of arg attribute names
// Note: it removes placeholder tag only.
// The attribute name will become the original name
// Note: the input is a list of original arg attribute names
template <typename OpTy>
std::enable_if_t<OpTy::template hasTrait<FunctionOpInterface::Trait>()>
removeArgAttrPlaceholders(OpTy op, ArrayRef<StringRef> argAttrNames) {
  for (size_t idx = 0; idx < op.getNumArguments(); ++idx) {
    for (const auto &name : argAttrNames) {
      auto placeholder = getAttrPlaceholderName(name);
      auto attr = op.getArgAttr(idx, placeholder);
      if (attr == nullptr) {
        continue;
      }

      op.setArgAttr(idx, name, attr);
      op.removeArgAttr(idx, placeholder);
    }
  }
}

// Return FuncOp from a CallOp
mlir::func::FuncOp getFuncOp(func::CallOp);

// Return true if attrs has any of filterAttrs
bool hasAnyOfAttrs(llvm::ArrayRef<mlir::NamedAttribute> attrs,
                   llvm::ArrayRef<llvm::StringRef> filterAttrs);

// add `attrs` into an operation
void addAttrs(mlir::Operation *, llvm::ArrayRef<mlir::NamedAttribute> attrs);

std::optional<unsigned> findOperandIndex(mlir::Operation *, mlir::Value);

std::optional<unsigned> findResultIndex(mlir::Operation *, mlir::Value);

SmallVector<Value, 4>
getInputsOfCluster(const llvm::SmallVector<Operation *, 8> &cluster);

SmallVector<Value, 4> getOutputsOfCluster(
    const llvm::SmallVector<Operation *, 8> &cluster,
    const llvm::DenseMap<Value, int64_t> *outputStats = nullptr);

// return true, if memref is only used in op in the filters, or alloc or dealloc
bool isMemrefTrivial(mlir::Value memref,
                     llvm::ArrayRef<mlir::Operation *> filters);

// deep check an op satisfy check function
// e.g. deep check an op's leaves are all constant
bool deepCheck(Operation *op, std::function<bool(mlir::Operation *)> checkFunc);

// deep check an op satisfy check function
// allow a memory to store previous result for accelerating
bool deepCheckWithMemory(Operation *op,
                         std::function<bool(mlir::Operation *)> checkFunc,
                         llvm::DenseMap<Operation *, bool> &memory);

// count number of a value is used
// if a value is used twice by a user, it will count twice
inline int useCount(Value val) {
  return static_cast<int>(
      std::distance(val.getUses().begin(), val.getUses().end()));
}
// count number of users a value has
// if a value is used twice by a user, it will count one
int userCount(Value val);

inline Location getFusedLoc(ArrayRef<Operation *> ops, OpBuilder &op_builder) {
  llvm::SmallVector<Location> locs;
  for (auto op : ops) {
    locs.push_back(op->getLoc());
  }
  return op_builder.getFusedLoc(locs);
}

/// Returns a `memref.dim` or `tensor.dim` operation to get the shape of `v` at
/// `dim`.
Value getDimValue(OpBuilder &builder, Location loc, Value v, int64_t dim);

/// Returns a `memref.dim` or `tensor.dim` operation to get the shape of `v` at
/// `dim`. If the shape is constant, returns the shape as an `IntegerAttr`.
OpFoldResult getDim(OpBuilder &builder, Location loc, Value v, int64_t dim);

/// Returns a memref.subview or a tensor.extract_slice based on the type of the
/// `source`.
Value getSlice(OpBuilder &b, Location loc, Value source,
               ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes,
               ArrayRef<OpFoldResult> strides);

/// Given a type of OpFoldResult, try to extract a constant Attribute if it's a
/// Value. If not, return the original ofr.
OpFoldResult canonicalizeOpFoldResult(OpFoldResult ofr,
                                      bool enableFold = false);

/// Given an array of OpFoldResult, try to extract a constant Attribute from
/// each value if it's a Value. If not, return the original ofr.
SmallVector<OpFoldResult> canonicalizeOpFoldResult(ArrayRef<OpFoldResult> ofrs,
                                                   bool enableFold = false);

// Return true if block contains single op
template <typename... Ops> bool isBlockSingleOp(Block *block) {
  if (block == nullptr)
    return false;

  Operation *retOp = block->getTerminator();
  if (retOp->getNumOperands() != 1)
    return false;

  auto computeOp = retOp->getOperand(0).getDefiningOp();
  if (computeOp && (isa<Ops>(computeOp) || ...)) {
    return (computeOp->getOperand(0) == block->getArgument(0) &&
            computeOp->getOperand(1) == block->getArgument(1)) ||
           (computeOp->getOperand(0) == block->getArgument(1) &&
            computeOp->getOperand(1) == block->getArgument(0));
  }

  return false;
}

// get all RetOp ops within an op's nested regions
template <typename RetOp, WalkOrder Order = WalkOrder::PostOrder>
SmallVector<RetOp, 4> getOpsNested(Operation *op) {
  SmallVector<RetOp, 4> res;
  op->walk<Order>([&](RetOp ret) { res.push_back(ret); });
  return res;
}

} // namespace mlir

#endif // BYTEIR_UTILS_UTILS_H
