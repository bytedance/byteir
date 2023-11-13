//===- Serialization.cpp -------------------------------------------------===//
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

#include "byteir/Dialect/Byre/Serialization.h"
#include "byteir/Dialect/Ace/AceDialect.h"
#include "byteir/Dialect/Byre/ByreDialect.h"
#include "byteir/Dialect/Byre/Serialization/ByreSerialOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::byre;
using namespace mlir::byre::serialization;

namespace {
// mapping attribute to serialized type
Attribute mappingAttrTo(Attribute attr);
// mapping attribute from serialized type
Attribute mappingAttrFrom(Attribute attr);

Attribute mappingAttrTo(Attribute attr) {
  if (!attr)
    return Attribute();

  if (llvm::isa<SerializableAttrInterface>(attr))
    return attr;

  auto ctx = attr.getContext();
  if (auto integerAttr = llvm::dyn_cast<IntegerAttr>(attr)) {
    return IntegerV1Attr::get(ctx, mappingTypeTo(integerAttr.getType()),
                              integerAttr.getValue());
  }
  if (auto floatAttr = llvm::dyn_cast<FloatAttr>(attr)) {
    return FloatV1Attr::get(ctx, mappingTypeTo(floatAttr.getType()),
                            floatAttr.getValue());
  }
  if (auto unitAttr = llvm::dyn_cast<UnitAttr>(attr)) {
    return UnitV1Attr::get(ctx);
  }
  if (auto arrayAttr = llvm::dyn_cast<ArrayAttr>(attr)) {
    llvm::SmallVector<mlir::Attribute> values;
    for (auto i : arrayAttr.getValue()) {
      values.push_back(mappingAttrTo(i));
    }
    return ArrayV1Attr::get(ctx, values);
  }
  if (auto dictAttr = llvm::dyn_cast<DictionaryAttr>(attr)) {
    llvm::SmallVector<std::pair<mlir::Attribute, mlir::Attribute>> serialAttrs;
    for (NamedAttribute i : dictAttr.getValue()) {
      serialAttrs.push_back(
          {mappingAttrTo(i.getName()), mappingAttrTo(i.getValue())});
    }
    return DictionaryV1Attr::get(ctx, serialAttrs);
  }
  if (auto stringAttr = llvm::dyn_cast<StringAttr>(attr)) {
    return StringV1Attr::get(ctx, stringAttr.getValue());
  }
  if (auto typeAttr = llvm::dyn_cast<TypeAttr>(attr)) {
    return TypeV1Attr::get(ctx, mappingTypeTo(typeAttr.getValue()));
  }
  if (auto symbolRefAttr = llvm::dyn_cast<SymbolRefAttr>(attr)) {
    llvm::SmallVector<mlir::Attribute> nestedReferences;
    for (auto i : symbolRefAttr.getNestedReferences()) {
      nestedReferences.push_back(mappingAttrTo(i));
    }
    return SymbolRefV1Attr::get(
        ctx, mappingAttrTo(symbolRefAttr.getRootReference()), nestedReferences);
  }
  if (auto tensorAttr = llvm::dyn_cast<DenseIntOrFPElementsAttr>(attr)) {
    return DenseIntOrFPElementsV1Attr::get(
        ctx, mappingTypeTo(tensorAttr.getType()), tensorAttr.getRawData());
  }
  if (auto denseStringAttr = llvm::dyn_cast<DenseStringElementsAttr>(attr)) {
    llvm::SmallVector<llvm::StringRef> values =
        llvm::SmallVector<llvm::StringRef>(
            denseStringAttr.getValues<llvm::StringRef>().begin(),
            denseStringAttr.getValues<llvm::StringRef>().end());
    return DenseStringElementsV1Attr::get(
        mappingTypeTo(denseStringAttr.getType()), values);
  }
  if (auto denseArrayAttr = llvm::dyn_cast<DenseArrayAttr>(attr)) {
    return DenseArrayV1Attr::get(
        ctx, mappingTypeTo(denseArrayAttr.getElementType()),
        denseArrayAttr.getSize(), denseArrayAttr.getRawData());
  }

  return Attribute();
}

Attribute mappingAttrFrom(Attribute attr) {
  if (!attr)
    return Attribute();

  auto ctx = attr.getContext();
  if (auto argType = llvm::dyn_cast<ArgTypeV1Attr>(attr)) {
    auto argTypeValue = argType.getValue();
    auto newArgTypeValue = EntryFuncArgType::None;
    if (bitEnumContainsAll(argTypeValue, ArgTypeV1::Input)) {
      newArgTypeValue = newArgTypeValue | EntryFuncArgType::Input;
    }
    if (bitEnumContainsAll(argTypeValue, ArgTypeV1::Output)) {
      newArgTypeValue = newArgTypeValue | EntryFuncArgType::Output;
    }
    if (bitEnumContainsAll(argTypeValue, ArgTypeV1::Weight)) {
      newArgTypeValue = newArgTypeValue | EntryFuncArgType::Weight;
    }
    return EntryFuncArgTypeAttr::get(ctx, newArgTypeValue);
  }
  if (auto memoryEffect = llvm::dyn_cast<MemoryEffectV1Attr>(attr)) {
    auto memoryEffectValue = memoryEffect.getValue();
    auto newMemoryEffectValue = MemoryEffect::None;
    if (bitEnumContainsAll(memoryEffectValue, MemoryEffectV1::Read)) {
      newMemoryEffectValue = newMemoryEffectValue | MemoryEffect::Read;
    }
    if (bitEnumContainsAll(memoryEffectValue, MemoryEffectV1::Write)) {
      newMemoryEffectValue = newMemoryEffectValue | MemoryEffect::Write;
    }
    return MemoryEffectAttr::get(ctx, newMemoryEffectValue);
  }

  if (auto integerAttr = llvm::dyn_cast<IntegerV1Attr>(attr)) {
    return IntegerAttr::get(mappingTypeFrom(integerAttr.getType()),
                            integerAttr.getValue());
  }
  if (auto floatAttr = llvm::dyn_cast<FloatV1Attr>(attr)) {
    return FloatAttr::get(mappingTypeFrom(floatAttr.getType()),
                          floatAttr.getValue());
  }
  if (auto unitAttr = llvm::dyn_cast<UnitV1Attr>(attr)) {
    return UnitAttr::get(ctx);
  }
  if (auto arrayAttr = llvm::dyn_cast<ArrayV1Attr>(attr)) {
    llvm::SmallVector<mlir::Attribute> values;
    for (auto i : arrayAttr.getValue()) {
      values.push_back(mappingAttrFrom(i));
    }
    return ArrayAttr::get(ctx, values);
  }
  if (auto dictAttr = llvm::dyn_cast<DictionaryV1Attr>(attr)) {
    llvm::SmallVector<NamedAttribute> attrs;
    for (auto p : dictAttr.getValue()) {
      attrs.push_back(
          NamedAttribute(mappingAttrFrom(p.first).cast<StringAttr>(),
                         mappingAttrFrom(p.second)));
    }
    return DictionaryAttr::get(ctx, attrs);
  }
  if (auto stringAttr = llvm::dyn_cast<StringV1Attr>(attr)) {
    return StringAttr::get(ctx, stringAttr.getValue());
  }
  if (auto typeAttr = llvm::dyn_cast<TypeV1Attr>(attr)) {
    return TypeAttr::get(mappingTypeFrom(typeAttr.getValue()));
  }
  if (auto symbolRefAttr = llvm::dyn_cast<SymbolRefV1Attr>(attr)) {
    mlir::StringAttr rootReferences =
        mappingAttrFrom(symbolRefAttr.getRootReference()).cast<StringAttr>();
    llvm::SmallVector<mlir::FlatSymbolRefAttr> nestedReferences;
    for (auto i : symbolRefAttr.getNestedReferences()) {
      nestedReferences.push_back(
          SymbolRefAttr::get(mappingAttrFrom(i).cast<StringAttr>()));
    }
    return SymbolRefAttr::get(ctx, rootReferences, nestedReferences);
  }
  if (auto tensorAttr = llvm::dyn_cast<DenseIntOrFPElementsV1Attr>(attr)) {
    auto rankedTensorType =
        mappingTypeFrom(tensorAttr.getType()).cast<RankedTensorType>();
    return DenseIntOrFPElementsAttr::getFromRawBuffer(rankedTensorType,
                                                      tensorAttr.getData());
  }
  if (auto denseStringAttr = llvm::dyn_cast<DenseStringElementsV1Attr>(attr)) {
    return DenseStringElementsAttr::get(
        mappingTypeFrom(denseStringAttr.getType()).cast<ShapedType>(),
        denseStringAttr.getValue());
  }
  if (auto denseArrayAttr = llvm::dyn_cast<DenseArrayV1Attr>(attr)) {
    return DenseArrayAttr::get(
        ctx, mappingTypeFrom(denseArrayAttr.getElementType()),
        denseArrayAttr.getSize(), denseArrayAttr.getData());
  }

  if (llvm::isa<SerializableAttrInterface>(attr))
    return Attribute();

  return attr;
}

void populateToByreSerialTypeAndAttrRewriter(AttrTypeReplacer &replacer) {
  replacer.addReplacement(
      [](Type type) -> std::optional<std::pair<Type, WalkResult>> {
        if (auto newType = mappingTypeTo(type)) {
          return std::pair(newType, WalkResult::advance());
        }

        return std::pair(Type(), WalkResult::interrupt());
      });
  replacer.addReplacement(
      [](Attribute attr) -> std::optional<std::pair<Attribute, WalkResult>> {
        if (auto newAttr = mappingAttrTo(attr)) {
          return std::pair(newAttr, WalkResult::advance());
        }

        return std::pair(Attribute(), WalkResult::interrupt());
      });
}

void populateFromByreSerialTypeAndAttrRewriter(AttrTypeReplacer &replacer) {
  replacer.addReplacement(
      [](Type type) -> std::optional<std::pair<Type, WalkResult>> {
        if (auto newType = mappingTypeFrom(type)) {
          return std::pair(newType, WalkResult::advance());
        }

        return std::pair(Type(), WalkResult::interrupt());
      });
  replacer.addReplacement(
      [](Attribute attr) -> std::optional<std::pair<Attribute, WalkResult>> {
        if (auto newAttr = mappingAttrFrom(attr)) {
          return std::pair(newAttr, WalkResult::advance());
        }

        return std::pair(Attribute(), WalkResult::interrupt());
      });
}

Operation *createNewOpGeneric(Operation *op, StringRef operationName,
                              AttrTypeReplacer &replacer, OpBuilder &b) {
  NamedAttrList attrs;
  for (auto &&attr : op->getAttrs()) {
    auto newAttr = replacer.replace(attr.getValue());
    if (!newAttr)
      return nullptr;

    attrs.append(attr.getName(), newAttr);
  }

  SmallVector<Type> types;
  for (auto &&type : op->getResultTypes()) {
    auto newType = replacer.replace(type);
    if (!newType)
      return nullptr;

    types.push_back(newType);
  }

  SmallVector<std::unique_ptr<Region>> regions;
  for (auto &&region : op->getRegions()) {
    auto newRegion = std::make_unique<Region>();
    IRMapping mapping;
    region.cloneInto(newRegion.get(), mapping);
    for (auto &&[bb0, bb1] :
         llvm::zip(region.getBlocks(), newRegion->getBlocks())) {
      for (auto &&[arg0, arg1] :
           llvm::zip(bb0.getArguments(), bb1.getArguments())) {
        auto newArgType = replacer.replace(arg0.getType());
        if (!newArgType)
          return nullptr;

        arg1.setType(newArgType);
        arg1.setLoc(b.getUnknownLoc());
      }
    }
    regions.push_back(std::move(newRegion));
  }

  OperationState state(b.getUnknownLoc(), operationName, op->getOperands(),
                       types, attrs, op->getSuccessors(), regions);
  Operation *newOp = b.create(state);

  return newOp;
}

template <typename T>
T createNewOpGeneric(Operation *op, AttrTypeReplacer &replacer, OpBuilder &b) {
  return llvm::cast_or_null<T>(
      createNewOpGeneric(op, T::getOperationName(), replacer, b));
}

LogicalResult replaceByreWithByreSerialRecursively(Operation *);
LogicalResult replaceByreSerialWithByreRecursively(Operation *);

template <typename T, typename U>
LogicalResult replaceByreWithByreSerialImpl(U op) {
  AttrTypeReplacer replacer;
  populateToByreSerialTypeAndAttrRewriter(replacer);
  OpBuilder b(op);
  auto newOp = createNewOpGeneric<T>(op, replacer, b);
  if (!newOp)
    return failure();

  if (failed(replaceByreWithByreSerialRecursively(newOp)))
    return failure();

  op->replaceAllUsesWith(newOp);
  op->erase();
  return success();
}

LogicalResult replaceByreWithByreSerialImpl(func::FuncOp func) {
  MLIRContext *ctx = func->getContext();
  Builder builder(ctx);
  SmallVector<Attribute> entryFuncArgTypes;
  for (int64_t i = 0; i < func.getNumArguments(); ++i) {
    if (auto argType = func.getArgAttrOfType<EntryFuncArgTypeAttr>(
            i, ByreDialect::getEntryPointFuncArgTypeAttrName())) {
      auto argTypeValue = argType.getValue();
      auto newArgTypeValue = ArgTypeV1::None;
      if (bitEnumContainsAll(argTypeValue, EntryFuncArgType::Input)) {
        newArgTypeValue = newArgTypeValue | ArgTypeV1::Input;
      }
      if (bitEnumContainsAll(argTypeValue, EntryFuncArgType::Output)) {
        newArgTypeValue = newArgTypeValue | ArgTypeV1::Output;
      }
      if (bitEnumContainsAll(argTypeValue, EntryFuncArgType::Weight)) {
        newArgTypeValue = newArgTypeValue | ArgTypeV1::Weight;
      }
      func.removeArgAttr(i, ByreDialect::getEntryPointFuncArgTypeAttrName());
      entryFuncArgTypes.push_back(ArgTypeV1Attr::get(ctx, newArgTypeValue));
    } else {
      return failure();
    }
  }
  func->setAttr("byre_arg_types", ArrayAttr::get(ctx, entryFuncArgTypes));

  SmallVector<Attribute> entryFuncArgNames;
  for (int64_t i = 0; i < func.getNumArguments(); ++i) {
    if (auto argName = func.getArgAttr(
            i, ByreDialect::getEntryPointFuncArgNameAttrName())) {
      func.removeArgAttr(i, ByreDialect::getEntryPointFuncArgNameAttrName());
      entryFuncArgNames.push_back(argName);
    } else {
      return failure();
    }
  }
  func->setAttr("byre_arg_names", ArrayAttr::get(ctx, entryFuncArgNames));

  SmallVector<Attribute> entryFuncArgAliasIndexs;
  for (int64_t i = 0; i < func.getNumArguments(); ++i) {
    if (auto argAliasIndex = func.getArgAttr(
            i, ByreDialect::getEntryPointFuncArgAliasIndexAttrName())) {
      func.removeArgAttr(i,
                         ByreDialect::getEntryPointFuncArgAliasIndexAttrName());
      entryFuncArgAliasIndexs.push_back(builder.getI64IntegerAttr(i));
      entryFuncArgAliasIndexs.push_back(argAliasIndex);
    }
  }
  if (entryFuncArgAliasIndexs.size() > 0) {
    func->setAttr("byre_arg_alias_indexs",
                  ArrayAttr::get(ctx, entryFuncArgAliasIndexs));
  }

  for (int64_t i = 0; i < func.getNumArguments(); ++i) {
    if (auto argAttrs = func.getArgAttrDict(i)) {
      if (!argAttrs.empty())
        return failure();
    }
  }
  func->removeAttr(func.getArgAttrsAttrName());

  for (int64_t i = 0; i < func.getNumResults(); ++i) {
    if (auto resAttrs = func.getResultAttrDict(i)) {
      if (!resAttrs.empty())
        return failure();
    }
  }
  func->removeAttr(func.getResAttrsAttrName());

  if (auto attr = func->getAttr(ByreDialect::getEntryPointFunctionAttrName())) {
    func->setAttr("byre_entry_point", attr);
    func->removeAttr(ByreDialect::getEntryPointFunctionAttrName());
  }

  if (auto attr = func->getAttr("byteir.entry_point")) {
    func->setAttr("byteir_entry_point", attr);
    func->removeAttr("byteir.entry_point");
  }

  if (auto attr = func->getAttr("tf.original_input_names")) {
    func->setAttr("tf_original_input_names", attr);
    func->removeAttr("tf.original_input_names");
  }

  return replaceByreWithByreSerialImpl<FuncOpV1>(func);
}

LogicalResult replaceByreWithByreSerialImpl(ComputeOp compute) {
  MLIRContext *ctx = compute->getContext();
  if (auto effects = compute->getAttrOfType<ArrayAttr>("memory_effects")) {
    SmallVector<Attribute> attrs;
    for (auto &&attr : effects.getValue()) {
      if (auto memoryEffect = llvm::dyn_cast<MemoryEffectAttr>(attr)) {
        auto memoryEffectValue = memoryEffect.getValue();
        auto newMemoryEffectValue = MemoryEffectV1::None;
        if (bitEnumContainsAll(memoryEffectValue, MemoryEffect::Read)) {
          newMemoryEffectValue = newMemoryEffectValue | MemoryEffectV1::Read;
        }
        if (bitEnumContainsAll(memoryEffectValue, MemoryEffect::Write)) {
          newMemoryEffectValue = newMemoryEffectValue | MemoryEffectV1::Write;
        }
        attrs.push_back(MemoryEffectV1Attr::get(ctx, newMemoryEffectValue));
      } else {
        return failure();
      }
    }
    compute->setAttr("memory_effects", ArrayAttr::get(ctx, attrs));
  }
  NamedAttrList attrs = compute->getAttrs();
  attrs.erase("callee");
  attrs.erase("memory_effects");
  for (auto &&i : attrs) {
    compute->removeAttr(i.getName());
  }
  auto extraArgs = attrs.getDictionary(ctx);
  compute->setAttr("extra_args", extraArgs);
  return replaceByreWithByreSerialImpl<ComputeOpV1>(compute);
}

template <typename T, typename U>
LogicalResult replaceByreSerialWithByreImpl(U op) {
  AttrTypeReplacer replacer;
  populateFromByreSerialTypeAndAttrRewriter(replacer);
  OpBuilder b(op);
  auto newOp = createNewOpGeneric<T>(op, replacer, b);
  if (!newOp)
    return failure();

  if (failed(replaceByreSerialWithByreRecursively(newOp)))
    return failure();

  op->replaceAllUsesWith(newOp);
  op->erase();
  return success();
}

LogicalResult replaceByreSerialWithByreImpl(FuncOpV1 func) {
  AttrTypeReplacer replacer;
  populateFromByreSerialTypeAndAttrRewriter(replacer);
  OpBuilder b(func);
  auto newFunc = createNewOpGeneric<func::FuncOp>(func, replacer, b);
  if (!newFunc)
    return failure();

  if (failed(replaceByreSerialWithByreRecursively(newFunc)))
    return failure();

  if (auto entryFuncArgTypes =
          newFunc->getAttrOfType<ArrayAttr>("byre_arg_types")) {
    for (size_t i = 0; i < newFunc.getNumArguments(); ++i) {
      newFunc.setArgAttr(i, ByreDialect::getEntryPointFuncArgTypeAttrName(),
                         entryFuncArgTypes[i]);
    }
    newFunc->removeAttr("byre_arg_types");
  }

  if (auto entryFuncArgNames =
          newFunc->getAttrOfType<ArrayAttr>("byre_arg_names")) {
    for (size_t i = 0; i < newFunc.getNumArguments(); ++i) {
      newFunc.setArgAttr(i, ByreDialect::getEntryPointFuncArgNameAttrName(),
                         entryFuncArgNames[i]);
    }
    newFunc->removeAttr("byre_arg_names");
  }

  if (auto entryFuncArgAliasIndexs =
          newFunc->getAttrOfType<ArrayAttr>("byre_arg_alias_indexs")) {
    for (size_t i = 0; i < entryFuncArgAliasIndexs.size(); i += 2) {
      newFunc.setArgAttr(entryFuncArgAliasIndexs[i]
                             .cast<IntegerAttr>()
                             .getValue()
                             .getSExtValue(),
                         ByreDialect::getEntryPointFuncArgAliasIndexAttrName(),
                         entryFuncArgAliasIndexs[i + 1]);
    }
    newFunc->removeAttr("byre_arg_alias_indexs");
  }

  if (auto attr = newFunc->getAttr("byre_entry_point")) {
    newFunc->setAttr(ByreDialect::getEntryPointFunctionAttrName(), attr);
    newFunc->removeAttr("byre_entry_point");
  }

  if (auto attr = newFunc->getAttr("byteir_entry_point")) {
    newFunc->setAttr("byteir.entry_point", attr);
    newFunc->removeAttr("byteir_entry_point");
  }

  if (auto attr = newFunc->getAttr("tf_original_input_names")) {
    newFunc->setAttr("tf.original_input_names", attr);
    newFunc->removeAttr("tf_original_input_names");
  }

  func->replaceAllUsesWith(newFunc);
  func->erase();
  return success();
}

LogicalResult replaceByreSerialWithByreImpl(ComputeOpV1 compute) {
  AttrTypeReplacer replacer;
  populateFromByreSerialTypeAndAttrRewriter(replacer);
  OpBuilder b(compute);
  auto newOp = createNewOpGeneric<ComputeOp>(compute, replacer, b);
  if (!newOp)
    return failure();

  if (auto extraArgs = newOp->getAttrOfType<DictionaryAttr>("extra_args")) {
    NamedAttrList attrs = newOp->getAttrs();
    attrs.erase("extra_args");
    attrs.append(extraArgs.getValue());
    newOp->setAttrs(attrs);
  }
  compute->replaceAllUsesWith(newOp);
  compute->erase();
  return success();
}

LogicalResult replaceByreWithByreSerial(Operation *op) {
  return llvm::TypeSwitch<Operation *, LogicalResult>(op)
      .Case([](func::FuncOp op) { return replaceByreWithByreSerialImpl(op); })
      .Case([](func::ReturnOp op) {
        return replaceByreWithByreSerialImpl<ReturnOpV1>(op);
      })
      .Case([](memref::AllocOp op) {
        return replaceByreWithByreSerialImpl<AllocOpV1>(op);
      })
      .Case([](ComputeOp op) { return replaceByreWithByreSerialImpl(op); })
      .Case([](AliasOp op) {
        return replaceByreWithByreSerialImpl<AliasOpV1>(op);
      })
      .Case(
          [](CopyOp op) { return replaceByreWithByreSerialImpl<CopyOpV1>(op); })
      .Case([](GroupCopyOp op) {
        return replaceByreWithByreSerialImpl<GroupCopyOpV1>(op);
      })
      .Default([&](Operation *op) { return failure(); });
}

LogicalResult replaceByreSerialWithByre(Operation *op) {
  return llvm::TypeSwitch<Operation *, LogicalResult>(op)
      .Case([](FuncOpV1 op) { return replaceByreSerialWithByreImpl(op); })
      .Case([](ReturnOpV1 op) {
        return replaceByreSerialWithByreImpl<func::ReturnOp>(op);
      })
      .Case([](AllocOpV1 op) {
        return replaceByreSerialWithByreImpl<memref::AllocOp>(op);
      })
      .Case([](ComputeOpV1 op) { return replaceByreSerialWithByreImpl(op); })
      .Case([](AliasOpV1 op) {
        return replaceByreSerialWithByreImpl<AliasOp>(op);
      })
      .Case(
          [](CopyOpV1 op) { return replaceByreSerialWithByreImpl<CopyOp>(op); })
      .Case([](GroupCopyOpV1 op) {
        return replaceByreSerialWithByreImpl<GroupCopyOp>(op);
      })
      .Default([&](Operation *op) { return failure(); });
}

LogicalResult replaceByreWithByreSerialRecursively(Operation *op) {
  WalkResult result = op->walk([](Operation *op) {
    if (llvm::isa<SerializableOpInterface>(op))
      return WalkResult::advance();

    auto status = replaceByreWithByreSerial(op);

    if (failed(status)) {
      op->emitError() << " failed to convert to serializable byre";
      return WalkResult::interrupt();
    }

    return WalkResult::advance();
  });
  if (result.wasInterrupted())
    return failure();
  return success();
}

LogicalResult replaceByreSerialWithByreRecursively(Operation *op) {
  WalkResult result = op->walk([](SerializableOpInterface op) {
    auto status = replaceByreSerialWithByre(op);
    if (failed(status)) {
      op->emitError() << " failed to convert from serializable byre";
      return WalkResult::interrupt();
    }

    return WalkResult::advance();
  });

  if (result.wasInterrupted())
    return failure();
  return success();
}
} // namespace

Type mlir::byre::mappingTypeTo(Type type) {
  if (!type)
    return {};

  if (llvm::isa<SerializableTypeInterface>(type))
    return type;

  auto ctx = type.getContext();
  if (auto concreteType = type.dyn_cast<IntegerType>()) {
    if (!concreteType.isSignless() && !concreteType.isUnsigned())
      return {};

    if (concreteType.getWidth() == 1 && concreteType.isSignless())
      return BooleanV1Type::get(ctx);

    bool isSignless = concreteType.isSignless();
    switch (concreteType.getWidth()) {
    case 4:
      return isSignless ? IntegerI4V1Type::get(ctx).cast<Type>()
                        : IntegerUI4V1Type::get(ctx).cast<Type>();
    case 8:
      return isSignless ? IntegerI8V1Type::get(ctx).cast<Type>()
                        : IntegerUI8V1Type::get(ctx).cast<Type>();
    case 16:
      return isSignless ? IntegerI16V1Type::get(ctx).cast<Type>()
                        : IntegerUI16V1Type::get(ctx).cast<Type>();
    case 32:
      return isSignless ? IntegerI32V1Type::get(ctx).cast<Type>()
                        : IntegerUI32V1Type::get(ctx).cast<Type>();
    case 64:
      return isSignless ? IntegerI64V1Type::get(ctx).cast<Type>()
                        : IntegerUI64V1Type::get(ctx).cast<Type>();
    }
  }

  if (auto concreteType = type.dyn_cast<IndexType>()) {
    return IndexV1Type::get(ctx);
  }

  if (auto concreteType = type.dyn_cast<FloatType>()) {
    if (concreteType.isa<BFloat16Type>())
      return FloatBF16V1Type::get(ctx);
    if (concreteType.isa<Float16Type>())
      return FloatF16V1Type::get(ctx);
    if (concreteType.isa<Float32Type>())
      return FloatF32V1Type::get(ctx);
    if (concreteType.isa<Float64Type>())
      return FloatF64V1Type::get(ctx);
    if (concreteType.isa<Float8E4M3FNType>())
      return FloatF8E4M3FNV1Type::get(ctx);
    if (concreteType.isa<Float8E5M2Type>())
      return FloatF8E5M2V1Type::get(ctx);
    if (concreteType.isa<Float8E4M3FNUZType>())
      return FloatF8E4M3FNUZV1Type::get(ctx);
    if (concreteType.isa<Float8E5M2FNUZType>())
      return FloatF8E5M2FNUZV1Type::get(ctx);
    if (concreteType.isa<Float8E4M3B11FNUZType>())
      return FloatF8E4M3B11FNUZV1Type::get(ctx);
  }

  if (auto concreteType = type.dyn_cast<ace::StringType>()) {
    return StringV1Type::get(ctx);
  }

  if (auto concreteType = type.dyn_cast<MemRefType>()) {
    // TODO: support layout
    return MemrefV1Type::get(ctx, concreteType.getShape(),
                             mappingTypeTo(concreteType.getElementType()),
                             mappingAttrTo(concreteType.getMemorySpace()));
  }

  if (auto concreteType = type.dyn_cast<RankedTensorType>()) {
    return RankedTensorV1Type::get(ctx, concreteType.getShape(),
                                   mappingTypeTo(concreteType.getElementType()),
                                   mappingAttrTo(concreteType.getEncoding()));
  }

  if (auto concreteType = type.dyn_cast<FunctionType>()) {
    llvm::SmallVector<Type> inputs, outputs;
    for (auto i : concreteType.getInputs()) {
      inputs.push_back(mappingTypeTo(i));
    }
    for (auto i : concreteType.getResults()) {
      outputs.push_back(mappingTypeTo(i));
    }
    return FunctionV1Type::get(ctx, inputs, outputs);
  }
  return {};
}

Type mlir::byre::mappingTypeFrom(Type type) {
  if (!type)
    return {};

  auto ctx = type.getContext();
  if (type.isa<BooleanV1Type>())
    return IntegerType::get(ctx, 1);
  if (type.isa<IntegerI4V1Type>())
    return IntegerType::get(ctx, 4);
  if (type.isa<IntegerUI4V1Type>())
    return IntegerType::get(ctx, 4, IntegerType::Unsigned);
  if (type.isa<IntegerI8V1Type>())
    return IntegerType::get(ctx, 8);
  if (type.isa<IntegerUI8V1Type>())
    return IntegerType::get(ctx, 8, IntegerType::Unsigned);
  if (type.isa<IntegerI16V1Type>())
    return IntegerType::get(ctx, 16);
  if (type.isa<IntegerUI16V1Type>())
    return IntegerType::get(ctx, 16, IntegerType::Unsigned);
  if (type.isa<IntegerI32V1Type>())
    return IntegerType::get(ctx, 32);
  if (type.isa<IntegerUI32V1Type>())
    return IntegerType::get(ctx, 32, IntegerType::Unsigned);
  if (type.isa<IntegerI64V1Type>())
    return IntegerType::get(ctx, 64);
  if (type.isa<IntegerUI64V1Type>())
    return IntegerType::get(ctx, 64, IntegerType::Unsigned);

  if (type.isa<IndexV1Type>())
    return IndexType::get(ctx);

  if (type.isa<FloatBF16V1Type>())
    return BFloat16Type::get(ctx);
  if (type.isa<FloatF16V1Type>())
    return Float16Type::get(ctx);
  if (type.isa<FloatF32V1Type>())
    return Float32Type::get(ctx);
  if (type.isa<FloatF64V1Type>())
    return Float64Type::get(ctx);
  if (type.isa<FloatF8E4M3FNV1Type>())
    return Float8E4M3FNType::get(ctx);
  if (type.isa<FloatF8E5M2V1Type>())
    return Float8E5M2Type::get(ctx);
  if (type.isa<FloatF8E4M3FNUZV1Type>())
    return Float8E4M3FNUZType::get(ctx);
  if (type.isa<FloatF8E5M2FNUZV1Type>())
    return Float8E5M2FNUZType::get(ctx);
  if (type.isa<FloatF8E4M3B11FNUZV1Type>())
    return Float8E4M3B11FNUZType::get(ctx);

  if (type.isa<StringV1Type>())
    return ace::StringType::get(ctx);

  if (auto concreteType = type.dyn_cast<MemrefV1Type>()) {
    // TODO: support layout
    return MemRefType::get(concreteType.getShape(),
                           mappingTypeFrom(concreteType.getElementType()),
                           MemRefLayoutAttrInterface{},
                           mappingAttrFrom(concreteType.getMemorySpace()));
  }

  if (auto concreteType = type.dyn_cast<RankedTensorV1Type>()) {
    return RankedTensorType::get(concreteType.getShape(),
                                 mappingTypeFrom(concreteType.getElementType()),
                                 mappingAttrFrom(concreteType.getEncoding()));
  }

  if (auto concreteType = type.dyn_cast<FunctionV1Type>()) {
    llvm::SmallVector<Type> inputs, outputs;
    for (auto i : concreteType.getInputs()) {
      inputs.push_back(mappingTypeFrom(i));
    }
    for (auto i : concreteType.getOutputs()) {
      outputs.push_back(mappingTypeFrom(i));
    }
    return FunctionType::get(ctx, inputs, outputs);
  }

  if (llvm::isa<SerializableTypeInterface>(type))
    return {};

  return type;
}

LogicalResult mlir::byre::verifySerializableIR(Operation *topLevel,
                                               bool verifyLocations) {
  AttrTypeWalker typeAttrChecker;
  typeAttrChecker.addWalk([](Type type) {
    if (llvm::isa<SerializableTypeInterface>(type))
      return WalkResult::advance();
    return WalkResult::interrupt();
  });
  typeAttrChecker.addWalk([](Attribute attr) {
    if (llvm::isa<SerializableAttrInterface>(attr))
      return WalkResult::advance();
    if (llvm::isa<UnknownLoc>(attr))
      return WalkResult::advance();
    return WalkResult::interrupt();
  });

  WalkResult result = topLevel->walk([&](Operation *op) {
    if (!llvm::isa<byre::serialization::SerializableOpInterface>(op)) {
      op->emitError() << " was not serializable";
      return WalkResult::interrupt();
    }

    if (verifyLocations &&
        typeAttrChecker.walk(op->getLoc()).wasInterrupted()) {
      op->emitError() << op->getLoc() << " was not serializable";
      return WalkResult::interrupt();
    }

    if (!op->getDiscardableAttrs().empty()) {
      op->emitError() << " was not serializable with discardable attribute";
      return WalkResult::interrupt();
    }

    for (auto &&attr : op->getAttrs()) {
      if (typeAttrChecker.walk(attr.getValue()).wasInterrupted()) {
        op->emitError() << attr.getValue() << " was not serializable";
        return WalkResult::interrupt();
      }
    }

    for (auto &&type : op->getResultTypes()) {
      if (typeAttrChecker.walk(type).wasInterrupted()) {
        op->emitError() << type << " was not serializable";
        return WalkResult::interrupt();
      }
    }

    for (auto &&region : op->getRegions())
      for (auto &&block : region.getBlocks())
        for (auto &&arg : block.getArguments()) {
          if (typeAttrChecker.walk(arg.getType()).wasInterrupted()) {
            op->emitError() << arg.getType() << " was not serializable";
            return WalkResult::interrupt();
          }
          if (verifyLocations &&
              typeAttrChecker.walk(arg.getLoc()).wasInterrupted()) {
            op->emitError() << arg.getLoc() << " was not serializable";
            return WalkResult::interrupt();
          }
        }
    return WalkResult::advance();
  });

  if (result.wasInterrupted())
    return failure();

  return success();
}

Operation *mlir::byre::convertToSerializableByre(ModuleOp topLevelOp) {
  if (!topLevelOp)
    return nullptr;

  AttrTypeReplacer replacer;
  populateToByreSerialTypeAndAttrRewriter(replacer);
  OpBuilder b(topLevelOp->getContext());
  Operation *mod = createNewOpGeneric<ModuleOpV1>(topLevelOp, replacer, b);
  if (!mod)
    return nullptr;

  if (auto attr = mod->getAttr(ByreDialect::getContainerModuleAttrName())) {
    mod->setAttr("container_module", attr);
    mod->removeAttr(ByreDialect::getContainerModuleAttrName());
  }

  if (failed(replaceByreWithByreSerialRecursively(mod))) {
    mod->erase();
    return nullptr;
  }

  return mod;
}

ModuleOp mlir::byre::convertFromSerializableByre(Operation *topLevelOp) {
  if (!llvm::isa_and_nonnull<ModuleOpV1>(topLevelOp))
    return nullptr;

  AttrTypeReplacer replacer;
  populateFromByreSerialTypeAndAttrRewriter(replacer);
  OpBuilder b(topLevelOp->getContext());
  ModuleOp mod = createNewOpGeneric<ModuleOp>(topLevelOp, replacer, b);
  if (!mod)
    return nullptr;

  if (auto attr = mod->getAttr("container_module")) {
    mod->setAttr(ByreDialect::getContainerModuleAttrName(), attr);
    mod->removeAttr("container_module");
  }

  if (failed(replaceByreSerialWithByreRecursively(mod))) {
    mod->erase();
    return nullptr;
  }
  return mod;
}

LogicalResult mlir::byre::replaceFuncWithSerializableFunc(func::FuncOp func) {
  return replaceByreWithByreSerialImpl(func);
}

LogicalResult mlir::byre::replaceSerializableFuncWithFunc(Operation *op) {
  auto func = llvm::dyn_cast_or_null<FuncOpV1>(op);
  if (!func)
    return failure();

  return replaceByreSerialWithByreImpl(func);
}