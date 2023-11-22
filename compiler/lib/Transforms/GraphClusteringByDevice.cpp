//===- GraphClusteringByDevice.cpp ----------------------------*--- C++ -*-===//
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

#include "byteir/Transforms/GraphClusteringByDevice.h"

#include "byteir/Dialect/mhlo/Util/Util.h"
#include "byteir/Utils/IRRewrite.h"
#include "byteir/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/SetVector.h"

#include <list>

#include "PassDetail.h"

using namespace mlir;
using namespace llvm;

namespace {

constexpr const char *DEVICE_ATTR_HOST = "host";

struct FunctionMetadata {
  StringRef anchorName;
  // The device where function will run
  StringRef deviceAttr;
  // The original function name before partition.
  StringRef originalName;
  // The insertion point of partition functions.
  Block::iterator insertionPoint;
  // The partitioned function name.
  llvm::StringRef partitionName;
  // The input values of the function.
  llvm::SmallVector<Value, 4> inputs;
  // The result values of the function.
  llvm::SmallVector<Value, 4> results;
  // The operations to be included in the body of the function.
  llvm::SmallVector<Operation *, 8> ops;

  func::FuncOp partitionOp;
};

void insertOpsRecursively(Operation *op, SmallDenseSet<Operation *> &opSet) {
  auto pair = opSet.insert(op);
  if (!pair.second)
    return;
  for (Value v : op->getOperands()) {
    if (Operation *defOp = v.getDefiningOp()) {
      insertOpsRecursively(defOp, opSet);
    }
  }
}

std::optional<SmallVector<FunctionMetadata, 4>>
getFunctionMetadatasCustom(func::FuncOp funcOp, StringRef attrName,
                           StringRef deviceAttr, StringRef deviceAnchorName,
                           bool dupOutputs) {
  SmallVector<FunctionMetadata, 4> metadatas;

  return metadatas;
}

std::optional<SmallVector<FunctionMetadata, 4>>
getFunctionMetadatasFallback(func::FuncOp funcOp, StringRef attrName,
                             StringRef deviceAttr, StringRef deviceAnchorName,
                             bool dupOutputs) {
  SmallVector<FunctionMetadata, 4> metadatas;
  SmallDenseSet<Operation *> hostOps;
  for (Operation &op : funcOp.front().without_terminator()) {
    if (op.hasAttr(attrName)) {
      StringAttr attr = op.getAttrOfType<StringAttr>(attrName);
      if (attr.getValue().str() == DEVICE_ATTR_HOST) {
        insertOpsRecursively(&op, hostOps);
      }
    }
  }

  Operation &retOp = funcOp.front().back();
  llvm::DenseMap<Value, int64_t> retStats;
  for (const auto &operand : retOp.getOperands()) {
    if (retStats.count(operand)) {
      retStats[operand] += 1;
    } else {
      retStats.insert(std::make_pair(operand, 1));
    }
  }

  if (hostOps.size() > 0) {
    FunctionMetadata hostFuncMetadata;
    hostFuncMetadata.anchorName = getHostAnchorName();
    hostFuncMetadata.deviceAttr = DEVICE_ATTR_HOST;
    hostFuncMetadata.originalName = funcOp.getSymName();
    hostFuncMetadata.insertionPoint = ++Block::iterator(funcOp);
    for (Operation &op : funcOp.front().without_terminator()) {
      if (hostOps.count(&op)) {
        hostFuncMetadata.ops.push_back(&op);
      }
    }
    hostFuncMetadata.inputs = getInputsOfCluster(hostFuncMetadata.ops);
    hostFuncMetadata.results = getOutputsOfCluster(
        hostFuncMetadata.ops, dupOutputs ? &retStats : nullptr);
    metadatas.push_back(hostFuncMetadata);
  }

  FunctionMetadata deviceFuncMetadata;
  deviceFuncMetadata.anchorName = deviceAnchorName;
  deviceFuncMetadata.deviceAttr = deviceAttr;
  deviceFuncMetadata.originalName = funcOp.getSymName();
  deviceFuncMetadata.insertionPoint = ++Block::iterator(funcOp);
  for (Operation &op : funcOp.front().without_terminator()) {
    if (!hostOps.count(&op)) {
      deviceFuncMetadata.ops.push_back(&op);
    }
  }
  if (deviceFuncMetadata.ops.size() > 0) {
    deviceFuncMetadata.inputs = getInputsOfCluster(deviceFuncMetadata.ops);
    deviceFuncMetadata.results = getOutputsOfCluster(
        deviceFuncMetadata.ops, dupOutputs ? &retStats : nullptr);

    metadatas.push_back(deviceFuncMetadata);
  }

  return metadatas;
}

struct ActiveDeviceCluster {
  using OpList = llvm::SetVector<Operation *>;
  OpList operations;
  ActiveDeviceCluster *mergedInto;

  ActiveDeviceCluster(Operation *op) {
    operations.insert(op);
    mergedInto = nullptr;
  }

  ActiveDeviceCluster *getRoot() {
    if (!this->mergedInto)
      return this;

    return this->mergedInto = this->mergedInto->getRoot();
  }

  bool isBeforeInBlock(ActiveDeviceCluster *other) {
    if (this->operations.back()->isBeforeInBlock(other->operations.front())) {
      return true;
    }
    return false;
  }

  // return merged ActiveDeviceCluster or nullptr for merge failure
  // arg order sensitive, prefer merge lhs into rhs
  static ActiveDeviceCluster *tryMerge(ActiveDeviceCluster *lhs,
                                       ActiveDeviceCluster *rhs);

  struct CompareByNumOps {
    bool operator()(ActiveDeviceCluster *lhs, ActiveDeviceCluster *rhs) {
      return lhs->operations.size() > rhs->operations.size();
    }
  };

private:
  static bool tryMergeInto(ActiveDeviceCluster *from, ActiveDeviceCluster *to);

  static bool anyDefIn(Operation *op, const OpList &operations) {
    for (auto &&operand : op->getOperands())
      if (operations.count(operand.getDefiningOp()))
        return true;
    return false;
  }

  static bool anyUseIn(Operation *op, const OpList &operations) {
    for (auto &&use : op->getUses())
      if (operations.count(use.getOwner()))
        return true;
    return false;
  }

  // operations in \p src that can be moved up to \p target will be store in
  // \p moveUp in pre-order, and the remaining operations will be kept in \p src
  // in pre-order
  static auto computeMoveUpSet(const OpList &target, OpList &src,
                               OpList &moveUp) {
    std::vector<Operation *> vec = src.takeVector();
    OpList &remain = src;
    for (auto &&op : vec) {
      if (anyDefIn(op, target) || anyDefIn(op, remain)) {
        remain.insert(op);
      } else {
        moveUp.insert(op);
      }
    }
  }

  // operations in \p src that can be moved down to \p target will be store in
  // \p moveDown in post-order, and the remaining operations will be kept in \p
  // src in pre-order
  static auto computeMoveDownSet(const OpList &target, OpList &src,
                                 OpList &moveDown) {
    std::vector<Operation *> vec = src.takeVector();
    OpList &remain = src;
    for (auto &&op : llvm::reverse(vec)) {
      if (anyUseIn(op, target) || anyUseIn(op, remain)) {
        remain.insert(op);
      } else {
        moveDown.insert(op);
      }
    }

    vec = remain.takeVector();
    remain.insert(vec.rbegin(), vec.rend());
  }
};

bool ActiveDeviceCluster::tryMergeInto(ActiveDeviceCluster *from,
                                       ActiveDeviceCluster *to) {
  static auto takePointer = [](Operation &op) { return &op; };
  if (from->isBeforeInBlock(to)) {
    OpList toMove(
        llvm::map_iterator(std::next(from->operations.back()->getIterator()),
                           takePointer),
        llvm::map_iterator(to->operations.front()->getIterator(), takePointer));
    OpList moveUp, moveDown;

    computeMoveUpSet(from->operations, toMove, moveUp);
    computeMoveDownSet(to->operations, toMove, moveDown);

    if (!toMove.empty())
      return false;

    for (auto &&op : moveUp) {
      op->moveBefore(from->operations.front());
    }

    for (auto &&op : moveDown) {
      op->moveAfter(to->operations.back());
    }

    std::vector<Operation *> toOperations = to->operations.takeVector();
    from->operations.insert(toOperations.begin(), toOperations.end());
    to->operations = std::move(from->operations);
  } else {
    assert(to->isBeforeInBlock(from) && "invalid cluster order");
    OpList toMove(
        llvm::map_iterator(std::next(to->operations.back()->getIterator()),
                           takePointer),
        llvm::map_iterator(from->operations.front()->getIterator(),
                           takePointer));
    OpList moveUp, moveDown;

    computeMoveDownSet(from->operations, toMove, moveDown);
    computeMoveUpSet(to->operations, toMove, moveUp);

    if (!toMove.empty())
      return false;

    for (auto &&op : moveUp) {
      op->moveBefore(to->operations.front());
    }

    for (auto &&op : moveDown) {
      op->moveAfter(from->operations.back());
    }

    std::vector<Operation *> fromOperations = from->operations.takeVector();
    to->operations.insert(fromOperations.begin(), fromOperations.end());
  }

  from->mergedInto = to;
  return true;
}

ActiveDeviceCluster *ActiveDeviceCluster::tryMerge(ActiveDeviceCluster *lhs,
                                                   ActiveDeviceCluster *rhs) {
  if (!lhs || !rhs || lhs == rhs)
    return nullptr;

  if (lhs->mergedInto || rhs->mergedInto)
    return nullptr;

  if (tryMergeInto(lhs, rhs)) {
    return rhs;
  }

  if (tryMergeInto(rhs, lhs)) {
    return lhs;
  }

  return nullptr;
}

class DeviceClusteringAlgoBaseHelper {
public:
  std::optional<SmallVector<FunctionMetadata, 4>>
  getFunctionMetadatas(StringRef attrName, StringRef deviceAttr,
                       StringRef deviceAnchorName, bool dupOutputs);

protected:
  DeviceClusteringAlgoBaseHelper(func::FuncOp funcOp, StringRef attrName);

  ActiveDeviceCluster *getCluster(Operation *op) {
    auto &&iter = op2cluster.find(op);
    if (iter == op2cluster.end())
      return nullptr;

    return iter->second.getRoot();
  }

  ActiveDeviceCluster *getCluster(Value value) {
    return getCluster(value.getDefiningOp());
  }

  // void mergeDeviceClustersProgressively();
  void populateCandidates();

  func::FuncOp funcOp;
  llvm::DenseMap<Operation *, ActiveDeviceCluster> op2cluster;
  std::vector<ActiveDeviceCluster *> candidates;
};

DeviceClusteringAlgoBaseHelper::DeviceClusteringAlgoBaseHelper(
    func::FuncOp funcOp, StringRef attrName)
    : funcOp(funcOp) {
  for (auto &&op : funcOp.front().without_terminator()) {
    if (op.hasAttr(attrName)) {
      StringAttr attr = op.getAttrOfType<StringAttr>(attrName);
      if (attr.getValue().str() == "hbm") {
        continue;
      }
    }

    op2cluster.try_emplace(&op, ActiveDeviceCluster(&op));
  }
}

std::optional<SmallVector<FunctionMetadata, 4>>
DeviceClusteringAlgoBaseHelper::getFunctionMetadatas(StringRef attrName,
                                                     StringRef deviceAttr,
                                                     StringRef deviceAnchorName,
                                                     bool dupOutputs) {
  if (candidates.empty())
    return std::nullopt;

  auto &&firstCluster = candidates[0];
  if (firstCluster->operations.empty())
    return std::nullopt;

  SmallVector<FunctionMetadata, 4> metadatas;
  Operation &retOp = funcOp.front().back();
  llvm::DenseMap<Value, int64_t> retStats;
  for (const auto &operand : retOp.getOperands()) {
    if (retStats.count(operand)) {
      retStats[operand] += 1;
    } else {
      retStats.insert(std::make_pair(operand, 1));
    }
  }

  FunctionMetadata deviceFuncMetadata;
  deviceFuncMetadata.anchorName = deviceAnchorName;
  deviceFuncMetadata.deviceAttr = deviceAttr;
  deviceFuncMetadata.originalName = funcOp.getSymName();
  deviceFuncMetadata.insertionPoint = ++Block::iterator(funcOp);
  deviceFuncMetadata.ops = llvm::to_vector(firstCluster->operations);
  deviceFuncMetadata.inputs = getInputsOfCluster(deviceFuncMetadata.ops);
  deviceFuncMetadata.results = getOutputsOfCluster(
      deviceFuncMetadata.ops, dupOutputs ? &retStats : nullptr);
  metadatas.push_back(deviceFuncMetadata);

  return metadatas;
}

void DeviceClusteringAlgoBaseHelper::populateCandidates() {
  std::list<ActiveDeviceCluster *> workList;
  for (auto &&[_, cluster] : op2cluster) {
    if (!cluster.mergedInto) {
      workList.push_back(&cluster);
    }
  }
  workList.sort(ActiveDeviceCluster::CompareByNumOps());
  candidates.clear();
  while (!workList.empty()) {
    ActiveDeviceCluster *cluster = workList.front();
    workList.pop_front();
    for (auto &&iter = workList.begin(); iter != workList.end();) {
      if (auto merged = ActiveDeviceCluster::tryMerge(*iter, cluster)) {
        cluster = merged;
        iter = workList.erase(iter);
      } else {
        iter++;
      }
    }
    candidates.push_back(cluster);
  }
  llvm::sort(candidates, ActiveDeviceCluster::CompareByNumOps());
}

// all derived classes are expected to implement
// `mergeDeviceClustersProgressively()`
template <typename Derived>
class DeviceClusteringAlgoBase : public DeviceClusteringAlgoBaseHelper {
public:
  DeviceClusteringAlgoBase(func::FuncOp funcOp, StringRef attrName)
      : DeviceClusteringAlgoBaseHelper(funcOp, attrName) {
    static_assert(std::is_base_of<DeviceClusteringAlgoBase, Derived>::value);

    static_cast<Derived *>(this)->mergeDeviceClustersProgressively();
    populateCandidates();
  }
};

class TopDownDeviceClustering
    : public DeviceClusteringAlgoBase<TopDownDeviceClustering> {
public:
  using DeviceClusteringAlgoBase::DeviceClusteringAlgoBase;
  void mergeDeviceClustersProgressively();
};

void TopDownDeviceClustering::mergeDeviceClustersProgressively() {
  for (auto &&op : funcOp.front().without_terminator()) {
    auto curCluster = getCluster(&op);
    for (auto &&operand : op.getOperands()) {
      auto preCluster = getCluster(operand);
      if (auto merged = ActiveDeviceCluster::tryMerge(curCluster, preCluster)) {
        curCluster = merged;
      }
    }
  }
}

class BottomUpDeviceClustering
    : public DeviceClusteringAlgoBase<BottomUpDeviceClustering> {
public:
  using DeviceClusteringAlgoBase::DeviceClusteringAlgoBase;
  void mergeDeviceClustersProgressively();
};

void BottomUpDeviceClustering::mergeDeviceClustersProgressively() {
  for (auto &&op : llvm::reverse(funcOp.front().without_terminator())) {
    auto curCluster = getCluster(&op);
    for (auto &&use : op.getUses()) {
      auto preCluster = getCluster(use.getOwner());
      if (auto merged = ActiveDeviceCluster::tryMerge(preCluster, curCluster)) {
        curCluster = merged;
      }
    }
  }
}

void createFunctions(ModuleOp module_op,
                     SmallVector<FunctionMetadata, 4> &metadatas,
                     StringRef attrName) {
  MLIRContext *context = module_op.getContext();
  SymbolTable symbolTable(module_op);
  for (auto &metadata : metadatas) {
    llvm::SmallVector<mlir::Type, 4> inputTypes;
    llvm::SmallVector<mlir::Type, 4> resultTypes;
    for (Value input : metadata.inputs) {
      inputTypes.push_back(input.getType());
    }
    for (Value result : metadata.results) {
      resultTypes.push_back(result.getType());
    }
    std::string funcName =
        (metadata.originalName + "_" + metadata.deviceAttr).str();
    FunctionType funcType = FunctionType::get(context, inputTypes, resultTypes);
    func::FuncOp funcOp =
        func::FuncOp::create(UnknownLoc::get(context), funcName, funcType);
    funcOp->setAttr(attrName, StringAttr::get(context, metadata.deviceAttr));
    funcOp->setAttr(metadata.anchorName, UnitAttr::get(context));
    funcOp.setPublic();
    Block *block = funcOp.addEntryBlock();

    // Clones and moves the operations into the function's body. And the cloned
    // operation should use the arguments of the newly created funcOp as
    // appropriate.
    OpBuilder builder(block, block->end());
    IRMapping mapping;
    for (int i : llvm::seq<int>(0, metadata.inputs.size())) {
      Value originalValue = metadata.inputs[i];
      Value newValue = funcOp.getArgument(i);
      mapping.map(originalValue, newValue);
    }
    for (Operation *op : metadata.ops) {
      builder.clone(*op, mapping);
    }
    // Creates the ReturnOp so that the per-host function returns the
    // correct values of the cloned operations.
    llvm::SmallVector<Value, 4> resultsAfterMapping;
    for (Value result : metadata.results) {
      resultsAfterMapping.push_back(mapping.lookupOrDefault(result));
    }
    builder.create<func::ReturnOp>(UnknownLoc::get(context),
                                   resultsAfterMapping);
    symbolTable.insert(funcOp, metadata.insertionPoint++);
    // Record the actual name. The symbol table might rename the FuncOp if there
    // is name collision.
    metadata.partitionName = funcOp.getName();
    metadata.partitionOp = funcOp;
  }
}

void createCalls(MLIRContext *context,
                 const SmallVector<FunctionMetadata, 4> &metadatas,
                 Operation *retOp, bool dupOutputs) {
  IRMapping mapping;
  for (auto &metadata : metadatas) {
    // Creates the CallOp.
    OpBuilder builder(metadata.ops.back());
    llvm::SmallVector<Type, 4> resultTypes;
    for (Value result : metadata.results) {
      resultTypes.push_back(result.getType());
    }
    llvm::SmallVector<Value, 4> inputsAfterMapping;
    for (Value input : metadata.inputs) {
      inputsAfterMapping.push_back(mapping.lookupOrDefault(input));
    }

    func::CallOp callOp = builder.create<func::CallOp>(
        UnknownLoc::get(context), metadata.partitionOp, inputsAfterMapping);
    // Clones the CallOp operation to replace its callee args with
    // the results of the other CallOp operations using the
    // `mapping` as appropriate.
    Operation *clonedCallOp = builder.clone(*callOp.getOperation(), mapping);
    callOp.erase();

    llvm::DenseMap<Value, SmallVector<int>> retOperand2Indices;
    for (int i = retOp->getNumOperands() - 1; i >= 0; --i) {
      Value value = retOp->getOperand(i);
      retOperand2Indices[value].push_back(i);
    }

    // Replaces usages of the results of the original operations with the
    // results of the CallOp operations.
    for (int i : llvm::seq<int>(0, metadata.results.size())) {
      Value originalValue = metadata.results[i];
      Value newValue = clonedCallOp->getResult(i);
      if (dupOutputs) {
        originalValue.replaceAllUsesExcept(newValue, retOp);
        if (retOperand2Indices.find(originalValue) !=
            retOperand2Indices.end()) {
          assert(retOperand2Indices[originalValue].size() > 0 &&
                 "Corresponding indices vector must not be empty");
          int idx = retOperand2Indices[originalValue].back();
          retOperand2Indices[originalValue].pop_back();
          retOp->getOpOperand(idx).set(newValue);
        }
      } else {
        originalValue.replaceAllUsesWith(newValue);
      }
      mapping.map(originalValue, newValue);
    }
  }
}

struct GraphClusteringByDevicePass
    : public GraphClusteringByDeviceBase<GraphClusteringByDevicePass> {

  explicit GraphClusteringByDevicePass(std::string attrName, std::string device,
                                       std::string deviceAnchorName,
                                       bool dupNonSplat, bool dupOutputs,
                                       GraphClusteringAlgo clusterAlgo)
      : GraphClusteringByDeviceBase<
            GraphClusteringByDevicePass>::GraphClusteringByDeviceBase() {
    this->attrName = attrName;
    this->device = device;
    this->deviceAnchorName = deviceAnchorName;
    this->dupNonSplat = dupNonSplat;
    this->dupOutputs = dupOutputs;
    this->clusterAlgo = clusterAlgo;
  }

  void runOnOperation() override;
};

void GraphClusteringByDevicePass::runOnOperation() {
  ModuleOp moduleOp = getOperation();
  MLIRContext *context = &getContext();
  SmallVector<func::FuncOp, 4> originalFuncs;
  const auto isResultUsedByReturnOp =
      [](Operation *op, llvm::SmallDenseSet<Value> &retValues) {
        return llvm::any_of(op->getResults(), [&retValues](Value v) {
          return retValues.count(v) > 0;
        });
      };
  for (auto funcOp : moduleOp.getOps<func::FuncOp>()) {
    llvm::SmallDenseSet<Value> retValues;
    for (const auto &operand : funcOp.front().back().getOperands()) {
      retValues.insert(operand);
    }
    for (auto &block : funcOp.getBlocks()) {
      if (dupNonSplat)
        replicateDefiningOp(
            &block, [&retValues, &isResultUsedByReturnOp](Operation *op) {
              return !isResultUsedByReturnOp(op, retValues) &&
                     isMhloConstantLike(op);
            });
      else
        replicateDefiningOp(
            &block, [&retValues, &isResultUsedByReturnOp](Operation *op) {
              return !isResultUsedByReturnOp(op, retValues) &&
                     isSplatMhloConstantLike(op);
            });
    }
    originalFuncs.push_back(funcOp);
  }
  for (auto funcOp : originalFuncs) {
    std::optional<SmallVector<FunctionMetadata, 4>> metadatas;
    switch (this->clusterAlgo) {
    case GraphClusteringAlgo::kCustom:
      metadatas = getFunctionMetadatasCustom(funcOp, attrName, device,
                                             deviceAnchorName, dupOutputs);
      break;
    case GraphClusteringAlgo::kTopDown:
      metadatas = TopDownDeviceClustering(funcOp, attrName)
                      .getFunctionMetadatas(attrName, device, deviceAnchorName,
                                            dupOutputs);
      break;
    case GraphClusteringAlgo::kBottomUp:
      metadatas = BottomUpDeviceClustering(funcOp, attrName)
                      .getFunctionMetadatas(attrName, device, deviceAnchorName,
                                            dupOutputs);
      break;
    case GraphClusteringAlgo::kFallback:
    default:
      metadatas = getFunctionMetadatasFallback(funcOp, attrName, device,
                                               deviceAnchorName, dupOutputs);
    }

    if (!metadatas) {
      signalPassFailure();
      return;
    }

    Operation &retOp = funcOp.front().back();
    createFunctions(moduleOp, *metadatas, attrName);
    createCalls(context, *metadatas, &retOp, dupOutputs);

    // Erases the original operations which have been cloned in the partitioned
    // functions.
    for (auto &metadata : *metadatas) {
      for (int i = static_cast<int>(metadata.ops.size()) - 1; i >= 0; i--) {
        metadata.ops[i]->erase();
      }
    }
  }
}

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createGraphClusteringByDevicePass(std::string attrName,
                                        std::string device,
                                        std::string deviceAnchorName,
                                        bool dupNonSplat, bool dupOutputs,
                                        GraphClusteringAlgo clusterAlgo) {
  return std::make_unique<GraphClusteringByDevicePass>(
      attrName, device, deviceAnchorName, dupNonSplat, dupOutputs, clusterAlgo);
}
