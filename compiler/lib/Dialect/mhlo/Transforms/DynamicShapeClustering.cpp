//===- DynamicShapeClustering.cpp -----------------------------*--- C++ -*-===//
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

#include "byteir/Dialect/mhlo/Transforms/DynamicShapeClustering.h"
#include "byteir/Dialect/Shape/IR/ShapeExtOps.h"
#include "byteir/Dialect/mhlo/Util/FusionUtil.h"
#include "byteir/Dialect/mhlo/Util/Util.h"
#include "byteir/Utils/IRRewrite.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/Support/Debug.h"
#include <string>
#include <vector>

#include "./PassDetail.h"

using namespace mlir;

#define DEBUG_TYPE "dynamic-shape-clustering"

namespace {

inline bool isShapeOp(Operation *op) {
  return isa<tensor::TensorDialect, shape::ShapeDialect>(op->getDialect()) ||
         isa<arith::IndexCastOp, mhlo::ComputeReshapeShapeOp>(op);
}
struct DynamicSourceAnalysis {
  DynamicSourceAnalysis(Operation *op);

  void calPostDomsByTie() {
    operation->walk(
        [&](shape_ext::TieOp tieOp) { postDomByTieMem[tieOp] = true; });
    operation->walk([&](Operation *op) { calPostDomByTieRecursively(op); });
  }

  bool calPostDomByTieRecursively(Operation *op) {
    auto it = postDomByTieMem.find(op);
    if (it != postDomByTieMem.end())
      return it->second;

    if (op->getNumResults() == 0) {
      postDomByTieMem[op] = false;
      return false;
    }

    for (Value res : op->getResults()) {
      bool allDomed = llvm::all_of(res.getUsers(), [&](Operation *user) {
        return calPostDomByTieRecursively(user);
      });
      if (!allDomed) {
        postDomByTieMem[op] = false;
        return false;
      }
    }

    postDomByTieMem[op] = true;
    return true;
  }

  void calDynamicSource() {
    std::vector<shape_ext::TieOp> tieOps;
    DenseMap<Value, DenseSet<Value>> dynamicSourcesMem;

    operation->walk([&](shape_ext::TieOp tieOp) { tieOps.push_back(tieOp); });
    for (shape_ext::TieOp tieOp : tieOps) {
      Value v = tieOp.getValue();
      DenseSet<Value> sources;
      for (Value dimSize : tieOp.getDims()) {
        DenseSet<Value> &dimSources =
            calDynSrcRecursively(dimSize, dynamicSourcesMem);
        for (Value source : dimSources) {
          sources.insert(source);
        }
      }
      dynamicSources[v] = sources;
    }
  }

  DenseSet<Value> &
  calDynSrcRecursively(Value v,
                       DenseMap<Value, DenseSet<Value>> &dynamicSourcesMem) {
    auto it = dynamicSourcesMem.find(v);
    if (it != dynamicSourcesMem.end())
      return it->second;

    Operation *defOp = v.getDefiningOp();
    // This requires no mhlo ops in shape reification implementation
    if (nullptr == defOp || llvm::isa<mhlo::MhloDialect>(defOp->getDialect())) {
      dynamicSourcesMem[v] = {v};
      return dynamicSourcesMem[v];
    }

    DenseSet<Value> res;
    for (Value inp : defOp->getOperands()) {
      for (Value source : calDynSrcRecursively(inp, dynamicSourcesMem)) {
        res.insert(source);
      }
    }
    dynamicSourcesMem[v] = res;
    return dynamicSourcesMem[v];
  }

  void removeTieOps() {
    std::vector<Operation *> ops;
    operation->walk<WalkOrder::PreOrder>(
        [&](Operation *op) { ops.push_back(op); });

    for (auto it = ops.rbegin(); it != ops.rend(); ++it) {
      if (postDomByTieMem[*it]) {
        (*it)->erase();
      }
    }
  }

  void print(llvm::raw_ostream &os) {
    os << "=================== DynamicSourceAnalysis Printer "
          "=====================\n";
    for (auto it : dynamicSources) {
      Value v = it.first;
      os << "Sources of " << v << "\n";
      for (Value source : it.second)
        os << source << "\n";
      os << "\n";
    }
  }

  bool isDynamicSource(Value v) {
    auto it = dynamicSources.find(v);
    if (it == dynamicSources.end())
      return false;
    if (it->second.size() == 1 && v == *(it->second.begin())) {
      return true;
    }
    return false;
  }

  DenseMap<Operation *, bool> postDomByTieMem;
  DenseMap<Value, DenseSet<Value>> dynamicSources;
  Operation *operation;
};

DynamicSourceAnalysis::DynamicSourceAnalysis(Operation *operation)
    : operation(operation) {
  calPostDomsByTie();
  calDynamicSource();
}

struct DynamicShapeClusteringPass
    : public DynamicShapeClusteringBase<DynamicShapeClusteringPass> {

  explicit DynamicShapeClusteringPass(const std::string &anchorAttr)
      : DynamicShapeClusteringBase<DynamicShapeClusteringPass>() {
    this->anchorAttr = anchorAttr;
  }

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    SymbolTable symTable(moduleOp);
    SmallVector<func::FuncOp> funcOps;
    for (auto funcOp : moduleOp.getOps<func::FuncOp>()) {
      if (this->anchorAttr == "" || funcOp->hasAttr(this->anchorAttr)) {
        funcOps.push_back(funcOp);
      }
    }

    for (auto funcOp : funcOps) {
      DynamicSourceAnalysis dynSrcAnalysis(funcOp);
      // TODO: make removeTieOps optional or as an ioslated pass.
      LLVM_DEBUG(dynSrcAnalysis.print(llvm::dbgs()));
      dynSrcAnalysis.removeTieOps();

      auto isFusibleCandidate = [&](Operation *op) {
        if (isShapeOp(op) || op->hasTrait<OpTrait::ConstantLike>())
          return true;

        if (llvm::isa<mhlo::MhloDialect>(op->getDialect())) {
          for (Value operand : op->getOperands()) {
            if (auto shapeType = operand.getType().dyn_cast<ShapedType>()) {
              if (!shapeType.hasStaticShape())
                return true;
            }
          }
          // the input of DynamicBroadcastInDim are all static, but the result
          // is dynamic
          for (auto value : op->getResults()) {
            if (auto shapeType = value.getType().dyn_cast<ShapedType>()) {
              if (!dynSrcAnalysis.isDynamicSource(value) &&
                  !shapeType.hasStaticShape())
                return true;
            }
          }
        }

        return false;
      };

      auto isFusibleStart = [&](Operation *op) {
        for (Value operand : op->getOperands()) {
          if (dynSrcAnalysis.isDynamicSource(operand)) {
            return true;
          }
        }

        if (op->hasTrait<OpTrait::ConstantLike>())
          return true;

        return false;
      };

      auto isFusibleTrigger = [&](Operation *op) { return true; };

      auto isFusibleWith = [&](Operation *target, Operation *start) {
        if (isShapeOp(start) || isShapeOp(target))
          return true;

        DenseSet<Value> targetSources;
        DenseSet<Value> startSources;
        for (Value v : target->getResults()) {
          for (Value s : dynSrcAnalysis.dynamicSources[v]) {
            targetSources.insert(s);
          }
        }
        for (Value v : start->getResults()) {
          for (Value s : dynSrcAnalysis.dynamicSources[v]) {
            startSources.insert(s);
          }
        }

        if (!startSources.empty() &&
            target->hasTrait<OpTrait::ConstantLike>()) {
          return true;
        }

        llvm::set_intersect(targetSources, startSources);
        return !targetSources.empty();
      };

      for (auto &block : funcOp.getBlocks()) {
        replicateDefiningOp(&block, [](Operation *op) {
          return op->hasTrait<OpTrait::ConstantLike>();
        });
      }

      ProducerFusionPlanner planner(funcOp, isFusibleCandidate, isFusibleStart,
                                    isFusibleTrigger, isFusibleWith);
      planner.run();

      const MhloFusionPlan &plan = planner.getFusionPlan();

      std::string namePrefix = funcOp.getSymName().str() + "_sub_";
      int idx = 0;
      for (auto it = plan.rbegin(); it != plan.rend(); ++it) {
        auto &pattern = *it;
        if (pattern.size() == 1 &&
            pattern[0]->hasTrait<OpTrait::ConstantLike>())
          continue;
        OpBuilder b(pattern.back());
        std::string name = namePrefix + std::to_string(idx++);
        func::FuncOp subFnOp = *createFuncOpFromPattern(b, name, pattern);
        subFnOp->setAttr(getDynamicFuncAttrName(), b.getUnitAttr());
        symTable.insert(subFnOp);
      }
    }
  };
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createDynamicShapeClusteringPass(llvm::StringRef anchorTag /*=""*/) {
  return std::make_unique<DynamicShapeClusteringPass>(anchorTag.str());
}
