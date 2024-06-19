//===- LinalgDataPlace.cpp ------------------------------------*--- C++ -*-===//
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

#include "byteir/Dialect/Linalg/Transforms/LinalgDataPlace.h"
#include "byteir/Utils/MemUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "PassDetail.h"

using namespace llvm;
using namespace mlir;
using namespace mlir::linalg;

#define DEBUG_TYPE "linalg-data-place"

namespace {

// Local utils
// Return memory space from 'memSpaces' (a list of memory space) for a gvien
// idx. If out-of-bound, use the last value.
static int64_t getSpace(ArrayRef<int64_t> memSpaces, unsigned idx) {
  if (memSpaces.size() == 0)
    return getUnplacedSpace();

  if (idx < memSpaces.size()) {
    return memSpaces[idx];
  }
  return memSpaces.back();
}

// support promote for a Tensor Value
static LogicalResult promoteTensorValue(OpBuilder &b, mlir::Value val,
                                        dataPlaceType /*placeType*/) {
  // only support regular alloc (not space) now
  // TODO: use placeType to support space and/or alloca

  // only support Tensor only
  auto valType = dyn_cast<TensorType>(val.getType());

  if (!valType) {
    return failure();
  }

  // support DestinationStyleOpInterface and TensorSemantics only
  auto destOp = val.getDefiningOp<DestinationStyleOpInterface>();
  if (!destOp || !destOp.hasTensorSemantics()) {
    return failure();
  }

  // override DefiningOp init with an empty
  auto opOperand = destOp.getTiedOpOperand(cast<OpResult>(val));
  b.setInsertionPoint(destOp);
  // create an empty for overriding
  tensor::EmptyOp emptyOp = b.create<tensor::EmptyOp>(
      destOp.getLoc(), valType.getShape(), valType.getElementType());
  // override DefiningOp's init
  destOp->setOperand(opOperand->getOperandNumber(), emptyOp);

  // insert a copy before insertSlice and overide insertSlice's source
  for (auto user : val.getUsers()) {
    if (auto insertSlice = dyn_cast<tensor::InsertSliceOp>(user)) {
      b.setInsertionPoint(insertSlice);
      auto loc = insertSlice.getLoc();
      // create an empty for a copy
      Value cpEmpty = b.create<tensor::EmptyOp>(loc, valType.getShape(),
                                                valType.getElementType());
      // insert a copy
      auto copyOp = b.create<linalg::CopyOp>(loc, val, cpEmpty);
      // override insertSlice's source
      insertSlice.getSourceMutable().assign(copyOp->getResult(0));
    }
  }

  return success();
}

static void dataPlaceImpl(OpBuilder &b, LinalgOp op) {
  if (op == nullptr)
    return;

  SmallVector<int64_t> memSpaces;

  if (auto arrayAttr = op->getAttrOfType<ArrayAttr>(getDataPlaceAttrName())) {
    for (auto attr : arrayAttr) {
      if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
        memSpaces.push_back(intAttr.getInt());
      } else {
        memSpaces.push_back(getUnplacedSpace());
      }
    }
  }

  auto loc = op.getLoc();
  SmallVector<Value, 4> operands;
  int idx = 0;

  // handle inputs
  for (Value input : op.getDpsInputs()) {
    int64_t space = getSpace(memSpaces, idx++);

    if (space == getUnplacedSpace()) {
      operands.push_back(input);
    } else {
      b.setInsertionPoint(op);
      auto maybeNewInput = createAlloc(b, input, space);

      if (maybeNewInput.has_value()) {
        operands.push_back(*maybeNewInput);
        // create copy
        b.create<linalg::CopyOp>(loc, input, *maybeNewInput);
      } else {
        operands.push_back(input);
      }
    }
  }

  // handle outputs
  SmallVector<bool, 4> outputReplaced;
  for (auto output : op.getDpsInits()) {
    int64_t space = getSpace(memSpaces, idx++);

    if (space == getUnplacedSpace()) {
      operands.push_back(output);
      outputReplaced.push_back(false);
    } else {
      b.setInsertionPoint(op);
      auto maybeNewInput = createAlloc(b, output, space);

      if (maybeNewInput.has_value()) {
        operands.push_back(*maybeNewInput);
        outputReplaced.push_back(true);
        // TODO check outputs as inout??
        // if so, do copy
      } else {
        operands.push_back(output);
        outputReplaced.push_back(false);
      }
    }
  }

  b.setInsertionPointAfter(op);
  auto cloned = clone(b, op, op->getResultTypes(), operands);
  cloned->removeAttr(getDataPlaceAttrName());

  idx = 0;
  int64_t numInputs = op.getNumDpsInputs();
  for (auto output : op.getDpsInits()) {
    if (outputReplaced[idx]) {
      // copy output
      b.create<linalg::CopyOp>(loc, operands[numInputs + idx], output);
      ++idx;
    }
  }

  op.erase();
}

static void collectAnchorOp(func::FuncOp func,
                            SmallVectorImpl<Operation *> &collection,
                            ArrayRef<int64_t> spaces) {
  auto ctx = func.getContext();

  // collect op with getDataPlaceAttrName as intial values
  func.walk([&](LinalgOp op) {
    // skip non-targeting or visited block
    if (op->hasAttr(getDataPlaceAttrName())) {

      // rewrite attribute to 'spaces' if it is UnitAttr
      if (op->hasAttrOfType<UnitAttr>(getDataPlaceAttrName())) {
        SmallVector<Attribute> arrayAttr;

        for (auto s : spaces) {
          arrayAttr.push_back(IntegerAttr::get(IntegerType::get(ctx, 32), s));
        }

        op->setAttr(getDataPlaceAttrName(), ArrayAttr::get(ctx, arrayAttr));
      } else if (!op->hasAttrOfType<ArrayAttr>(getDataPlaceAttrName())) {
        return;
      }
      collection.emplace_back(op);
    }
  });
}

struct LinalgDataPlacePass : public LinalgDataPlaceBase<LinalgDataPlacePass> {
  LinalgDataPlacePass() = default;

  explicit LinalgDataPlacePass(ArrayRef<int64_t> spaces) {
    this->memSpaces = spaces;
  }

  LinalgDataPlacePass(dataPlaceCollectType collect, bool useTensor) {
    isTensor = useTensor;
    collector = collect;
  }

  void runOnOperation() override {
    if (collector == nullptr)
      return;
    func::FuncOp funcOp = getOperation();

    mlir::DenseMap<mlir::Value, dataPlaceType> valCollection;
    funcOp.walk([&](Operation *op) { collector(op, valCollection); });

    OpBuilder b(funcOp.getContext());
    for (const auto &it : valCollection) {
      (void)promoteTensorValue(b, it.first, it.second);
    }
  }

  bool isTensor;
  dataPlaceCollectType collector = nullptr;
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>> mlir::createLinalgDataPlacePass() {
  return std::make_unique<LinalgDataPlacePass>(
      genericElementwiseTensorCollector, /*useTensor*/ true);
}

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createLinalgDataPlacePass(dataPlaceCollectType collector, bool isTensor) {
  return std::make_unique<LinalgDataPlacePass>(collector, isTensor);
}

void mlir::genericElementwiseTensorCollector(
    mlir::Operation *op,
    mlir::DenseMap<mlir::Value, dataPlaceType> &collection) {

  auto linalgGeneric = dyn_cast<linalg::GenericOp>(op);
  // only support GenericOp with tesnor now
  if (!linalgGeneric || !linalgGeneric.hasTensorSemantics()) {
    return;
  }

  bool hasInsertSlice = false;
  bool hasAnotherGenericInput = false;
  for (Value res : op->getResults()) {
    // has user
    for (auto user : res.getUsers()) {
      if (isa<tensor::InsertSliceOp>(user)) {
        hasInsertSlice = true;
      }
      if (isa<linalg::GenericOp>(user)) {
        hasAnotherGenericInput = true;
      }
    }
    if (hasInsertSlice && hasAnotherGenericInput) {
      collection.insert(std::make_pair(res, std::make_pair(Attribute(), true)));
    }
  }
}
