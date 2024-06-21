//===- TestPrintShapeAnalysis.cpp -----------------------------------------===//
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
#include "byteir/Dialect/mhlo/Analysis/ShapeAnalysis.h"
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::dataflow;

namespace {
struct TestPrintShapeAnalysisPass
    : public PassWrapper<TestPrintShapeAnalysisPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestPrintShapeAnalysisPass)

  StringRef getArgument() const final { return "test-print-shape-analysis"; }

  StringRef getDescription() const final { return "Print the shape analysis."; }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::tosa::TosaDialect>();
  }

  void runOnOperation() override {
    Operation *top = getOperation();

    DataFlowSolver solver;
    solver.load<MhloStaticShapeAnalysis>();
    solver.load<MhloStaticShapeValueAnalysis>();
    solver.load<DeadCodeAnalysis>();
    if (failed(solver.initializeAndRun(top)))
      return signalPassFailure();
    top->walk([&](Operation *op) {
      if (llvm::isa<InferShapedTypeOpInterface>(op)) {
        llvm::outs() << "for operation : " << *op
                     << ", inferred shapes are:\n\t";
        for (Value value : op->getResults()) {
          if (auto lattice = solver.lookupState<StaticShapeLattice>(value)) {
            if (!lattice->getValue().isUninitialized()) {
              lattice->getValue().print(llvm::outs());
            }
          }
        }
        llvm::outs() << "\n";
      }
      if (op->getNumResults()) {
        llvm::outs() << "for operation : " << *op
                     << ", inferred values are:\n\t";
        for (Value value : op->getResults()) {
          if (auto lattice =
                  solver.lookupState<Lattice<ConstantValue>>(value)) {
            if (!lattice->getValue().isUninitialized()) {
              lattice->getValue().print(llvm::outs());
            }
          }
        }
        llvm::outs() << "\n";
      }
    });
  }
};
} // namespace

namespace byteir {
namespace test {
void registerTestPrintShapeAnalysisPass() {
  PassRegistration<TestPrintShapeAnalysisPass>();
}
} // namespace test
} // namespace byteir
