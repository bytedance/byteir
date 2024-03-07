//===- ShapeAnalysis.h ----------------------------------------------------===//
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

#ifndef BYTEIR_DIALECT_MHLO_ANALYSIS_SHAPEANALYSIS_H
#define BYTEIR_DIALECT_MHLO_ANALYSIS_SHAPEANALYSIS_H

#include "byteir/Analysis/ShapeAnalysis.h"

namespace mlir {

class MhloShapeAnalysis : public ShapeAnalysis {
public:
  using ShapeAnalysis::ShapeAnalysis;

  LogicalResult inferResultShapesWithKnowledges(
      Operation *op, ShapeKnowledges shapeKnowledges,
      ShapeValueKnowledges shapeValueKnowledges,
      llvm::SmallVectorImpl<::mlir::ShapedTypeComponents> &results) override;
};

class MhloShapeValueAnalysis : public ShapeValueAnalysis {
public:
  using ShapeValueAnalysis::ShapeValueAnalysis;

  // in consistent with ShapeValueAnalysis, add customized handle logic for
  // ops in mhlo dialect
  void visitOperation(Operation *op,
                      ArrayRef<const ShapeValueLattice *> operands,
                      ArrayRef<ShapeValueLattice *> results) override;
};

class MhloBoundedShapeAnalysis : public MhloShapeAnalysis {
public:
  using MhloShapeAnalysis::MhloShapeAnalysis;

  void visitOperation(Operation *op, ArrayRef<const ShapeLattice *> operands,
                      ArrayRef<ShapeLattice *> results) override;
  LogicalResult inferResultShapesWithKnowledges(
      Operation *op, ShapeKnowledges shapeKnowledges,
      ShapeValueKnowledges shapeValueKnowledges,
      llvm::SmallVectorImpl<::mlir::ShapedTypeComponents> &results) override;
};

class MhloBoundedValueAnalysis : public BoundedValueAnalysis {
public:
  using BoundedValueAnalysis::BoundedValueAnalysis;

  void visitOperation(Operation *op,
                      ArrayRef<const BoundedValueLattice *> operands,
                      ArrayRef<BoundedValueLattice *> results) override;

protected:
  void visitOperation(Operation *op,
                      ArrayRef<const BoundedValueLattice *> operands,
                      // ArrayRef<ShapeLattice *> shapeLattices,
                      ArrayRef<ShapeValueLattice *> shapeValueLattices,
                      ArrayRef<BoundedValueLattice *> results);
  void foldOp(Operation *op, ArrayRef<Attribute> lowerAttrs,
              ArrayRef<Attribute> upperAttrs,
              ArrayRef<BoundedValueLattice *> results);
};

} // namespace mlir

#endif // BYTEIR_DIALECT_MHLO_ANALYSIS_SHAPEANALYSIS_H
