//===- Passes.td - Transforms pass definition file -------*--- tablegen -*-===//
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

#ifndef BYTEIR_DIALECT_SHAPE_PASSES
#define BYTEIR_DIALECT_SHAPE_PASSES

include "mlir/Pass/PassBase.td"
include "byteir/Dialect/Shape/IR/ShapeExtBase.td"

//===----------------------------------------------------------------------===//
// ResolveShapeConstraint
//===----------------------------------------------------------------------===//

def ResolveShapeConstraint : Pass<"resolve-shape-constraint", "mlir::func::FuncOp"> {
  let summary = "Resolve shape constraint represented by MeetOp";
  let constructor = "mlir::createResolveShapeConstraintPass()";
}

//===----------------------------------------------------------------------===//
// InsertTieShape
//===----------------------------------------------------------------------===//

def InsertTieShape : Pass<"insert-tie-shape", "mlir::func::FuncOp"> {
  let summary = "Insert TieOp";
  let description = [{
    If a Value has dynamic shape, add a TieOp for it. The first operand of the
    TieOp is the Value, and the following operands are the symbolic expression
    of the dynamic dims. Currently the dynamic dims are only the results of
    tensor.dim, and they'll be reified later and optimized later.
  }];
  let constructor = "mlir::createInsertTieShapePass()";
  let dependentDialects = [
    "mlir::mhlo::MhloDialect",
    "mlir::shape_ext::ShapeExtDialect",
    "mlir::tensor::TensorDialect",
  ];
}

//===----------------------------------------------------------------------===//
// SetAssumingAlwaysTrue
//===----------------------------------------------------------------------===//

def SetAssumingAlwaysTrue: Pass<"set-assuming-always-true", "mlir::func::FuncOp"> {
  let summary = "Make shape.assuming always true";
  let description = [{
    set the true to witness in shape.assuming
  }];
  let constructor = "mlir::createSetAssumingAlwaysTruePass()";
  let dependentDialects = [
    "mlir::shape_ext::ShapeExtDialect",
    "mlir::tensor::TensorDialect",
    "mlir::shape::ShapeDialect",
  ];
}

#endif // BYTEIR_DIALECT_SHAPE_PASSES
