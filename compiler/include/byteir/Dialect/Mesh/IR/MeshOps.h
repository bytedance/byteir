//===- MeshOps.h ----------------------------------------------*--- C++ -*-===//
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

#ifndef BYTEIR_DIALECT_MESH_MESHOPS_H
#define BYTEIR_DIALECT_MESH_MESHOPS_H

#include "byteir/Dialect/Mesh/Interfaces/InferDTensorInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {

class DenseIntElementsAttr;

namespace func {
class FuncOp;
} // namespace func

namespace mesh {

class ClusterOp;

constexpr StringRef getMeshClusterAttrName() { return "mesh_cluster"; }

// Get the corresponding `mesh.cluster` op from funcOp's `mesh_cluster`
// attribute.
FailureOr<ClusterOp> getMeshClusterOp(func::FuncOp funcOp);

// Get the corresponding `mesh.cluster` op from parent func.func op
FailureOr<ClusterOp> getMeshClusterOp(Operation *op);

// Calculate the replica groups represented by device ids according to the mesh
// axes and cluster dimension size
FailureOr<DenseIntElementsAttr> getReplicaGroups(MLIRContext *ctx,
                                                 ArrayRef<int64_t> clusterShape,
                                                 ArrayRef<int64_t> axes);

FailureOr<DenseIntElementsAttr> getReplicaGroups(MLIRContext *ctx,
                                                 ArrayRef<int64_t> clusterShape,
                                                 DenseSet<int64_t> axes);
// Get rid of the tailing empty sub-array
void simplifyShardingOptionOrAnnotation(
    SmallVector<SmallVector<int64_t>> &array);

void populateMeshOpsCanonicalizationPatterns(RewritePatternSet &patterns);

} // namespace mesh

} // namespace mlir

#include "byteir/Dialect/Mesh/IR/MeshOpsDialect.h.inc"

#define GET_OP_CLASSES
#include "byteir/Dialect/Mesh/IR/MeshOps.h.inc"

#define GET_ATTRDEF_CLASSES
#include "byteir/Dialect/Mesh/IR/MeshOpsAttributes.h.inc"

#define GET_TYPEDEF_CLASSES
#include "byteir/Dialect/Mesh/IR/MeshOpsTypes.h.inc"

#include "byteir/Dialect/Mesh/IR/MeshOpsEnums.h.inc"

#endif // BYTEIR_DIALECT_MESH_MESHOPS_H
