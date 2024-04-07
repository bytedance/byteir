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


#ifndef BYTEIR_DIALECT_MHLO_PASSES
#define BYTEIR_DIALECT_MHLO_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// Bounded Shape Inference
//===----------------------------------------------------------------------===//

def BoundedShapeInference : Pass<"bounded-shape-infer", "func::FuncOp"> {
  let summary = "Bounded shape inference";
  let description = [{
    Some operations have dynamic output shapes, which could not be deduced at 
    compiling time, but their bound value could be. Note we only need to 
    implement the bounded shape inference functions for the dynamic operations, 
    and reuse the static operations' shape inference implementation. 
  }];
  let constructor = "mlir::createBoundedShapeInferencePass()";
}

//===----------------------------------------------------------------------===//
// ClusterConstraint
//===----------------------------------------------------------------------===//

def ClusterConstraint : Pass<"cluster-constraint", "mlir::func::FuncOp"> {
  let summary = "Cluster mhlo ops with constraint";
  let description = [{
    This pass do some early check in the tensor world(i.e. mhlo) and also treat
    those constraint values as constants and fuse them into mhlo.fusion
  }];
  let constructor = "mlir::createClusterConstraintPass()";
}

//===----------------------------------------------------------------------===//
// ConvBackwardFusion
//===----------------------------------------------------------------------===//

def ConvBackwardFusion : Pass<"fuse-conv-backward", "mlir::func::FuncOp"> {
  let summary = "Fuse convolution backward data & filter";
  let constructor = "mlir::createConvBackwardFusionPass()";
}

//===----------------------------------------------------------------------===//
// ConvForwardFusion
//===----------------------------------------------------------------------===//

def ConvForwardFusion : Pass<"fuse-conv-forward", "mlir::func::FuncOp"> {
  let summary = "Fuse convolution forward, such as conv with bias activation";
  let constructor = "mlir::createConvForwardFusionPass()";
}

//===----------------------------------------------------------------------===//
// Convert Rng To CustomCall
//===----------------------------------------------------------------------===//

def ConvertOpToCustomCall : Pass<"convert-op-to-customcall", "ModuleOp"> {
  let summary = "Convert op to mhlo.custom_call";
  let constructor = "mlir::createConvertOpToCustomCallPass()";
  let options = [
    Option<"anchorTag", "anchor-tag", "std::string",
            /*default=*/"",
            "Optional unitAttr anchored tag to apply this pass">
  ];
}

//===----------------------------------------------------------------------===//
// ConvertFuncToCustomCall // No command-line
//===----------------------------------------------------------------------===//

//def ConvertFuncToCustomCall : Pass<"convert-func-to-custom", "mlir::ModuleOp"> {
//  let summary = "Convert Func to CustomCall";
//  let constructor = "mlir::createConvertFuncToCustomCallPass()";
//}

//===----------------------------------------------------------------------===//
// ConvertInsertion (for CallOps and FunOps now)  // No command-line
//===----------------------------------------------------------------------===//

//def ConvertInsertion : Pass<"insert-convert", "ModuleOp"> {
//  let summary = "Insert ConvertOps (for CallOps and func::FuncOps now)";
//  let constructor = "mlir::createConvertInsertionPass()";
//  let dependentDialects = [
//    "mlir::mhlo::MhloDialect"
//  ];
//}

//===----------------------------------------------------------------------===//
// DotTransposeFusion : this pass should run after HloTransposeDotToDotGeneral
//===----------------------------------------------------------------------===//

def DotTransposeFusion : Pass<"fuse-dot-transpose", "mlir::func::FuncOp"> {
  let summary = "Fuse dot/dot_general + transpose";
  let constructor = "mlir::createDotTransposeFusionPass()";
}

//===----------------------------------------------------------------------===//
// DynamicShapeClustering
//===----------------------------------------------------------------------===//

def DynamicShapeClustering : Pass<"dynamic-shape-clustering", "mlir::ModuleOp"> {
  let summary = "Dynamic shape clustering";
  let description = [{
    Clustering based on the shape information. After the pass, the dynamic 
    source are only from function arguments.
  }];
  let constructor = "mlir::createDynamicShapeClusteringPass()";
  let options = [
    Option<"anchorAttr", "anchor-attr", "std::string",
            /*default=*/"",
            "An optional Unit attribute anchoring on target functions.">,
  ];
}

//===----------------------------------------------------------------------===//
// FuncArgRearrangement // No command-line
//===----------------------------------------------------------------------===//

// def FuncArgRearrangement : Pass<"rearrange-func-arg", "ModuleOp"> {
//  let summary = "Func Arg Rearrangement: pack, reorder args and returns of func";
//  let constructor = "mlir::createFuncArgRearrangementPass()";
//  let options = [
//    Option<"anchorAttr", "anchor-attr", "std::string",
//            /*default=*/"",
//            "An optional Unit attribute anchoring on target functions.">,
//  ];
//}

//===----------------------------------------------------------------------===//
// FuseBMMDimension
//===----------------------------------------------------------------------===//

def FuseBMMDimension : Pass<"fuse-bmm-dimension", "func::FuncOp"> {
  let summary = "rewrite bmm operators as 3d bmm, may add reshapeOp";
  let constructor = "mlir::createFuseBMMDimensionPass()";
}

//===----------------------------------------------------------------------===//
// FusionOutlining
//===----------------------------------------------------------------------===//

def FusionOutlining : Pass<"fusion-outlining", "ModuleOp"> {
  let summary = "Outline mhlo FusionOp to a func::FuncOp";
  let constructor = "mlir::createFusionOutliningPass()";
  let dependentDialects = [
    "mlir::func::FuncDialect"
  ];
}

//===----------------------------------------------------------------------===//
// HloFolder
//===----------------------------------------------------------------------===//

// TODO(liuyuanqiang): re-implement this as folder
def HloFolder : Pass<"hlo-fold", "mlir::func::FuncOp"> {
  let summary = "Fold mhlo ops";
  let constructor = "mlir::createHloFolderPass()";
}

//===----------------------------------------------------------------------===//
// HloTransposeDotToDotGeneral
//===----------------------------------------------------------------------===//

def HloTransposeDotToDotGeneral : Pass<"hlo-transpose-dot-to-dot-general", "mlir::func::FuncOp"> {
  let summary = "Fuse transpose dot to dot_general";
  let constructor = "mlir::createHloTransposeDotToDotGeneralPass()";
}

//===----------------------------------------------------------------------===//
// IOConvertFusion
//===----------------------------------------------------------------------===//

def IOConvertFusion : Pass<"fuse-io-convert", "mlir::func::FuncOp"> {
  let summary = "Fuse op with Input/Output convert";
  let constructor = "mlir::createIOConvertFusionPass()";
}

//===----------------------------------------------------------------------===//
// MatmulLayoutTransform
//===----------------------------------------------------------------------===//

def MatmulLayoutTransform: Pass<"matmul-layout-transform", "mlir::func::FuncOp"> {
  let summary = "Transform matmul layout to a specific one.";
  let constructor = "mlir::createMatmulLayoutTransformPass()";
  let options = [
    Option<"transposeConstantOnly", "transpose-const-only", "bool", /*default=*/"true",
           "Only insert transpose op for constants if set true">,
    Option<"targetLayout", "target-layout", "std::string", /*default=*/"",
           "Target Layout">,
  ];
}

//===----------------------------------------------------------------------===//
// HloMoveDown
//===----------------------------------------------------------------------===//

def HloMoveDown : Pass<"hlo-move-down", "mlir::func::FuncOp"> {
  let summary = "Move selected mhlo op down (to output)";
  let constructor = "mlir::createHloMoveDownPass()";
  let options = [
    Option<"allMultiUser", "all-multi-user", 
           "bool", /*default=*/"false",
           "whether to support multiple users and require all user legal">,
    Option<"multiUser", "multi-user", 
           "bool", /*default=*/"false",
           "whether to support multiple users, it might be overridden by all-multi-user">,
  ];
}

//===----------------------------------------------------------------------===//
// SliceMoveDownAndMerge
//===----------------------------------------------------------------------===//

def SliceMoveDownAndMerge : Pass<"slice-move-down-and-merge", "mlir::func::FuncOp"> {
  let summary = "Slice move down and merge branch";
  let constructor = "mlir::createSliceMoveDownAndMergePass()";
}

//===----------------------------------------------------------------------===//
// HloMoveUp
//===----------------------------------------------------------------------===//

def HloMoveUp : Pass<"hlo-move-up", "mlir::func::FuncOp"> {
  let summary = "Move selected mhlo op up (to input)";
  let constructor = "mlir::createHloMoveUpPass()";
  let options = [
    Option<"multiInput", "multi-input", 
           "bool", /*default=*/"false",
           "whether to support multiple inputs">,
  ];
}

//===----------------------------------------------------------------------===//
// InsertShapeConstraint
//===----------------------------------------------------------------------===//
def InsertShapeConstraint : Pass<"insert-shape-constraint", "func::FuncOp"> {
  let summary = "Insert shape constraint for some special operations";
  let description = [{
    Some operations have constraints on the inputs and(or) outputs. Ex. mhlo.add
    requires that all the inputs and outputs' shape should be same. And a MeetOp 
    will be created to express the constraint.
  }];
  let constructor = "mlir::createInsertShapeConstraintPass()";
  let dependentDialects = [
    "mlir::shape::ShapeDialect",
    "mlir::shape_ext::ShapeExtDialect",
    "mlir::tensor::TensorDialect"
  ];
}

//===----------------------------------------------------------------------===//
// LayoutTransformation
//===----------------------------------------------------------------------===//

def LayoutTransformation : Pass<"transform-layout", "mlir::func::FuncOp"> {
  let summary = "Layout Transformation: Conv2d, BatchNormTraining";
  let constructor = "mlir::createLayoutTransformationPass()";
  let options = [
    Option<"targetLayout", "target-layout", "std::string", /*default=*/"",
           "Target Layout">,
  ];
}

//===----------------------------------------------------------------------===//
// ReduceFusion
//===----------------------------------------------------------------------===//

def ReduceFusion : Pass<"fuse-reduce", "mlir::func::FuncOp"> {
  let summary = "Fuse reduce and reduce-window ops";
  let constructor = "mlir::createReduceFusionPass()";
}

//===----------------------------------------------------------------------===//
// ReshapeGather
//===----------------------------------------------------------------------===//

def ReshapeGather : Pass<"reshape-gather", "mlir::func::FuncOp"> {
  let summary = "Reshape gather index to be 1D";
  let constructor = "mlir::createReshapeGatherPass()";
}

//===----------------------------------------------------------------------===//
// RewriteWithConstraint
//===----------------------------------------------------------------------===//

def RewriteWithConstraint :  Pass<"rewrite-with-constraint", "mlir::func::FuncOp"> {
  let summary = "Rewrite operation with constraint";
  let constructor = "mlir::createRewriteWithConstraintPass()";
}

//===----------------------------------------------------------------------===//
// Static Shape Inference
//===----------------------------------------------------------------------===//

def StaticShapeInference : Pass<"static-shape-infer", "func::FuncOp"> {
  let summary = "Static shape inference";
  let constructor = "mlir::createStaticShapeInferencePass()";
  let options = [
    Option<"overrideShape", "override-shape", 
           "bool", /*default=*/"true",
           "whether to override shape with infered shape">,
  ];
}

//===----------------------------------------------------------------------===//
// TrivialFusion
//===----------------------------------------------------------------------===//

def TrivialFusion : Pass<"fuse-trivial", "mlir::func::FuncOp"> {
  let summary = "Fuse trivial single ops";
  let constructor = "mlir::createTrivialFusionPass()";
}

//===----------------------------------------------------------------------===//
// Unfuse BatchNorm
//===----------------------------------------------------------------------===//

def UnfuseBatchNorm : Pass<"unfuse-batch-norm", "mlir::func::FuncOp"> {
  let summary = "unfuse batch norm into multiplication and addition";
  let constructor = "mlir::createUnfuseBatchNormPass()";
}

#endif // BYTEIR_DIALECT_MHLO_PASSES