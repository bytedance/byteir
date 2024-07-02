// RUN: byteir-opt %s -resolve-shape-constraint | FileCheck %s

func.func @meet_const(%arg0 : tensor<?x4xf32>, %arg1 : tensor<?x4xf32>) -> tensor<?x4xf32> {
  %0 = shape.shape_of %arg0 : tensor<?x4xf32> -> tensor<2xindex>
  %a = shape.value_as_shape %0 : tensor<2xindex> -> !shape.shape
  %1 = shape.shape_of %arg1 : tensor<?x4xf32> -> tensor<2xindex>
  %b = shape.value_as_shape %1 : tensor<2xindex> -> !shape.shape
  %c0 = arith.constant 0 : index
  %c = shape.const_size 1024
  %a0 = shape.get_extent %a, %c0 : !shape.shape, index -> !shape.size
  %b0 = shape.get_extent %b, %c0 : !shape.shape, index -> !shape.size
  %sum = shape.add %a0, %b0 : !shape.size, !shape.size -> !shape.size
  "shape_ext.meet"(%c, %sum) : (!shape.size, !shape.size) -> ()
  %result = "mhlo.concatenate"(%arg0, %arg1) { dimension = 0 : i64 } : (tensor<?x4xf32>, tensor<?x4xf32>) -> tensor<?x4xf32>
  "shape_ext.tie"(%result, %sum) : (tensor<?x4xf32>, !shape.size) -> ()
  return %result : tensor<?x4xf32>
}
// CHECK-LABEL: @meet_const(%arg0: tensor<?x4xf32>, %arg1: tensor<?x4xf32>) -> tensor<1024x4xf32>
// CHECK-NEXT: %0 = "mhlo.concatenate"(%arg0, %arg1) <{dimension = 0 : i64}> : (tensor<?x4xf32>, tensor<?x4xf32>) -> tensor<1024x4xf32>
// CHECK-NEXT: return %0 : tensor<1024x4xf32>

func.func @einsum_shape_constraint(%arg0: tensor<?x2x2xf32>, %arg1: tensor<2x2x3xf32>, %arg2: tensor<2x2x2xf32>) -> tensor<2x2x2x3xf32> {
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %0 = tensor.dim %arg0, %c0 : tensor<?x2x2xf32>
    "shape_ext.tie"(%arg0, %0) : (tensor<?x2x2xf32>, index) -> ()
    %1 = "mhlo.einsum"(%arg1, %arg0) {einsum_config = "edc,bqe->dcbq"} : (tensor<2x2x3xf32>, tensor<?x2x2xf32>) -> tensor<2x3x?x2xf32>
    "shape_ext.meet"(%c2, %c2) : (index, index) -> ()
    "shape_ext.meet"(%c3, %c3) : (index, index) -> ()
    %2 = tensor.dim %1, %c2 : tensor<2x3x?x2xf32>
    "shape_ext.meet"(%0, %2) : (index, index) -> ()
    "shape_ext.meet"(%c2, %c2) : (index, index) -> ()
    "shape_ext.meet"(%c2, %c2) : (index, index) -> ()
    "shape_ext.tie"(%1, %2) : (tensor<2x3x?x2xf32>, index) -> ()
    %3 = "mhlo.einsum"(%1, %arg2) {einsum_config = "dcbq,btd->bqtc"} : (tensor<2x3x?x2xf32>, tensor<2x2x2xf32>) -> tensor<2x2x2x3xf32>
    "shape_ext.meet"(%c2, %2) : (index, index) -> ()
    "shape_ext.meet"(%c2, %c2) : (index, index) -> ()
    "shape_ext.meet"(%c2, %c2) : (index, index) -> ()
    "shape_ext.meet"(%c2, %c2) : (index, index) -> ()
    "shape_ext.meet"(%c3, %c3) : (index, index) -> ()
    "shape_ext.meet"(%c2, %c2) : (index, index) -> ()
    return %3 : tensor<2x2x2x3xf32>
}
// CHECK-LABEL: @einsum_shape_constraint(%arg0: tensor<2x2x2xf32>
// CHECK-NEXT: %0 = "mhlo.einsum"(%arg1, %arg0) <{einsum_config = "edc,bqe->dcbq"}> : (tensor<2x2x3xf32>, tensor<2x2x2xf32>) -> tensor<2x3x2x2xf32>
// CHECK-NEXT: %1 = "mhlo.einsum"(%0, %arg2) <{einsum_config = "dcbq,btd->bqtc"}> : (tensor<2x3x2x2xf32>, tensor<2x2x2xf32>) -> tensor<2x2x2x3xf32>
// CHECK-NEXT: return %1 : tensor<2x2x2x3xf32>

func.func @concat(%arg0: tensor<?x4x?xf32>, %arg1: tensor<2x?x?xf32>) -> tensor<?x4x?xf32> {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %dim = tensor.dim %arg1, %c1 : tensor<2x?x?xf32>
  %dim_0 = tensor.dim %arg1, %c2 : tensor<2x?x?xf32>
  "shape_ext.tie"(%arg1, %dim, %dim_0) : (tensor<2x?x?xf32>, index, index) -> ()
  %dim_1 = tensor.dim %arg0, %c0 : tensor<?x4x?xf32>
  %dim_2 = tensor.dim %arg0, %c2 : tensor<?x4x?xf32>
  "shape_ext.tie"(%arg0, %dim_1, %dim_2) : (tensor<?x4x?xf32>, index, index) -> ()
  %0 = "mhlo.concatenate"(%arg0, %arg1) {dimension = 0 : i64} : (tensor<?x4x?xf32>, tensor<2x?x?xf32>) -> tensor<?x4x?xf32>
  "shape_ext.meet"(%c4, %dim) : (index, index) -> ()
  "shape_ext.meet"(%dim_2, %dim_0) : (index, index) -> ()
  %1 = arith.addi %dim_1, %c2 : index
  "shape_ext.tie"(%0, %1, %dim_2) : (tensor<?x4x?xf32>, index, index) -> ()
  return %0 : tensor<?x4x?xf32>
}
// CHECK: func.func @concat(%arg0: tensor<?x4x?xf32>, %arg1: tensor<2x4x?xf32>) -> tensor<?x4x?xf32>
