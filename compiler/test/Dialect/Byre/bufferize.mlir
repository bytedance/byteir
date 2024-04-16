// RUN: byteir-opt -byteir-one-shot-bufferize -split-input-file %s | FileCheck %s

func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) attributes {__placeholder__byre.entry_point} {
  %0 = tensor.empty() : tensor<4xf32>
  %1 = tensor.empty() : tensor<4xf32>
  %2:2 = byre.compute_on_tensor @some_kernel ins(%arg0, %arg1 : tensor<4xf32>, tensor<4xf32>) outs(%0, %1 : tensor<4xf32>, tensor<4xf32>) : tensor<4xf32>, tensor<4xf32>
  return %2#0, %2#1 : tensor<4xf32>, tensor<4xf32>
}

// CHECK-LABEL: func.func @main
// CHECK-NEXT: %[[ALLOC:.*]] = memref.alloc() : memref<4xf32>
// CHECK-NEXT: %[[ALLOC0:.*]] = memref.alloc() : memref<4xf32>
// CHECK-NEXT: byre.compute @some_kernel(%arg0, %arg1, %[[ALLOC]], %[[ALLOC0]])
// CHECK-SAME: {memory_effects = [1 : i32, 1 : i32, 2 : i32, 2 : i32]} : memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>

// -----

func.func @forward(%arg0: tensor<?x20xf32>, %arg1: tensor<20x?xf32>) -> tensor<?x?xf32> attributes {__placeholder__byre.entry_point} {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?x20xf32>
  %dim_0 = tensor.dim %arg1, %c1 : tensor<20x?xf32>
  %0 = tensor.empty(%dim, %dim_0) : tensor<?x?xf32>
  %1 = byre.compute_on_tensor @MatmulOp {lhs_contracting_dimension = 1 : i64, rhs_contracting_dimension = 0 : i64} ins(%arg0, %arg1 : tensor<?x20xf32>, tensor<20x?xf32>) outs(%0 : tensor<?x?xf32>) : tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}

// CHECK-LABEL: func.func @forward
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-NEXT: %[[DIM:.*]] = memref.dim %arg0, %[[C0]] : memref<?x20xf32>
// CHECK-NEXT: %[[DIM0:.*]] = memref.dim %arg1, %[[C1]] : memref<20x?xf32>
// CHECK-NEXT: %[[ALLOC:.*]] = memref.alloc(%[[DIM]], %[[DIM0]]) : memref<?x?xf32>
// CHECK-NEXT: byre.compute @MatmulOp(%arg0, %arg1, %[[ALLOC]])
// CHECK-SAME: {lhs_contracting_dimension = 1 : i64, memory_effects = [1 : i32, 1 : i32, 2 : i32], rhs_contracting_dimension = 0 : i64} : memref<?x20xf32>, memref<20x?xf32>, memref<?x?xf32>

// -----

func.func @slice_gemm(%arg0: tensor<768x1024xf32>) -> tensor<256x512xf32> attributes {__placeholder__byre.entry_point} {
  %cst = arith.constant dense_resource<__elided__> : tensor<512x512xf32>
  %extracted_slice = tensor.extract_slice %arg0[0, 0] [256, 512] [1, 2] : tensor<768x1024xf32> to tensor<256x512xf32>
  %0 = tensor.empty() : tensor<256x512xf32>
  %1 = byre.compute_on_tensor @MatmulOp {lhs_contracting_dimension = 1 : i64, rhs_contracting_dimension = 0 : i64} ins(%extracted_slice, %cst : tensor<256x512xf32>, tensor<512x512xf32>) outs(%0 : tensor<256x512xf32>) : tensor<256x512xf32>
  return %1 : tensor<256x512xf32>
}

// CHECK-LABEL: func.func @slice_gemm
// CHECK: %[[SUBVIEW:.*]] = memref.subview
// CHECK-NEXT: %[[ALLOC:.*]] = memref.alloc() : memref<256x512xf32>
// CHECK-NEXT: %[[ALLOC0:.*]] = memref.alloc() : memref<256x512xf32>
// CHECK-NEXT: memref.copy %[[SUBVIEW]], %[[ALLOC0]] : memref<256x512xf32, strided<[1024, 2]>> to memref<256x512xf32>
// CHECK-NEXT: byre.compute @MatmulOp(%[[ALLOC0]], %0, %[[ALLOC]])
