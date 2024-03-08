// RUN: byteir-opt %s -set-arg-space="entry-func=main all-space=cpu" --split-input-file | FileCheck %s

func.func private @nested(%arg0 : memref<2x4xf32>, %arg1 : memref<2x4xf32>) -> (memref<2x4xf32>) attributes {device = "gpu"}
// CHECK-LABEL: func.func private @nested(memref<2x4xf32, "gpu">, memref<2x4xf32, "gpu">) -> memref<2x4xf32, "gpu"> attributes {device = "gpu"}

func.func private @local(%arg0 : memref<2x4xf32>, %arg1 : memref<2x4xf32>) -> (memref<2x4xf32>) attributes {device = "gpu"}  {
  %0 = call @nested(%arg0, %arg1) : (memref<2x4xf32>, memref<2x4xf32>) -> (memref<2x4xf32>)
  %1 = memref.alloc() : memref<2x4xf32>
  %2 = memref.alloc() : memref<2x4xf32>
  "lmhlo.abs"(%arg0, %arg1) : (memref<2x4xf32>, memref<2x4xf32>) -> ()
  "lmhlo.abs"(%arg1, %1) : (memref<2x4xf32>, memref<2x4xf32>) -> ()
  "lmhlo.abs"(%1, %2) : (memref<2x4xf32>, memref<2x4xf32>) -> ()
  return %2: memref<2x4xf32> 
}
// CHECK-LABEL: func.func private @local(%arg0: memref<2x4xf32, "gpu">, %arg1: memref<2x4xf32, "gpu">) -> memref<2x4xf32, "gpu"> attributes {device = "gpu"}
// CHECK-NEXT:    %0 = call @nested(%arg0, %arg1) : (memref<2x4xf32, "gpu">, memref<2x4xf32, "gpu">) -> memref<2x4xf32, "gpu">
// CHECK-NEXT:    %alloc = memref.alloc() : memref<2x4xf32, "gpu">
// CHECK-NEXT:    %alloc_0 = memref.alloc() : memref<2x4xf32, "gpu">
// CHECK-NEXT:    "lmhlo.abs"(%arg0, %arg1) {device = "gpu"} : (memref<2x4xf32, "gpu">, memref<2x4xf32, "gpu">) -> ()
// CHECK-NEXT:    "lmhlo.abs"(%arg1, %alloc) {device = "gpu"} : (memref<2x4xf32, "gpu">, memref<2x4xf32, "gpu">) -> ()
// CHECK-NEXT:    "lmhlo.abs"(%alloc, %alloc_0) {device = "gpu"} : (memref<2x4xf32, "gpu">, memref<2x4xf32, "gpu">) -> ()
// CHECK-NEXT:    return %alloc_0 : memref<2x4xf32, "gpu">


func.func @main(%arg0 : memref<2x4xf32>, %arg1 : memref<2x4xf32>, %arg2 : memref<2x4xf32>) -> (memref<2x4xf32>, memref<2x4xf32>) {
  %0 = call @local(%arg0, %arg0) : (memref<2x4xf32>, memref<2x4xf32>) -> (memref<2x4xf32>)
  "lmhlo.add"(%arg0, %arg1, %arg2) : (memref<2x4xf32>, memref<2x4xf32>, memref<2x4xf32>) -> ()
  return %0, %0: memref<2x4xf32>, memref<2x4xf32> 
}
// CHECK-LABEL: func.func @main(%arg0: memref<2x4xf32, "cpu">, %arg1: memref<2x4xf32, "cpu">, %arg2: memref<2x4xf32, "cpu">) -> (memref<2x4xf32, "cpu">, memref<2x4xf32, "cpu">)
// CHECK-NEXT:    %alloc = memref.alloc() : memref<2x4xf32, "gpu">
// CHECK-NEXT:    memref.copy %arg0, %alloc : memref<2x4xf32, "cpu"> to memref<2x4xf32, "gpu">
// CHECK-NEXT:    %0 = call @local(%alloc, %alloc) : (memref<2x4xf32, "gpu">, memref<2x4xf32, "gpu">) -> memref<2x4xf32, "gpu">
// CHECK-NEXT:    %alloc_0 = memref.alloc() : memref<2x4xf32, "cpu">
// CHECK-NEXT:    memref.copy %0, %alloc_0 : memref<2x4xf32, "gpu"> to memref<2x4xf32, "cpu">
// CHECK-NEXT:    "lmhlo.add"(%arg0, %arg1, %arg2) : (memref<2x4xf32, "cpu">, memref<2x4xf32, "cpu">, memref<2x4xf32, "cpu">) -> ()
// CHECK-NEXT:    return %alloc_0, %alloc_0 : memref<2x4xf32, "cpu">, memref<2x4xf32, "cpu">

// -----

func.func private @device1(%arg : memref<2x4xf32>) -> memref<2x4xf32> attributes {device = "device1"}
func.func private @device0(%arg : memref<2x4xf32>) -> memref<2x4xf32> attributes {device = "device0"}

func.func @main(%arg0 : memref<2x4xf32>) -> memref<2x4xf32> {
  %0 = call @device0(%arg0) : (memref<2x4xf32>) -> (memref<2x4xf32>)
  %1 = call @device1(%0) : (memref<2x4xf32>) -> (memref<2x4xf32>)
  return %1: memref<2x4xf32>
}
// CHECK-LABEL: func.func @main(%arg0: memref<2x4xf32, "cpu">) -> memref<2x4xf32, "cpu">
//  CHECK-NEXT:     %alloc = memref.alloc() : memref<2x4xf32, "device0">
//  CHECK-NEXT:     memref.copy %arg0, %alloc : memref<2x4xf32, "cpu"> to memref<2x4xf32, "device0">
//  CHECK-NEXT:     %0 = call @device0(%alloc) : (memref<2x4xf32, "device0">) -> memref<2x4xf32, "device0">
//  CHECK-NEXT:     %alloc_0 = memref.alloc() : memref<2x4xf32, "device1">
//  CHECK-NEXT:     memref.copy %0, %alloc_0 : memref<2x4xf32, "device0"> to memref<2x4xf32, "device1">
//  CHECK-NEXT:     %1 = call @device1(%alloc_0) : (memref<2x4xf32, "device1">) -> memref<2x4xf32, "device1">
//  CHECK-NEXT:     %alloc_1 = memref.alloc() : memref<2x4xf32, "cpu">
//  CHECK-NEXT:     memref.copy %1, %alloc_1 : memref<2x4xf32, "device1"> to memref<2x4xf32, "cpu">
//  CHECK-NEXT:     return %alloc_1 : memref<2x4xf32, "cpu">

// -----

func.func private @foo(%arg : memref<8xf32>) -> memref<8xf32> attributes {device = "gpu"}

func.func @main(%arg0 : memref<2x4xf32>) -> memref<8xf32> {
  %0 = memref.collapse_shape %arg0 [[0, 1]] : memref<2x4xf32> into memref<8xf32>
  %1 = call @foo(%0) : (memref<8xf32>) -> memref<8xf32>
  return %1 : memref<8xf32>
}
// CHECK-LABEL: func.func @main
//  CHECK-SAME: (%[[ARG:.+]]: memref<2x4xf32, "cpu">) -> memref<8xf32, "cpu">
//  CHECK-NEXT:     %[[COLLAPSED:.+]] = memref.collapse_shape %[[ARG]] {{\[}}[0, 1]] : memref<2x4xf32, "cpu"> into memref<8xf32, "cpu">
//  CHECK-NEXT:     %[[ALLOC0:.+]] = memref.alloc() : memref<8xf32, "gpu">
//  CHECK-NEXT:     memref.copy %[[COLLAPSED]], %[[ALLOC0]] : memref<8xf32, "cpu"> to memref<8xf32, "gpu">
//  CHECK-NEXT:     %[[CALL:.+]] = call @foo(%[[ALLOC0]]) : (memref<8xf32, "gpu">) -> memref<8xf32, "gpu">
//  CHECK-NEXT:     %[[ALLOC1:.+]] = memref.alloc() : memref<8xf32, "cpu">
//  CHECK-NEXT:     memref.copy %[[CALL]], %[[ALLOC1]] : memref<8xf32, "gpu"> to memref<8xf32, "cpu">
//  CHECK-NEXT:     return %[[ALLOC1]] : memref<8xf32, "cpu">

// -----

func.func private @foo(%arg : memref<2x4xf32>) -> memref<2x4xf32> attributes {device = "gpu"}

func.func @main(%arg0 : memref<2x4xf32>) -> memref<8xf32> {
  %0 = call @foo(%arg0) : (memref<2x4xf32>) -> memref<2x4xf32>
  %1 = memref.collapse_shape %0 [[0, 1]] : memref<2x4xf32> into memref<8xf32>
  return %1 : memref<8xf32>
}
// CHECK-LABEL: func.func @main
//  CHECK-SAME: (%[[ARG:.+]]: memref<2x4xf32, "cpu">) -> memref<8xf32, "cpu">
//  CHECK-NEXT:     %[[ALLOC0:.+]] = memref.alloc() : memref<2x4xf32, "gpu">
//  CHECK-NEXT:     memref.copy %[[ARG]], %[[ALLOC0]] : memref<2x4xf32, "cpu"> to memref<2x4xf32, "gpu">
//  CHECK-NEXT:     %[[CALL:.+]] = call @foo(%[[ALLOC0]]) : (memref<2x4xf32, "gpu">) -> memref<2x4xf32, "gpu">
//  CHECK-NEXT:     %[[COLLAPSED:.+]] = memref.collapse_shape %[[CALL]] {{\[}}[0, 1]] : memref<2x4xf32, "gpu"> into memref<8xf32, "gpu">
//  CHECK-NEXT:     %[[ALLOC1:.+]] = memref.alloc() : memref<8xf32, "cpu">
//  CHECK-NEXT:     memref.copy %[[COLLAPSED]], %[[ALLOC1]] : memref<8xf32, "gpu"> to memref<8xf32, "cpu">
//  CHECK-NEXT:     return %[[ALLOC1]] : memref<8xf32, "cpu">

// -----

func.func private @foo(%arg : memref<2x4xf32>) -> memref<2x4xf32> attributes {device = "gpu"}

func.func @main(%arg0 : memref<2x4xf32>) -> memref<8xf32> {
  %0 = call @foo(%arg0) : (memref<2x4xf32>) -> memref<2x4xf32>
  %1 = memref.collapse_shape %0 [[0, 1]] : memref<2x4xf32> into memref<8xf32>
  return %1 : memref<8xf32>
}
// CHECK-LABEL: func.func @main
//  CHECK-SAME: (%[[ARG:.+]]: memref<2x4xf32, "cpu">) -> memref<8xf32, "cpu">
//  CHECK-NEXT:     %[[ALLOC0:.+]] = memref.alloc() : memref<2x4xf32, "gpu">
//  CHECK-NEXT:     memref.copy %[[ARG]], %[[ALLOC0]] : memref<2x4xf32, "cpu"> to memref<2x4xf32, "gpu">
//  CHECK-NEXT:     %[[CALL:.+]] = call @foo(%[[ALLOC0]]) : (memref<2x4xf32, "gpu">) -> memref<2x4xf32, "gpu">
//  CHECK-NEXT:     %[[COLLAPSED:.+]] = memref.collapse_shape %[[CALL]] {{\[}}[0, 1]] : memref<2x4xf32, "gpu"> into memref<8xf32, "gpu">
//  CHECK-NEXT:     %[[ALLOC1:.+]] = memref.alloc() : memref<8xf32, "cpu">
//  CHECK-NEXT:     memref.copy %[[COLLAPSED]], %[[ALLOC1]] : memref<8xf32, "gpu"> to memref<8xf32, "cpu">
//  CHECK-NEXT:     return %[[ALLOC1]] : memref<8xf32, "cpu">

// -----

func.func private @foo(%arg : memref<2x4xf32>) -> memref<2x4xf32> attributes {device = "gpu"}

func.func @main(%arg0 : memref<2x4xf32>) -> memref<8xf32> {
  %0 = call @foo(%arg0) : (memref<2x4xf32>) -> memref<2x4xf32>
  %1 = memref.collapse_shape %0 [[0, 1]] {device = "cpu"} : memref<2x4xf32> into memref<8xf32>
  return %1 : memref<8xf32>
}
// CHECK-LABEL: func.func @main
//  CHECK-SAME: (%[[ARG:.+]]: memref<2x4xf32, "cpu">) -> memref<8xf32, "cpu">
//  CHECK-NEXT:     %[[ALLOC0:.+]] = memref.alloc() : memref<2x4xf32, "gpu">
//  CHECK-NEXT:     memref.copy %[[ARG]], %[[ALLOC0]] : memref<2x4xf32, "cpu"> to memref<2x4xf32, "gpu">
//  CHECK-NEXT:     %[[CALL:.+]] = call @foo(%[[ALLOC0]]) : (memref<2x4xf32, "gpu">) -> memref<2x4xf32, "gpu">
//  CHECK-NEXT:     %[[ALLOC1:.+]] = memref.alloc() : memref<2x4xf32, "cpu">
//  CHECK-NEXT:     memref.copy %[[CALL]], %[[ALLOC1]] : memref<2x4xf32, "gpu"> to memref<2x4xf32, "cpu">
//  CHECK-NEXT:     %[[COLLAPSED:.+]] = memref.collapse_shape %[[ALLOC1]] {{\[}}[0, 1]] {device = "cpu"} : memref<2x4xf32, "cpu"> into memref<8xf32, "cpu">
//  CHECK-NEXT:     return %[[COLLAPSED]] : memref<8xf32, "cpu">

// -----

memref.global "private" constant @__constant : memref<2x4xf32> = dense<0.0>
func.func private @foo0(%arg0 : memref<2x4xf32>, %arg1 : memref<2x4xf32>) -> memref<2x4xf32> attributes {device = "device0"}
func.func private @foo1(%arg0 : memref<2x4xf32>, %arg1 : memref<2x4xf32>) -> memref<2x4xf32> attributes {device = "device1"}

func.func @main(%arg0 : memref<2x4xf32>) -> (memref<2x4xf32>, memref<2x4xf32>) {
  %0 = memref.get_global @__constant : memref<2x4xf32>
  %1 = call @foo0(%arg0, %0) : (memref<2x4xf32>, memref<2x4xf32>) -> memref<2x4xf32>
  %2 = memref.get_global @__constant : memref<2x4xf32>
  %3 = call @foo1(%arg0, %2) : (memref<2x4xf32>, memref<2x4xf32>) -> memref<2x4xf32>
  return %1, %3 : memref<2x4xf32>, memref<2x4xf32>
}
// CHECK-LABEL: func.func @main
//   CHECK: memref.get_global @__constant_device0
//     CHECK-SAME: memref<2x4xf32, "device0">
//   CHECK: memref.get_global @__constant_device1
//     CHECK-SAME: memref<2x4xf32, "device1">
// CHECK-LABEL: memref.global "private" constant @__constant_device0 : memref<2x4xf32, "device0">
// CHECK-LABEL: memref.global "private" constant @__constant_device1 : memref<2x4xf32, "device1">

// -----

func.func private @device0(%arg : memref<?x4xf32>) -> memref<2x4xf32> attributes {device = "device0"}
func.func private @device1(%arg : memref<2x4xf32>) -> memref<?x4xf32> attributes {device = "device1"}

func.func @main(%arg0 : memref<?x4xf32>) -> (memref<?x4xf32>) {
  %1 = call @device0(%arg0) : (memref<?x4xf32>) -> (memref<2x4xf32>)
  %2 = call @device1(%1) : (memref<2x4xf32>) -> (memref<?x4xf32>)
  return  %2:  memref<?x4xf32>
}

// CHECK-LABEL: func.func @main
// CHECK-SAME: (%[[ARG0:.+]]: memref<?x4xf32, "cpu">)
// CHECK-NEXT: %[[C0:.+]] = arith.constant 0 : index
// CHECK-NEXT: %[[DIM:.+]] = memref.dim %[[ARG0]], %[[C0]] : memref<?x4xf32, "cpu">
// CHECK-NEXT: %[[ALLOC:.+]] = memref.alloc(%[[DIM]]) : memref<?x4xf32, "device0">
// CHECK-NEXT: memref.copy %[[ARG0]], %[[ALLOC]] : memref<?x4xf32, "cpu"> to memref<?x4xf32, "device0">
// CHECK-NEXT: %[[V0:.+]] = call @device0(%[[ALLOC]]) : (memref<?x4xf32, "device0">) -> memref<2x4xf32, "device0">
// CHECK-NEXT: %[[ALLOC0:.+]] = memref.alloc() : memref<2x4xf32, "device1">
// CHECK-NEXT: memref.copy %[[V0]], %[[ALLOC0]] : memref<2x4xf32, "device0"> to memref<2x4xf32, "device1">
// CHECK-NEXT: %[[V1:.+]] = call @device1(%[[ALLOC0]]) : (memref<2x4xf32, "device1">) -> memref<?x4xf32, "device1">
// CHECK-NEXT: %[[C01:.+]] = arith.constant 0 : index
// CHECK-NEXT: %[[DIM2:.+]] = memref.dim %[[V1]], %[[C01]] : memref<?x4xf32, "device1">
// CHECK-NEXT: %[[ALLOC3:.+]] = memref.alloc(%[[DIM2]]) : memref<?x4xf32, "cpu">
// CHECK-NEXT: memref.copy %[[V1]], %[[ALLOC3]] : memref<?x4xf32, "device1"> to memref<?x4xf32, "cpu">
// CHECK-NEXT: return %[[ALLOC3]] : memref<?x4xf32, "cpu">


// -----


func.func private @device0(%arg : memref<?x4xf32>) -> memref<2x4xf32> attributes {device = "device0"}

func.func @main(%arg0 : memref<?x4xf32>) -> (memref<2x4xf32>, memref<2x4xf32>) {
  %0 = call @device0(%arg0) : (memref<?x4xf32>) -> (memref<2x4xf32>)
  %1 = call @device0(%arg0) : (memref<?x4xf32>) -> (memref<2x4xf32>)
  return  %0, %1:  memref<2x4xf32>, memref<2x4xf32>
}

// CHECK-LABEL: func.func @main
// CHECK-SAME: (%[[ARG0:.+]]: memref<?x4xf32, "cpu">)
// CHECK-NEXT: %[[C0:.+]] = arith.constant 0 : index
// CHECK-NEXT: %[[DIM:.+]] = memref.dim %[[ARG0]], %[[C0]] : memref<?x4xf32, "cpu">
// CHECK-NEXT: %[[ALLOC:.+]] = memref.alloc(%[[DIM]]) : memref<?x4xf32, "device0">
// CHECK-NEXT: memref.copy %[[ARG0]], %[[ALLOC]] : memref<?x4xf32, "cpu"> to memref<?x4xf32, "device0">
// CHECK-NEXT: %[[V0:.+]] = call @device0(%[[ALLOC]]) : (memref<?x4xf32, "device0">) -> memref<2x4xf32, "device0">
// CHECK-NEXT: %[[ALLOC0:.+]] = memref.alloc() : memref<2x4xf32, "cpu">
// CHECK-NEXT: memref.copy %[[V0]], %[[ALLOC0]] : memref<2x4xf32, "device0"> to memref<2x4xf32, "cpu">
// CHECK-NEXT: %[[V1:.+]] = call @device0(%[[ALLOC]]) : (memref<?x4xf32, "device0">) -> memref<2x4xf32, "device0">
// CHECK-NEXT: %[[ALLOC1:.+]] = memref.alloc() : memref<2x4xf32, "cpu">
// CHECK-NEXT: memref.copy %[[V1]], %[[ALLOC1]] : memref<2x4xf32, "device0"> to memref<2x4xf32, "cpu">
// CHECK-NEXT: return %[[ALLOC0]], %[[ALLOC1]] : memref<2x4xf32, "cpu">, memref<2x4xf32, "cpu">
