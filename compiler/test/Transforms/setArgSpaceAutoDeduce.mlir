// RUN: byteir-opt %s -set-arg-space="entry-func=main auto-deduce=true" --allow-unregistered-dialect --split-input-file | FileCheck %s

func.func private @foo_a(%arg : memref<2x4xf32>) -> memref<2x4xf32> attributes {device="gpu"}
func.func private @foo_b(%arg : memref<2x4xf32>) -> memref<2x4xf32> attributes {device="cpu"} {
  %0 = "foo.bar"(%arg) {device="cpu"} : (memref<2x4xf32>) -> memref<2x4xf32>
  return %0 : memref<2x4xf32>
}

func.func @main(%arg0 : memref<2x4xf32>) -> (memref<2x4xf32>, memref<2x4xf32>, memref<2x4xf32>) {
  %0 = call @foo_a(%arg0) : (memref<2x4xf32>) -> memref<2x4xf32>
  %1 = call @foo_b(%0) : (memref<2x4xf32>) -> memref<2x4xf32>
  %2 = "foo.bar"(%1) {device="cpu"} : (memref<2x4xf32>) -> memref<2x4xf32>
  return %0, %1, %2: memref<2x4xf32>, memref<2x4xf32>, memref<2x4xf32>
}

// CHECK-LABEL: func.func @main
//   CHECK-SAME: (%[[ARG:.*]]: memref<2x4xf32, "gpu">) -> (memref<2x4xf32, "gpu">, memref<2x4xf32, "cpu">, memref<2x4xf32, "cpu">)
// CHECK-NEXT: %[[FOO_A:.*]] = call @foo_a(%[[ARG]])
//   CHECK-SAME: (memref<2x4xf32, "gpu">) -> memref<2x4xf32, "gpu">
// CHECK-NEXT: %[[ALLOC:.*]] = memref.alloc() 
//   CHECK-SAME: memref<2x4xf32, "cpu">
// CHECK-NEXT: memref.copy %[[FOO_A]], %[[ALLOC]]
//   CHECK-SAME: memref<2x4xf32, "gpu"> to memref<2x4xf32, "cpu">
// CHECK-NEXT: %[[FOO_B:.*]] = call @foo_b(%[[ALLOC]])
//   CHECK-SAME: (memref<2x4xf32, "cpu">) -> memref<2x4xf32, "cpu">
// CHECK-NEXT: %[[FOO_BAR:.*]] = "foo.bar"(%[[FOO_B]])
//   CHECK-SAME: (memref<2x4xf32, "cpu">) -> memref<2x4xf32, "cpu">
// CHECK-NEXT: return %[[FOO_A]], %[[FOO_B]], %[[FOO_BAR]]
//   CHECK-SAME: memref<2x4xf32, "gpu">, memref<2x4xf32, "cpu">, memref<2x4xf32, "cpu">
