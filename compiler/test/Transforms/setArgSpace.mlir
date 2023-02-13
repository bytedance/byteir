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