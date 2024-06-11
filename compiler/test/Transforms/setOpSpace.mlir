// RUN: byteir-opt %s -set-op-space="entry-func=main space=cpu" --allow-unregistered-dialect| FileCheck %s

// CHECK-LABEL: func.func @main
func.func @main(%arg0 : memref<2x4xf32>, %arg1 : memref<2x4xf32>, %arg2 : memref<2x4xf32>) -> (memref<2x4xf32>, memref<2x4xf32>) {
  %0 = memref.alloc() : memref<2x4xf32>
  "lmhlo.add"(%arg0, %arg1, %arg2) : (memref<2x4xf32>, memref<2x4xf32>, memref<2x4xf32>) -> ()
// CHECK: lmhlo.add
// CHECK-SAME: device = "cpu"
  "lmhlo.add"(%arg0, %arg1, %0) : (memref<2x4xf32>, memref<2x4xf32>, memref<2x4xf32>) -> ()
// CHECK-NEXT: lmhlo.add
// CHECK-SAME: device = "cpu"
  return %0, %0: memref<2x4xf32>, memref<2x4xf32> 
}
