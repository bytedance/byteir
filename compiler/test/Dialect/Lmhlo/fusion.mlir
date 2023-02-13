// RUN: byteir-opt %s | FileCheck %s

func.func @lmhlo_add(%arg0: memref<4xf32>, %arg1: memref<4xf32>) -> memref<4xf32> {
  %0 = memref.alloc() : memref<4xf32>
  %1 = memref.alloc() : memref<4xf32>
  "lmhlo.fusion"() ( {
    "lmhlo.add"(%arg0, %arg1, %0) : (memref<4xf32>, memref<4xf32>, memref<4xf32>) -> ()
    "lmhlo.add"(%0, %arg1, %1) : (memref<4xf32>, memref<4xf32>, memref<4xf32>) -> ()
    "lmhlo.terminator"() : () -> ()
  }) : () -> ()
  return %1 : memref<4xf32>
}
// CHECK-LABEL: func.func @lmhlo_add
