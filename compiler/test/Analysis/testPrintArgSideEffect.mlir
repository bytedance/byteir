// RUN: byteir-opt -test-print-arg-side-effect -split-input-file %s | FileCheck %s

func.func @lmhlo(%arg0: memref<2xf32>, %arg1: memref<2xf32>) {
  %0 = memref.alloc() : memref<2xf32>
  %1 = memref.alloc() : memref<2xf32>
  "lmhlo.abs"(%arg0, %0) : (memref<2xf32>, memref<2xf32>) -> ()
  "lmhlo.add"(%0, %arg1, %1) : (memref<2xf32>, memref<2xf32>, memref<2xf32>) -> ()
  return
}
// CHECK-LABEL: ============= registry of arg side effect =============

// CHECK-LABEL: ============= Test Module =============
// CHECK-NEXT: Testing memref.alloc:
// CHECK-NEXT: Testing memref.alloc:
// CHECK-NEXT: Testing lmhlo.abs:
// CHECK-NEXT: arg 0 ArgSideEffectType: kInput
// CHECK-NEXT: arg 1 ArgSideEffectType: kOutput
// CHECK-NEXT: Testing lmhlo.add:
// CHECK-NEXT: arg 0 ArgSideEffectType: kInput
// CHECK-NEXT: arg 1 ArgSideEffectType: kInput
// CHECK-NEXT: arg 2 ArgSideEffectType: kOutput
