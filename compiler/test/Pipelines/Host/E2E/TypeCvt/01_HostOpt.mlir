// RUN: byteir-opt %s --host-opt --byre-opt | FileCheck %s

// CHECK-LABEL: func.func @Unknown
//   CHECK: %[[COLLAPSE0:.*]] = memref.collapse_shape %{{.*}} {{\[}}[0, 1, 2, 3]] : memref<1x224x224x3xf32> into memref<150528xf32>
//   CHECK: %[[COLLAPSE1:.*]] = memref.collapse_shape %{{.*}} {{\[}}[0, 1, 2, 3]] : memref<1x224x224x3xf16> into memref<150528xf16>
//   CHECK-NEXT: scf.for %arg2 = %c0 to %c150528 step %c1 {
//   CHECK-NEXT:   %[[LOAD:.*]] = memref.load %[[COLLAPSE0]]
//   CHECK-NEXT:   %[[TRUNCF:.*]] = arith.truncf %[[LOAD]] : f32 to f16
//   CHECK-NEXT:   memref.store %[[TRUNCF]], %[[COLLAPSE1]]
//   CHECK-NEXT: }

module {
  func.func private @Unknown0(%arg0: memref<1x224x224x3xf32>) -> memref<1x224x224x3xf16> attributes {__byteir_hlo_aggressive_fusion__} {
    %c1 = arith.constant 1 : index
    %c150528 = arith.constant 150528 : index
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() : memref<1x224x224x3xf16>
    %collapse_shape = memref.collapse_shape %arg0 [[0, 1, 2, 3]] : memref<1x224x224x3xf32> into memref<150528xf32>
    %collapse_shape_0 = memref.collapse_shape %alloc [[0, 1, 2, 3]] : memref<1x224x224x3xf16> into memref<150528xf16>
    scf.for %arg1 = %c0 to %c150528 step %c1 {
      %0 = memref.load %collapse_shape[%arg1] : memref<150528xf32>
      %1 = arith.truncf %0 : f32 to f16
      memref.store %1, %collapse_shape_0[%arg1] : memref<150528xf16>
    }
    return %alloc : memref<1x224x224x3xf16>
  }
  func.func @main(%arg0: memref<1x224x224x3xf32>) -> memref<1x224x224x3xf16> {
    %0 = call @Unknown0(%arg0) : (memref<1x224x224x3xf32>) -> memref<1x224x224x3xf16>
    return %0 : memref<1x224x224x3xf16>
  }
}