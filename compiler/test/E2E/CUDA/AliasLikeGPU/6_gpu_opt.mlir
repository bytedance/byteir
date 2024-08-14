// RUN: byteir-opt %s -gpu-opt | FileCheck %s

// CHECK-LABEL: func.func @main

module {
  func.func private @Unknown0(%arg0: memref<512x200xf32>, %arg1: memref<512x200xf32>) -> memref<512x200xf32> attributes {__byteir_elementwise_fusion__} {
    %c200 = arith.constant 200 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c102400 = arith.constant 102400 : index
    %alloc = memref.alloc() : memref<512x200xf32>
    scf.for %arg2 = %c0 to %c102400 step %c1 {
      %0 = arith.remsi %arg2, %c200 : index
      %1 = arith.divsi %arg2, %c200 : index
      %2 = memref.load %arg0[%1, %0] : memref<512x200xf32>
      %3 = memref.load %arg1[%1, %0] : memref<512x200xf32>
      %4 = arith.addf %2, %3 : f32
      memref.store %4, %alloc[%1, %0] : memref<512x200xf32>
    }
    return %alloc : memref<512x200xf32>
  }
  func.func @main(%arg0: memref<512x200xf32>, %arg1: memref<512x200xf32>) -> (memref<256x256xf32>, memref<512x200xf32>) attributes {__placeholder__byre.entry_point} {
    %subview = memref.subview %arg0[0, 0] [128, 200] [1, 1] : memref<512x200xf32> to memref<128x200xf32, strided<[200, 1]>>
    %subview_0 = memref.subview %arg1[10, 0] [128, 200] [1, 1] : memref<512x200xf32> to memref<128x200xf32, strided<[200, 1], offset: 2000>>
    %collapse_shape = memref.collapse_shape %subview [[0, 1]] : memref<128x200xf32, strided<[200, 1]>> into memref<25600xf32, strided<[1]>>
    %expand_shape = memref.expand_shape %collapse_shape [[0, 1]] output_shape [256, 100] : memref<25600xf32, strided<[1]>> into memref<256x100xf32>
    %collapse_shape_1 = memref.collapse_shape %subview_0 [[0, 1]] : memref<128x200xf32, strided<[200, 1], offset: 2000>> into memref<25600xf32, strided<[1], offset: 2000>>
    %expand_shape_2 = memref.expand_shape %collapse_shape_1 [[0, 1]] output_shape [100, 256] : memref<25600xf32, strided<[1], offset: 2000>> into memref<100x256xf32, strided<[256, 1], offset: 2000>>
    %alloc = memref.alloc() : memref<256x256xf32>
    %0 = "byre.alias"(%expand_shape_2) <{offset = 2000 : i64}> : (memref<100x256xf32, strided<[256, 1], offset: 2000>>) -> memref<100x256xf32>
    byre.compute @MatmulOp_f32f32_f32(%expand_shape, %0, %alloc) {lhs_contracting_dimension = 1 : i64, memory_effects = [1 : i32, 1 : i32, 2 : i32], rhs_contracting_dimension = 0 : i64} : memref<256x100xf32>, memref<100x256xf32>, memref<256x256xf32>
    %1 = call @Unknown0(%arg0, %arg1) : (memref<512x200xf32>, memref<512x200xf32>) -> memref<512x200xf32>
    return %alloc, %1 : memref<256x256xf32>, memref<512x200xf32>
  }
}