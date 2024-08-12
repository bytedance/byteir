// RUN: byteir-opt %s --host-opt -set-op-space="entry-func=main space=cpu" -set-arg-space="entry-func=main all-space=cpu" --byre-opt | FileCheck %s

// CHECK-LABEL: func.func @main

module {
  func.func private @Unknown0(%arg0: memref<512x200xf32>, %arg1: memref<512x200xf32>) -> memref<512x200xf32> attributes {__byteir_hlo_aggressive_fusion__} {
    %c0 = arith.constant 0 : index
    %c102400 = arith.constant 102400 : index
    %c1 = arith.constant 1 : index
    %alloc = memref.alloc() : memref<512x200xf32>
    %collapse_shape = memref.collapse_shape %arg0 [[0, 1]] : memref<512x200xf32> into memref<102400xf32>
    %collapse_shape_0 = memref.collapse_shape %arg1 [[0, 1]] : memref<512x200xf32> into memref<102400xf32>
    %collapse_shape_1 = memref.collapse_shape %alloc [[0, 1]] : memref<512x200xf32> into memref<102400xf32>
    scf.for %arg2 = %c0 to %c102400 step %c1 {
      %0 = memref.load %collapse_shape[%arg2] : memref<102400xf32>
      %1 = memref.load %collapse_shape_0[%arg2] : memref<102400xf32>
      %2 = arith.addf %0, %1 : f32
      memref.store %2, %collapse_shape_1[%arg2] : memref<102400xf32>
    }
    return %alloc : memref<512x200xf32>
  }
  func.func @main(%arg0: memref<512x200xf32>, %arg1: memref<512x200xf32>) -> (memref<128x2x100xf32>, memref<128x2x100xf32>, memref<1x100xf32>, memref<1x100xf32>, memref<512x200xf32>) attributes {__placeholder__byre.entry_point} {
    %subview = memref.subview %arg0[0, 0] [128, 200] [1, 1] : memref<512x200xf32> to memref<128x200xf32, strided<[200, 1]>>
    %subview_0 = memref.subview %arg1[10, 0] [128, 200] [1, 1] : memref<512x200xf32> to memref<128x200xf32, strided<[200, 1], offset: 2000>>
    %expand_shape = memref.expand_shape %subview [[0], [1, 2]] output_shape [128, 2, 100] : memref<128x200xf32, strided<[200, 1]>> into memref<128x2x100xf32, strided<[200, 100, 1]>>
    %expand_shape_1 = memref.expand_shape %subview_0 [[0], [1, 2]] output_shape [128, 2, 100] : memref<128x200xf32, strided<[200, 1], offset: 2000>> into memref<128x2x100xf32, strided<[200, 100, 1], offset: 2000>>
    %subview_2 = memref.subview %arg0[0, 0] [1, 100] [1, 1] : memref<512x200xf32> to memref<1x100xf32, strided<[200, 1]>>
    %subview_3 = memref.subview %arg1[10, 100] [1, 100] [1, 1] : memref<512x200xf32> to memref<1x100xf32, strided<[200, 1], offset: 2100>>
    %0 = call @Unknown0(%arg0, %arg1) : (memref<512x200xf32>, memref<512x200xf32>) -> memref<512x200xf32>
    %cast = memref.cast %expand_shape : memref<128x2x100xf32, strided<[200, 100, 1]>> to memref<128x2x100xf32>
    %alloc = memref.alloc() : memref<128x2x100xf32>
    memref.copy %expand_shape_1, %alloc : memref<128x2x100xf32, strided<[200, 100, 1], offset: 2000>> to memref<128x2x100xf32>
    %alloc_4 = memref.alloc() : memref<1x100xf32>
    memref.copy %subview_2, %alloc_4 : memref<1x100xf32, strided<[200, 1]>> to memref<1x100xf32>
    %alloc_5 = memref.alloc() : memref<1x100xf32>
    memref.copy %subview_3, %alloc_5 : memref<1x100xf32, strided<[200, 1], offset: 2100>> to memref<1x100xf32>
    return %cast, %alloc, %alloc_4, %alloc_5, %0 : memref<128x2x100xf32>, memref<128x2x100xf32>, memref<1x100xf32>, memref<1x100xf32>, memref<512x200xf32>
  }
}