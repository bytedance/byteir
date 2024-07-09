// RUN: byteir-opt %s --host-opt -set-op-space="entry-func=main space=cpu" -set-arg-space="entry-func=main all-space=cpu" --byre-opt | FileCheck %s

// CHECK-LABEL: func.func @Unknown

module {
  func.func private @Unknown0(%arg0: memref<128x2x100xf32>, %arg1: memref<128x2x100xf32>) -> memref<128x2x100xf32> attributes {__byteir_hlo_aggressive_fusion__} {
    %c0 = arith.constant 0 : index
    %c25600 = arith.constant 25600 : index
    %c1 = arith.constant 1 : index
    %alloc = memref.alloc() : memref<128x2x100xf32>
    %collapse_shape = memref.collapse_shape %arg0 [[0, 1, 2]] : memref<128x2x100xf32> into memref<25600xf32>
    %collapse_shape_0 = memref.collapse_shape %arg1 [[0, 1, 2]] : memref<128x2x100xf32> into memref<25600xf32>
    %collapse_shape_1 = memref.collapse_shape %alloc [[0, 1, 2]] : memref<128x2x100xf32> into memref<25600xf32>
    scf.for %arg2 = %c0 to %c25600 step %c1 {
      %0 = memref.load %collapse_shape[%arg2] : memref<25600xf32>
      %1 = memref.load %collapse_shape_0[%arg2] : memref<25600xf32>
      %2 = arith.addf %0, %1 : f32
      memref.store %2, %collapse_shape_1[%arg2] : memref<25600xf32>
    }
    return %alloc : memref<128x2x100xf32>
  }
  func.func @main(%arg0: memref<512x200xf32>, %arg1: memref<512x2x100xf32>) -> memref<128x2x100xf32> attributes {__placeholder__byre.entry_point} {
    %subview = memref.subview %arg0[0, 0] [128, 200] [1, 1] : memref<512x200xf32> to memref<128x200xf32, strided<[200, 1]>>
    %subview_0 = memref.subview %arg0[10, 0] [128, 200] [1, 1] : memref<512x200xf32> to memref<128x200xf32, strided<[200, 1], offset: 2000>>
    %expand_shape = memref.expand_shape %subview_0 [[0], [1, 2]] output_shape [128, 2, 100] : memref<128x200xf32, strided<[200, 1], offset: 2000>> into memref<128x2x100xf32, strided<[200, 100, 1], offset: 2000>>
    %expand_shape_1 = memref.expand_shape %subview [[0], [1, 2]] output_shape [128, 2, 100] : memref<128x200xf32, strided<[200, 1]>> into memref<128x2x100xf32, strided<[200, 100, 1]>>
    %cast = memref.cast %expand_shape_1 : memref<128x2x100xf32, strided<[200, 100, 1]>> to memref<128x2x100xf32>
    %reinterpret_cast = memref.reinterpret_cast %expand_shape to offset: [0], sizes: [128, 2, 100], strides: [200, 100, 1] : memref<128x2x100xf32, strided<[200, 100, 1], offset: 2000>> to memref<128x2x100xf32>
    %0 = call @Unknown0(%cast, %reinterpret_cast) : (memref<128x2x100xf32>, memref<128x2x100xf32>) -> memref<128x2x100xf32>
    return %0 : memref<128x2x100xf32>
  }
}