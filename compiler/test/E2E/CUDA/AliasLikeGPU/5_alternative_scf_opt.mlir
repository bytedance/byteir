// RUN: byteir-opt %s -scf-opt | FileCheck %s

// CHECK-LABEL: func.func @main

#map = affine_map<() -> ()>
module {
  func.func private @Unknown0(%arg0: memref<512x200xf32>, %arg1: memref<512x200xf32>) -> memref<512x200xf32> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %c512 = arith.constant 512 : index
    %c1 = arith.constant 1 : index
    %c200 = arith.constant 200 : index
    %alloc = memref.alloc() : memref<512x200xf32>
    scf.for %arg2 = %c0 to %c512 step %c1 {
      scf.for %arg3 = %c0 to %c200 step %c1 {
        %subview = memref.subview %arg0[%arg2, %arg3] [1, 1] [1, 1] : memref<512x200xf32> to memref<f32, strided<[], offset: ?>>
        %subview_0 = memref.subview %alloc[%arg2, %arg3] [1, 1] [1, 1] : memref<512x200xf32> to memref<f32, strided<[], offset: ?>>
        %subview_1 = memref.subview %arg1[%arg2, %arg3] [1, 1] [1, 1] : memref<512x200xf32> to memref<f32, strided<[], offset: ?>>
        linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []} ins(%subview, %subview_1 : memref<f32, strided<[], offset: ?>>, memref<f32, strided<[], offset: ?>>) outs(%subview_0 : memref<f32, strided<[], offset: ?>>) {
        ^bb0(%in: f32, %in_2: f32, %out: f32):
          %0 = arith.addf %in, %in_2 : f32
          linalg.yield %0 : f32
        }
      }
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
    %alloc_3 = memref.alloc() : memref<100x256xf32>
    memref.copy %expand_shape_2, %alloc_3 : memref<100x256xf32, strided<[256, 1], offset: 2000>> to memref<100x256xf32>
    byre.compute @MatmulOp_f32f32_f32(%expand_shape, %alloc_3, %alloc) {lhs_contracting_dimension = 1 : i64, memory_effects = [1 : i32, 1 : i32, 2 : i32], rhs_contracting_dimension = 0 : i64} : memref<256x100xf32>, memref<100x256xf32>, memref<256x256xf32>
    %0 = call @Unknown0(%arg0, %arg1) : (memref<512x200xf32>, memref<512x200xf32>) -> memref<512x200xf32>
    return %alloc, %0 : memref<256x256xf32>, memref<512x200xf32>
  }
}