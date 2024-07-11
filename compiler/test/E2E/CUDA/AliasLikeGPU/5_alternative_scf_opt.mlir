// RUN: byteir-opt %s -scf-opt | FileCheck %s

// CHECK-LABEL: func.func private @Unknown

#map = affine_map<() -> ()>
module {
  func.func private @Unknown0(%arg0: memref<128x2x100xf32>, %arg1: memref<128x2x100xf32>) -> memref<128x2x100xf32> attributes {__byteir_elementwise_fusion__} {
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c100 = arith.constant 100 : index
    %alloc = memref.alloc() : memref<128x2x100xf32>
    scf.for %arg2 = %c0 to %c128 step %c1 {
      scf.for %arg3 = %c0 to %c2 step %c1 {
        scf.for %arg4 = %c0 to %c100 step %c1 {
          %subview = memref.subview %arg0[%arg2, %arg3, %arg4] [1, 1, 1] [1, 1, 1] : memref<128x2x100xf32> to memref<f32, strided<[], offset: ?>>
          %subview_0 = memref.subview %alloc[%arg2, %arg3, %arg4] [1, 1, 1] [1, 1, 1] : memref<128x2x100xf32> to memref<f32, strided<[], offset: ?>>
          %subview_1 = memref.subview %arg1[%arg2, %arg3, %arg4] [1, 1, 1] [1, 1, 1] : memref<128x2x100xf32> to memref<f32, strided<[], offset: ?>>
          linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []} ins(%subview, %subview_1 : memref<f32, strided<[], offset: ?>>, memref<f32, strided<[], offset: ?>>) outs(%subview_0 : memref<f32, strided<[], offset: ?>>) {
          ^bb0(%in: f32, %in_2: f32, %out: f32):
            %0 = arith.addf %in, %in_2 : f32
            linalg.yield %0 : f32
          }
        }
      }
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