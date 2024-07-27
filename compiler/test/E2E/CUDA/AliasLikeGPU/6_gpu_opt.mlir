// RUN: byteir-opt %s -gpu-opt | FileCheck %s

// CHECK-LABEL: gpu.func @Unknown0

module {
  func.func private @Unknown0(%arg0: memref<128x200xf32>, %arg1: memref<128x200xf32>) -> memref<128x2x100xf32> attributes {__byteir_elementwise_fusion__} {
    %c100 = arith.constant 100 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c25600 = arith.constant 25600 : index
    %alloc = memref.alloc() : memref<128x2x100xf32>
    scf.for %arg2 = %c0 to %c25600 step %c1 {
      %0 = arith.remsi %arg2, %c100 : index
      %1 = arith.divsi %arg2, %c100 : index
      %2 = arith.remsi %1, %c2 : index
      %3 = arith.divsi %1, %c2 : index
      %4 = arith.muli %2, %c100 : index
      %5 = arith.addi %4, %0 : index
      %6 = memref.load %arg0[%3, %5] : memref<128x200xf32>
      %7 = memref.load %arg1[%3, %5] : memref<128x200xf32>
      %8 = arith.addf %6, %7 : f32
      memref.store %8, %alloc[%3, %2, %0] : memref<128x2x100xf32>
    }
    return %alloc : memref<128x2x100xf32>
  }
  func.func @main(%arg0: memref<512x200xf32>, %arg1: memref<512x2x100xf32>) -> memref<128x2x100xf32> attributes {__placeholder__byre.entry_point} {
    %subview = memref.subview %arg0[0, 0] [128, 200] [1, 1] : memref<512x200xf32> to memref<128x200xf32, strided<[200, 1]>>
    %subview_0 = memref.subview %arg0[10, 0] [128, 200] [1, 1] : memref<512x200xf32> to memref<128x200xf32, strided<[200, 1], offset: 2000>>
    %cast = memref.cast %subview : memref<128x200xf32, strided<[200, 1]>> to memref<128x200xf32>
    %reinterpret_cast = memref.reinterpret_cast %subview_0 to offset: [0], sizes: [128, 200], strides: [200, 1] : memref<128x200xf32, strided<[200, 1], offset: 2000>> to memref<128x200xf32>
    %0 = call @Unknown0(%cast, %reinterpret_cast) : (memref<128x200xf32>, memref<128x200xf32>) -> memref<128x2x100xf32>
    return %0 : memref<128x2x100xf32>
  }
}