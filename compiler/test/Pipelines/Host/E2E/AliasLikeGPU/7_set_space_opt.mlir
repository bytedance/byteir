// RUN: byteir-opt %s -remove-func-body="anchor-attr=__byteir_elementwise_fusion__" -set-op-space="entry-func=main space=cuda" -set-arg-space="entry-func=main all-space=cuda" | FileCheck %s

// CHECK-LABEL: gpu.func @Unknown0

module attributes {gpu.container_module} {
  gpu.module @unified {
    gpu.func @Unknown0(%arg0: memref<128x2x100xf32>, %arg1: memref<128x2x100xf32>, %arg2: memref<128x2x100xf32>) kernel {
      %c25600 = arith.constant 25600 : index
      %c2 = arith.constant 2 : index
      %c100 = arith.constant 100 : index
      %0 = gpu.block_id  x
      %1 = gpu.block_dim  x
      %2 = gpu.thread_id  x
      %3 = arith.muli %1, %0 : index
      %4 = arith.addi %2, %3 : index
      %5 = gpu.grid_dim  x
      %6 = arith.muli %1, %5 : index
      scf.for %arg3 = %4 to %c25600 step %6 {
        %7 = arith.remsi %arg3, %c100 : index
        %8 = arith.divsi %arg3, %c100 : index
        %9 = arith.remsi %8, %c2 : index
        %10 = arith.divsi %8, %c2 : index
        %11 = memref.load %arg0[%10, %9, %7] : memref<128x2x100xf32>
        %12 = memref.load %arg1[%10, %9, %7] : memref<128x2x100xf32>
        %13 = arith.addf %11, %12 : f32
        memref.store %13, %arg2[%10, %9, %7] : memref<128x2x100xf32>
      }
      gpu.return
    }
  }
  func.func private @Unknown0(%arg0: memref<128x2x100xf32>, %arg1: memref<128x2x100xf32>) -> memref<128x2x100xf32> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 25 : i32, __byre__arg_ranks = [3 : i32, 3 : i32, 3 : i32], __byre__kernel_name = "Unknown0", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name} {
    %c25 = arith.constant 25 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %alloc = memref.alloc() : memref<128x2x100xf32>
    gpu.launch_func  @unified::@Unknown0 blocks in (%c25, %c1, %c1) threads in (%c256, %c1, %c1)  args(%arg0 : memref<128x2x100xf32>, %arg1 : memref<128x2x100xf32>, %alloc : memref<128x2x100xf32>)
    return %alloc : memref<128x2x100xf32>
  }
  func.func @main(%arg0: memref<512x200xf32>, %arg1: memref<512x2x100xf32>) -> memref<128x2x100xf32> attributes {__placeholder__byre.entry_point} {
    %subview = memref.subview %arg0[0, 0] [128, 200] [1, 1] : memref<512x200xf32> to memref<128x200xf32, strided<[200, 1]>>
    %subview_0 = memref.subview %arg0[10, 0] [128, 200] [1, 1] : memref<512x200xf32> to memref<128x200xf32, strided<[200, 1], offset: 2000>>
    %expand_shape = memref.expand_shape %subview_0 [[0], [1, 2]] : memref<128x200xf32, strided<[200, 1], offset: 2000>> into memref<128x2x100xf32, strided<[200, 100, 1], offset: 2000>>
    %expand_shape_1 = memref.expand_shape %subview [[0], [1, 2]] : memref<128x200xf32, strided<[200, 1]>> into memref<128x2x100xf32, strided<[200, 100, 1]>>
    %cast = memref.cast %expand_shape_1 : memref<128x2x100xf32, strided<[200, 100, 1]>> to memref<128x2x100xf32>
    %reinterpret_cast = memref.reinterpret_cast %expand_shape to offset: [0], sizes: [128, 2, 100], strides: [200, 100, 1] : memref<128x2x100xf32, strided<[200, 100, 1], offset: 2000>> to memref<128x2x100xf32>
    %0 = call @Unknown0(%cast, %reinterpret_cast) : (memref<128x2x100xf32>, memref<128x2x100xf32>) -> memref<128x2x100xf32>
    return %0 : memref<128x2x100xf32>
  }
}