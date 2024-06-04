// RUN: byteir-opt %s -byre-opt="append-arg-types entry-func=main" | FileCheck %s

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
  func.func private @Unknown0(memref<128x2x100xf32, "cuda">, memref<128x2x100xf32, "cuda">) -> memref<128x2x100xf32, "cuda"> attributes {__byre__BlockSize.x = 256 : i32, __byre__GridSize.x = 25 : i32, __byre__arg_ranks = [3 : i32, 3 : i32, 3 : i32], __byre__kernel_name = "Unknown0", __byteir_elementwise_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "PTXOp", byre_force_compute_name, device = "cuda"}
  func.func @main(%arg0: memref<512x200xf32, "cuda">, %arg1: memref<512x2x100xf32, "cuda">) -> memref<128x2x100xf32, "cuda"> attributes {__placeholder__byre.entry_point} {
    %subview = memref.subview %arg0[0, 0] [128, 200] [1, 1] : memref<512x200xf32, "cuda"> to memref<128x200xf32, strided<[200, 1]>, "cuda">
    %subview_0 = memref.subview %arg0[10, 0] [128, 200] [1, 1] : memref<512x200xf32, "cuda"> to memref<128x200xf32, strided<[200, 1], offset: 2000>, "cuda">
    %expand_shape = memref.expand_shape %subview_0 [[0], [1, 2]] : memref<128x200xf32, strided<[200, 1], offset: 2000>, "cuda"> into memref<128x2x100xf32, strided<[200, 100, 1], offset: 2000>, "cuda">
    %expand_shape_1 = memref.expand_shape %subview [[0], [1, 2]] : memref<128x200xf32, strided<[200, 1]>, "cuda"> into memref<128x2x100xf32, strided<[200, 100, 1]>, "cuda">
    %cast = memref.cast %expand_shape_1 : memref<128x2x100xf32, strided<[200, 100, 1]>, "cuda"> to memref<128x2x100xf32, "cuda">
    %reinterpret_cast = memref.reinterpret_cast %expand_shape to offset: [0], sizes: [128, 2, 100], strides: [200, 100, 1] : memref<128x2x100xf32, strided<[200, 100, 1], offset: 2000>, "cuda"> to memref<128x2x100xf32, "cuda">
    %0 = call @Unknown0(%cast, %reinterpret_cast) : (memref<128x2x100xf32, "cuda">, memref<128x2x100xf32, "cuda">) -> memref<128x2x100xf32, "cuda">
    return %0 : memref<128x2x100xf32, "cuda">
  }
}