// RUN: byteir-opt %s -byre-opt="append-arg-types entry-func=main" | FileCheck %s

// CHECK-LABEL: func.func @main

module attributes {gpu.container_module} {
  gpu.module @unified {
    gpu.func @Unknown0(%arg0: memref<512x200xf32>, %arg1: memref<512x200xf32>, %arg2: memref<512x200xf32>) kernel {
      %c102400 = arith.constant 102400 : index
      %c200 = arith.constant 200 : index
      %block_id_x = gpu.block_id  x
      %block_dim_x = gpu.block_dim  x
      %thread_id_x = gpu.thread_id  x
      %0 = arith.muli %block_dim_x, %block_id_x : index
      %1 = arith.addi %thread_id_x, %0 : index
      %grid_dim_x = gpu.grid_dim  x
      %2 = arith.muli %block_dim_x, %grid_dim_x : index
      scf.for %arg3 = %1 to %c102400 step %2 {
        %3 = arith.remsi %arg3, %c200 : index
        %4 = arith.divsi %arg3, %c200 : index
        %5 = memref.load %arg0[%4, %3] : memref<512x200xf32>
        %6 = memref.load %arg1[%4, %3] : memref<512x200xf32>
        %7 = arith.addf %5, %6 : f32
        memref.store %7, %arg2[%4, %3] : memref<512x200xf32>
      }
      gpu.return
    }
  }
  func.func @main(%arg0: memref<512x200xf32, "cuda">, %arg1: memref<512x200xf32, "cuda">) -> (memref<256x256xf32, "cuda">, memref<512x200xf32, "cuda">) attributes {__placeholder__byre.entry_point} {
    %subview = memref.subview %arg0[0, 0] [128, 200] [1, 1] : memref<512x200xf32, "cuda"> to memref<128x200xf32, strided<[200, 1]>, "cuda">
    %subview_0 = memref.subview %arg1[10, 0] [128, 200] [1, 1] : memref<512x200xf32, "cuda"> to memref<128x200xf32, strided<[200, 1], offset: 2000>, "cuda">
    %collapse_shape = memref.collapse_shape %subview [[0, 1]] : memref<128x200xf32, strided<[200, 1]>, "cuda"> into memref<25600xf32, strided<[1]>, "cuda">
    %expand_shape = memref.expand_shape %collapse_shape [[0, 1]] output_shape [256, 100] : memref<25600xf32, strided<[1]>, "cuda"> into memref<256x100xf32, "cuda">
    %collapse_shape_1 = memref.collapse_shape %subview_0 [[0, 1]] : memref<128x200xf32, strided<[200, 1], offset: 2000>, "cuda"> into memref<25600xf32, strided<[1], offset: 2000>, "cuda">
    %expand_shape_2 = memref.expand_shape %collapse_shape_1 [[0, 1]] output_shape [100, 256] : memref<25600xf32, strided<[1], offset: 2000>, "cuda"> into memref<100x256xf32, strided<[256, 1], offset: 2000>, "cuda">
    %alloc = memref.alloc() : memref<256x256xf32, "cuda">
    %0 = "byre.alias"(%expand_shape_2) <{offset = 2000 : i64}> {device = "cuda"} : (memref<100x256xf32, strided<[256, 1], offset: 2000>, "cuda">) -> memref<100x256xf32, "cuda">
    byre.compute @MatmulOp_f32f32_f32(%expand_shape, %0, %alloc) {device = "cuda", lhs_contracting_dimension = 1 : i64, memory_effects = [1 : i32, 1 : i32, 2 : i32], rhs_contracting_dimension = 0 : i64} : memref<256x100xf32, "cuda">, memref<100x256xf32, "cuda">, memref<256x256xf32, "cuda">
    %alloc_3 = memref.alloc() : memref<512x200xf32, "cuda">
    byre.compute @PTXOp(%arg0, %arg1, %alloc_3) {BlockSize.x = 256 : i32, BlockSize.y = 1 : i32, BlockSize.z = 1 : i32, GridSize.x = 100 : i32, GridSize.y = 1 : i32, GridSize.z = 1 : i32, device = "cuda", device_file_name = "device_kernel.ptx", kernel_name = "Unknown0"} : memref<512x200xf32, "cuda">, memref<512x200xf32, "cuda">, memref<512x200xf32, "cuda">
    return %alloc, %alloc_3 : memref<256x256xf32, "cuda">, memref<512x200xf32, "cuda">
  }
}