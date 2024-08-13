// RUN: byteir-opt %s -nvvm-codegen | FileCheck %s

// CHECK-LABEL: llvm.func @Unknown0

module attributes {byre.container_module, gpu.container_module} {
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
  func.func @main(%arg0: memref<512x200xf32, "cuda"> {byre.argname = "Input0", byre.argtype = 1 : i32}, %arg1: memref<512x200xf32, "cuda"> {byre.argname = "Input1", byre.argtype = 1 : i32}, %arg2: memref<256x256xf32, "cuda"> {byre.argname = "Output0", byre.argtype = 2 : i32}, %arg3: memref<512x200xf32, "cuda"> {byre.argname = "Output1", byre.argtype = 2 : i32}) attributes {byre.entry_point} {
    %0 = "byre.alias"(%arg0) <{offset = 0 : i64}> : (memref<512x200xf32, "cuda">) -> memref<256x100xf32, "cuda">
    %1 = "byre.alias"(%arg1) <{offset = 2000 : i64}> : (memref<512x200xf32, "cuda">) -> memref<100x256xf32, "cuda">
    byre.compute @MatmulOp_f32f32_f32(%0, %1, %arg2) {device = "cuda", lhs_contracting_dimension = 1 : i64, memory_effects = [1 : i32, 1 : i32, 2 : i32], rhs_contracting_dimension = 0 : i64} : memref<256x100xf32, "cuda">, memref<100x256xf32, "cuda">, memref<256x256xf32, "cuda">
    byre.compute @PTXOp(%arg0, %arg1, %arg3) {BlockSize.x = 256 : i32, GridSize.x = 100 : i32, arg_ranks = [2 : i32, 2 : i32, 2 : i32], kernel_name = "Unknown0", memory_effects = [1 : i32, 1 : i32, 2 : i32]} : memref<512x200xf32, "cuda">, memref<512x200xf32, "cuda">, memref<512x200xf32, "cuda">
    return
  }
}