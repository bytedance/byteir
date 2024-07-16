// RUN: byteir-opt %s -nvvm-codegen | FileCheck %s

// CHECK-LABEL: llvm.func @Unknown0

module attributes {byre.container_module, gpu.container_module} {
  gpu.module @unified {
    gpu.func @Unknown0(%arg0: memref<128x2x100xf32>, %arg1: memref<128x2x100xf32>, %arg2: memref<128x2x100xf32>) kernel {
      %c25600 = arith.constant 25600 : index
      %c2 = arith.constant 2 : index
      %c100 = arith.constant 100 : index
      %block_id_x = gpu.block_id  x
      %block_dim_x = gpu.block_dim  x
      %thread_id_x = gpu.thread_id  x
      %0 = arith.muli %block_dim_x, %block_id_x : index
      %1 = arith.addi %thread_id_x, %0 : index
      %grid_dim_x = gpu.grid_dim  x
      %2 = arith.muli %block_dim_x, %grid_dim_x : index
      scf.for %arg3 = %1 to %c25600 step %2 {
        %3 = arith.remsi %arg3, %c100 : index
        %4 = arith.divsi %arg3, %c100 : index
        %5 = arith.remsi %4, %c2 : index
        %6 = arith.divsi %4, %c2 : index
        %7 = memref.load %arg0[%6, %5, %3] : memref<128x2x100xf32>
        %8 = memref.load %arg1[%6, %5, %3] : memref<128x2x100xf32>
        %9 = arith.addf %7, %8 : f32
        memref.store %9, %arg2[%6, %5, %3] : memref<128x2x100xf32>
      }
      gpu.return
    }
  }
  func.func @main(%arg0: memref<512x200xf32, "cuda"> {byre.argname = "Input0", byre.argtype = 1 : i32}, %arg1: memref<512x2x100xf32, "cuda"> {byre.argname = "Input1", byre.argtype = 1 : i32}, %arg2: memref<128x2x100xf32, "cuda"> {byre.argname = "Output0", byre.argtype = 2 : i32}) attributes {byre.entry_point} {
    %0 = "byre.alias"(%arg0) <{offset = 0 : i64}> : (memref<512x200xf32, "cuda">) -> memref<128x2x100xf32, "cuda">
    %1 = "byre.alias"(%arg0) <{offset = 2000 : i64}> : (memref<512x200xf32, "cuda">) -> memref<128x2x100xf32, "cuda">
    byre.compute @PTXOp(%0, %1, %arg2) {BlockSize.x = 256 : i32, GridSize.x = 25 : i32, arg_ranks = [3 : i32, 3 : i32, 3 : i32], kernel_name = "Unknown0", memory_effects = [1 : i32, 1 : i32, 2 : i32]} : memref<128x2x100xf32, "cuda">, memref<128x2x100xf32, "cuda">, memref<128x2x100xf32, "cuda">
    return
  }
}