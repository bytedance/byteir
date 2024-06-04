// RUN: byteir-opt %s -nvvm-codegen | FileCheck %s

// CHECK-LABEL: llvm.func @Unknown0

module attributes {byre.container_module, gpu.container_module} {
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
  func.func @main(%arg0: memref<512x200xf32, "cuda"> {byre.argname = "Input0", byre.argtype = 1 : i32}, %arg1: memref<512x2x100xf32, "cuda"> {byre.argname = "Input1", byre.argtype = 1 : i32}, %arg2: memref<128x2x100xf32, "cuda"> {byre.argname = "Output0", byre.argtype = 2 : i32}) attributes {byre.entry_point} {
    %0 = "byre.alias"(%arg0) <{offset = 0 : i64}> : (memref<512x200xf32, "cuda">) -> memref<128x2x100xf32, "cuda">
    %1 = "byre.alias"(%arg0) <{offset = 2000 : i64}> : (memref<512x200xf32, "cuda">) -> memref<128x2x100xf32, "cuda">
    byre.compute @PTXOp(%0, %1, %arg2) {BlockSize.x = 256 : i32, GridSize.x = 25 : i32, arg_ranks = [3 : i32, 3 : i32, 3 : i32], kernel_name = "Unknown0", memory_effects = [1 : i32, 1 : i32, 2 : i32]} : memref<128x2x100xf32, "cuda">, memref<128x2x100xf32, "cuda">, memref<128x2x100xf32, "cuda">
    return
  }
}