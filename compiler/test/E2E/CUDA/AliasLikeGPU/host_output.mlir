// RUN: byteir-opt %s  | FileCheck %s

// CHECK-LABEL: func.func @main

module attributes {byre.container_module, gpu.container_module} {
  func.func @main(%arg0: memref<512x200xf32, "cuda"> {byre.argname = "Input0", byre.argtype = 1 : i32}, %arg1: memref<512x200xf32, "cuda"> {byre.argname = "Input1", byre.argtype = 1 : i32}, %arg2: memref<256x256xf32, "cuda"> {byre.argname = "Output0", byre.argtype = 2 : i32}, %arg3: memref<512x200xf32, "cuda"> {byre.argname = "Output1", byre.argtype = 2 : i32}) attributes {byre.entry_point, device_file_name = "your_file"} {
    %alloc = memref.alloc() : memref<102400xi8, "cuda">
    %0 = "byre.alias"(%arg0) <{offset = 0 : i64}> : (memref<512x200xf32, "cuda">) -> memref<256x100xf32, "cuda">
    %1 = "byre.alias"(%alloc) <{offset = 0 : i64}> : (memref<102400xi8, "cuda">) -> memref<100x256xf32, "cuda">
    %2 = "byre.alias"(%arg1) <{offset = 2000 : i64}> : (memref<512x200xf32, "cuda">) -> memref<100x256xf32, "cuda">
    byre.copy(%2, %1) {callee = "cuda2cuda"} : memref<100x256xf32, "cuda">, memref<100x256xf32, "cuda">
    byre.compute @MatmulOp_f32f32_f32(%0, %1, %arg2) {device = "cuda", lhs_contracting_dimension = 1 : i64, memory_effects = [1 : i32, 1 : i32, 2 : i32], rhs_contracting_dimension = 0 : i64} : memref<256x100xf32, "cuda">, memref<100x256xf32, "cuda">, memref<256x256xf32, "cuda">
    byre.compute @PTXOp(%arg0, %arg1, %arg3) {BlockSize.x = 256 : i32, GridSize.x = 100 : i32, arg_ranks = [2 : i32, 2 : i32, 2 : i32], kernel_name = "Unknown0", memory_effects = [1 : i32, 1 : i32, 2 : i32]} : memref<512x200xf32, "cuda">, memref<512x200xf32, "cuda">, memref<512x200xf32, "cuda">
    return
  }
}