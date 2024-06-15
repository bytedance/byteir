// RUN: byteir-opt %s  | FileCheck %s

// CHECK-LABEL: func.func @main

module attributes {byre.container_module} {
  func.func @main(%arg0: memref<1x97xf32, "cpu"> {byre.argname = "Output0", byre.argtype = 2 : i32}) attributes {byre.entry_point, device_file_name = "your_file"} {
    %alloc = memref.alloc() : memref<256xi8, "cpu">
    %0 = "byre.alias"(%alloc) <{offset = 0 : i64}> {device = "cpu"} : (memref<256xi8, "cpu">) -> memref<i64, "cpu">
    byre.compute @GetSeed(%0) {device = "cpu", memory_effects = [2 : i32]} : memref<i64, "cpu">
    %1 = "byre.alias"(%alloc) <{offset = 128 : i64}> {device = "cpu"} : (memref<256xi8, "cpu">) -> memref<i64, "cpu">
    byre.compute @NextOffset(%1) {device = "cpu", memory_effects = [2 : i32]} : memref<i64, "cpu">
    byre.compute @LLVMJITOp(%0, %1, %arg0) {device = "cpu", kernel_name = "Unknown0", llvm_file_name = "host_kernels.ll", memory_effects = [1 : i32, 1 : i32, 2 : i32]} : memref<i64, "cpu">, memref<i64, "cpu">, memref<1x97xf32, "cpu">
    return
  }
}