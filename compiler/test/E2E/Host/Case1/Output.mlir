// RUN: byteir-opt %s  | FileCheck %s

// CHECK-LABEL: func.func @main

module attributes {byre.container_module} {
  func.func @main(%arg0: memref<1x100x27x48x3xf32, "cpu"> {byre.argname = "Input0", byre.argtype = 1 : i32}, %arg1: memref<51200xi32, "cpu"> {byre.argname = "Output0", byre.argtype = 2 : i32}) attributes {byre.entry_point, device_file_name = "your_file"} {
    %alloc = memref.alloc() : memref<3110400xi8, "cpu">
    %0 = "byre.alias"(%alloc) <{offset = 0 : i64}> {device = "cpu"} : (memref<3110400xi8, "cpu">) -> memref<1x100x27x48x3xi32, "cpu">
    byre.compute @LLVMJITOp(%arg0, %0) {device = "cpu", kernel_name = "Unknown0", llvm_file_name = "host_kernels.ll", memory_effects = [1 : i32, 2 : i32]} : memref<1x100x27x48x3xf32, "cpu">, memref<1x100x27x48x3xi32, "cpu">
    %1 = "byre.alias"(%alloc) <{offset = 0 : i64}> {device = "cpu"} : (memref<3110400xi8, "cpu">) -> memref<100x1296x3xi32, "cpu">
    %2 = "byre.alias"(%alloc) <{offset = 1555200 : i64}> {device = "cpu"} : (memref<3110400xi8, "cpu">) -> memref<100x1296x1xi32, "cpu">
    byre.compute @LLVMJITOp(%1, %2) {device = "cpu", kernel_name = "Unknown1", llvm_file_name = "host_kernels.ll", memory_effects = [1 : i32, 2 : i32]} : memref<100x1296x3xi32, "cpu">, memref<100x1296x1xi32, "cpu">
    %3 = "byre.alias"(%alloc) <{offset = 1555200 : i64}> {device = "cpu"} : (memref<3110400xi8, "cpu">) -> memref<100x1296xi32, "cpu">
    %4 = "byre.alias"(%alloc) <{offset = 2073600 : i64}> {device = "cpu"} : (memref<3110400xi8, "cpu">) -> memref<100x1296x1xi32, "cpu">
    byre.compute @LLVMJITOp(%1, %4) {device = "cpu", kernel_name = "Unknown2", llvm_file_name = "host_kernels.ll", memory_effects = [1 : i32, 2 : i32]} : memref<100x1296x3xi32, "cpu">, memref<100x1296x1xi32, "cpu">
    %5 = "byre.alias"(%alloc) <{offset = 2073600 : i64}> {device = "cpu"} : (memref<3110400xi8, "cpu">) -> memref<100x1296xi32, "cpu">
    %6 = "byre.alias"(%alloc) <{offset = 2592000 : i64}> {device = "cpu"} : (memref<3110400xi8, "cpu">) -> memref<100x1296x1xi32, "cpu">
    byre.compute @LLVMJITOp(%1, %6) {device = "cpu", kernel_name = "Unknown3", llvm_file_name = "host_kernels.ll", memory_effects = [1 : i32, 2 : i32]} : memref<100x1296x3xi32, "cpu">, memref<100x1296x1xi32, "cpu">
    %7 = "byre.alias"(%alloc) <{offset = 2592000 : i64}> {device = "cpu"} : (memref<3110400xi8, "cpu">) -> memref<100x1296xi32, "cpu">
    byre.compute @LLVMJITOp(%3, %5, %7, %arg1) {device = "cpu", kernel_name = "Unknown4", llvm_file_name = "host_kernels.ll", memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<100x1296xi32, "cpu">, memref<100x1296xi32, "cpu">, memref<100x1296xi32, "cpu">, memref<51200xi32, "cpu">
    return
  }
}