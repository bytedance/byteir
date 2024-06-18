// RUN: byteir-opt %s  | FileCheck %s

// CHECK-LABEL: func.func @main

module attributes {byre.container_module} {
  func.func @main(%arg0: memref<1x100x27x48x3xf32, "cpu"> {byre.argname = "Input0", byre.argtype = 1 : i32}, %arg1: memref<51200xi32, "cpu"> {byre.argname = "Output0", byre.argtype = 2 : i32}) attributes {byre.entry_point, device_file_name = "your_file"} {
    byre.compute @LLVMJITOp(%arg0, %arg1) {device = "cpu", kernel_name = "Unknown0", llvm_file_name = "host_kernels.ll", memory_effects = [1 : i32, 2 : i32]} : memref<1x100x27x48x3xf32, "cpu">, memref<51200xi32, "cpu">
    return
  }
}