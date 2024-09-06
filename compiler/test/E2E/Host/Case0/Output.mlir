// RUN: byteir-opt %s  | FileCheck %s

// CHECK-LABEL: func.func @main

module attributes {byre.container_module} {
  func.func @main(%arg0: memref<1xi64, "cpu"> {byre.argname = "Input0", byre.argtype = 1 : i32}, %arg1: memref<1xi64, "cpu"> {byre.argname = "Input1", byre.argtype = 1 : i32}, %arg2: memref<1xi64, "cpu"> {byre.argname = "Input2", byre.argtype = 1 : i32}, %arg3: memref<1x128xi32, "cpu"> {byre.argname = "Input3", byre.argtype = 1 : i32}, %arg4: memref<1x128xi32, "cpu"> {byre.argname = "Output0", byre.argtype = 2 : i32}, %arg5: memref<1x128xi32, "cpu"> {byre.argname = "Output1", byre.argtype = 2 : i32}) attributes {byre.entry_point} {
    byre.compute @LLVMJITOp(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5) {kernel_name = "Unknown0", llvm_file_name = "host_kernels.ll", memory_effects = [1 : i32, 1 : i32, 1 : i32, 1 : i32, 2 : i32, 2 : i32]} : memref<1xi64, "cpu">, memref<1xi64, "cpu">, memref<1xi64, "cpu">, memref<1x128xi32, "cpu">, memref<1x128xi32, "cpu">, memref<1x128xi32, "cpu">
    return
  }
}