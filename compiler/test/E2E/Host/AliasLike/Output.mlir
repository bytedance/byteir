// RUN: byteir-opt %s  | FileCheck %s

// CHECK-LABEL: func.func @main

module attributes {byre.container_module} {
  func.func @main(%arg0: memref<512x200xf32, "cpu"> {byre.argname = "Input0", byre.argtype = 1 : i32}, %arg1: memref<512x2x100xf32, "cpu"> {byre.argname = "Input1", byre.argtype = 1 : i32}, %arg2: memref<128x2x100xf32, "cpu"> {byre.argname = "Output0", byre.argtype = 2 : i32}) attributes {byre.entry_point, device_file_name = "your_file"} {
    %0 = "byre.alias"(%arg0) <{offset = 0 : i64}> {device = "cpu"} : (memref<512x200xf32, "cpu">) -> memref<128x2x100xf32, "cpu">
    %1 = "byre.alias"(%arg0) <{offset = 2000 : i64}> {device = "cpu"} : (memref<512x200xf32, "cpu">) -> memref<128x2x100xf32, "cpu">
    byre.compute @LLVMJITOp(%0, %1, %arg2) {device = "cpu", kernel_name = "Unknown0", llvm_file_name = "host_kernels.ll", memory_effects = [1 : i32, 1 : i32, 2 : i32]} : memref<128x2x100xf32, "cpu">, memref<128x2x100xf32, "cpu">, memref<128x2x100xf32, "cpu">
    return
  }
}