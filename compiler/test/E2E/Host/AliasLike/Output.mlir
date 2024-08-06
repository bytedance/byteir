// RUN: byteir-opt %s  | FileCheck %s

// CHECK-LABEL: func.func @main

module attributes {byre.container_module} {
  func.func @main(%arg0: memref<512x200xf32, "cpu"> {byre.argname = "Input0", byre.argtype = 1 : i32}, %arg1: memref<512x200xf32, "cpu"> {byre.argname = "Input1", byre.argtype = 1 : i32}, %arg2: memref<128x2x100xf32, "cpu"> {byre.argname = "Output0", byre.argtype = 2 : i32}, %arg3: memref<128x2x100xf32, "cpu"> {byre.argname = "Output1", byre.argtype = 2 : i32}, %arg4: memref<1x100xf32, "cpu"> {byre.argname = "Output2", byre.argtype = 2 : i32}, %arg5: memref<1x100xf32, "cpu"> {byre.argname = "Output3", byre.argtype = 2 : i32}, %arg6: memref<512x200xf32, "cpu"> {byre.argname = "Output4", byre.argtype = 2 : i32}) attributes {byre.entry_point, device_file_name = "your_file"} {
    byre.compute @LLVMJITOp(%arg0, %arg1, %arg6) {kernel_name = "Unknown0", llvm_file_name = "host_kernels.ll", memory_effects = [1 : i32, 1 : i32, 2 : i32]} : memref<512x200xf32, "cpu">, memref<512x200xf32, "cpu">, memref<512x200xf32, "cpu">
    %0 = "byre.alias"(%arg1) <{offset = 2000 : i64}> : (memref<512x200xf32, "cpu">) -> memref<128x2x100xf32, "cpu">
    byre.copy(%0, %arg3) {callee = "cpu2cpu"} : memref<128x2x100xf32, "cpu">, memref<128x2x100xf32, "cpu">
    %1 = "byre.alias"(%arg0) <{offset = 0 : i64}> : (memref<512x200xf32, "cpu">) -> memref<1x100xf32, "cpu">
    byre.copy(%1, %arg4) {callee = "cpu2cpu"} : memref<1x100xf32, "cpu">, memref<1x100xf32, "cpu">
    %2 = "byre.alias"(%arg1) <{offset = 2100 : i64}> : (memref<512x200xf32, "cpu">) -> memref<1x100xf32, "cpu">
    byre.copy(%2, %arg5) {callee = "cpu2cpu"} : memref<1x100xf32, "cpu">, memref<1x100xf32, "cpu">
    %3 = "byre.alias"(%arg0) <{offset = 0 : i64}> : (memref<512x200xf32, "cpu">) -> memref<128x2x100xf32, "cpu">
    byre.copy(%3, %arg2) {callee = "cpu2cpu"} : memref<128x2x100xf32, "cpu">, memref<128x2x100xf32, "cpu">
    return
  }
}