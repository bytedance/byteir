// RUN: byteir-opt %s -byre-host="device-file-name=your_file target=cpu" | FileCheck %s

// CHECK-LABEL: func.func @main

module attributes {byre.container_module} {
  module attributes {byteir.llvm_module} {
    func.func @Unknown0(%arg0: memref<i64>, %arg1: memref<i64>, %arg2: memref<1x97xf32>) attributes {__byre__kernel_name = "Unknown0", __byre__llvm_file_name = "host_kernels.ll", __byteir_hlo_aggressive_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "LLVMJITOp", byre_force_compute_name, llvm.emit_c_interface} {
      %cst = arith.constant 0.000000e+00 : f32
      %cst_0 = arith.constant 2.32830644E-10 : f32
      %c12345_i32 = arith.constant 12345 : i32
      %c1103515245_i32 = arith.constant 1103515245 : i32
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c97 = arith.constant 97 : index
      scf.for %arg3 = %c0 to %c97 step %c1 {
        %0 = memref.load %arg0[] : memref<i64>
        %1 = memref.load %arg1[] : memref<i64>
        %2 = arith.trunci %0 : i64 to i32
        %3 = arith.trunci %1 : i64 to i32
        %4 = arith.addi %2, %3 : i32
        %5 = arith.muli %4, %c1103515245_i32 : i32
        %6 = arith.addi %5, %c12345_i32 : i32
        %7 = arith.index_cast %arg3 : index to i32
        %8 = arith.addi %7, %6 : i32
        %9 = arith.muli %8, %c1103515245_i32 : i32
        %10 = arith.addi %9, %c12345_i32 : i32
        %11 = arith.uitofp %10 : i32 to f32
        %12 = arith.mulf %11, %cst_0 : f32
        %13 = arith.addf %12, %cst : f32
        memref.store %13, %arg2[%c0, %arg3] : memref<1x97xf32>
      }
      return
    }
  }
  func.func @main(%arg0: memref<1x97xf32> {byre.argname = "Output0", byre.argtype = 2 : i32}) attributes {byre.entry_point} {
    %alloc = memref.alloc() : memref<256xi8>
    %0 = "byre.alias"(%alloc) {offset = 0 : i64} : (memref<256xi8>) -> memref<i64>
    byre.compute @GetSeed(%0) {memory_effects = [2 : i32]} : memref<i64>
    %1 = "byre.alias"(%alloc) {offset = 128 : i64} : (memref<256xi8>) -> memref<i64>
    byre.compute @NextOffset(%1) {memory_effects = [2 : i32]} : memref<i64>
    byre.compute @LLVMJITOp(%0, %1, %arg0) {kernel_name = "Unknown0", llvm_file_name = "host_kernels.ll", memory_effects = [1 : i32, 1 : i32, 2 : i32]} : memref<i64>, memref<i64>, memref<1x97xf32>
    return
  }
}