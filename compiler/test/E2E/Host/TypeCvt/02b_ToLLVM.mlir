// RUN: byteir-opt %s --to-llvm | FileCheck %s

// CHECK: llvm.func

module attributes {byre.container_module} {
  module attributes {byteir.llvm_module} {
    func.func @Unknown0(%arg0: memref<1x224x224x3xf32>, %arg1: memref<1x224x224x3xf16>) attributes {__byre__kernel_name = "Unknown0", __byre__llvm_file_name = "host_kernels.ll", __byteir_hlo_aggressive_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "LLVMJITOp", byre_force_compute_name, llvm.emit_c_interface} {
      %c1 = arith.constant 1 : index
      %c150528 = arith.constant 150528 : index
      %c0 = arith.constant 0 : index
      %collapse_shape = memref.collapse_shape %arg0 [[0, 1, 2, 3]] : memref<1x224x224x3xf32> into memref<150528xf32>
      %collapse_shape_0 = memref.collapse_shape %arg1 [[0, 1, 2, 3]] : memref<1x224x224x3xf16> into memref<150528xf16>
      scf.for %arg2 = %c0 to %c150528 step %c1 {
        %0 = memref.load %collapse_shape[%arg2] : memref<150528xf32>
        %1 = arith.truncf %0 : f32 to f16
        memref.store %1, %collapse_shape_0[%arg2] : memref<150528xf16>
      }
      return
    }
  }
  func.func @main(%arg0: memref<1x224x224x3xf32> {byre.argname = "Input0", byre.argtype = 1 : i32}, %arg1: memref<1x224x224x3xf16> {byre.argname = "Output0", byre.argtype = 2 : i32}) attributes {byre.entry_point} {
    byre.compute @LLVMJITOp(%arg0, %arg1) {kernel_name = "Unknown0", llvm_file_name = "host_kernels.ll", memory_effects = [1 : i32, 2 : i32]} : memref<1x224x224x3xf32>, memref<1x224x224x3xf16>
    return
  }
}