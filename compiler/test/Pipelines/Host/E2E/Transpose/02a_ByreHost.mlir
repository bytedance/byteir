// RUN: byteir-opt %s -byre-host="device-file-name=your_file target=cpu" | FileCheck %s

// CHECK-LABEL: func.func @main

module attributes {byre.container_module} {
  module attributes {byteir.llvm_module} {
    func.func @Unknown0(%arg0: memref<1x32x64x64xf32>, %arg1: memref<1x64x64x32xf32>) attributes {__byre__kernel_name = "Unknown0", __byre__llvm_file_name = "host_kernels.ll", __byteir_hlo_aggressive_fusion__, arg_offsets = [0 : i32, 1 : i32], byre_compute_name = "LLVMJITOp", byre_force_compute_name, llvm.emit_c_interface} {
      %cst = arith.constant 0.000000e+00 : f32
      %c64 = arith.constant 64 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c2048 = arith.constant 2048 : index
      scf.for %arg2 = %c0 to %c2048 step %c1 {
        %0 = arith.remsi %arg2, %c8 : index
        %1 = arith.divsi %arg2, %c8 : index
        %2 = arith.remsi %1, %c64 : index
        %3 = arith.divsi %1, %c64 : index
        %4 = arith.muli %0, %c8 : index
        %5 = arith.muli %3, %c8 : index
        %6 = vector.transfer_read %arg0[%c0, %5, %2, %4], %cst {in_bounds = [true, true, true, true]} : memref<1x32x64x64xf32>, vector<1x8x1x8xf32>
        %7 = vector.transpose %6, [0, 2, 3, 1] : vector<1x8x1x8xf32> to vector<1x1x8x8xf32>
        vector.transfer_write %7, %arg1[%c0, %2, %4, %5] {in_bounds = [true, true, true, true]} : vector<1x1x8x8xf32>, memref<1x64x64x32xf32>
      }
      return
    }
  }
  func.func @main(%arg0: memref<1x32x64x64xf32> {byre.argname = "Input0", byre.argtype = 1 : i32}, %arg1: memref<1x64x64x32xf32> {byre.argname = "Output0", byre.argtype = 2 : i32}) attributes {byre.entry_point} {
    byre.compute @LLVMJITOp(%arg0, %arg1) {kernel_name = "Unknown0", llvm_file_name = "host_kernels.ll", memory_effects = [1 : i32, 2 : i32]} : memref<1x32x64x64xf32>, memref<1x64x64x32xf32>
    return
  }
}