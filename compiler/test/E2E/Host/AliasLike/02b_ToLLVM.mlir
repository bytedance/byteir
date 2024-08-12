// RUN: byteir-opt %s --to-llvm | FileCheck %s

// CHECK: llvm.func

module attributes {byre.container_module} {
  module attributes {byteir.llvm_module} {
    func.func @Unknown0(%arg0: memref<512x200xf32>, %arg1: memref<512x200xf32>, %arg2: memref<512x200xf32>) attributes {__byre__kernel_name = "Unknown0", __byre__llvm_file_name = "host_kernels.ll", __byteir_hlo_aggressive_fusion__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "LLVMJITOp", byre_force_compute_name, llvm.emit_c_interface} {
      %c0 = arith.constant 0 : index
      %c102400 = arith.constant 102400 : index
      %c1 = arith.constant 1 : index
      %collapse_shape = memref.collapse_shape %arg0 [[0, 1]] : memref<512x200xf32> into memref<102400xf32>
      %collapse_shape_0 = memref.collapse_shape %arg1 [[0, 1]] : memref<512x200xf32> into memref<102400xf32>
      %collapse_shape_1 = memref.collapse_shape %arg2 [[0, 1]] : memref<512x200xf32> into memref<102400xf32>
      scf.for %arg3 = %c0 to %c102400 step %c1 {
        %0 = memref.load %collapse_shape[%arg3] : memref<102400xf32>
        %1 = memref.load %collapse_shape_0[%arg3] : memref<102400xf32>
        %2 = arith.addf %0, %1 : f32
        memref.store %2, %collapse_shape_1[%arg3] : memref<102400xf32>
      }
      return
    }
  }
  func.func @main(%arg0: memref<512x200xf32, "cpu"> {byre.argname = "Input0", byre.argtype = 1 : i32}, %arg1: memref<512x200xf32, "cpu"> {byre.argname = "Input1", byre.argtype = 1 : i32}, %arg2: memref<128x2x100xf32, "cpu"> {byre.argname = "Output0", byre.argtype = 2 : i32}, %arg3: memref<128x2x100xf32, "cpu"> {byre.argname = "Output1", byre.argtype = 2 : i32}, %arg4: memref<1x100xf32, "cpu"> {byre.argname = "Output2", byre.argtype = 2 : i32}, %arg5: memref<1x100xf32, "cpu"> {byre.argname = "Output3", byre.argtype = 2 : i32}, %arg6: memref<512x200xf32, "cpu"> {byre.argname = "Output4", byre.argtype = 2 : i32}) attributes {byre.entry_point} {
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