// RUN: byteir-opt --to-llvm %s | FileCheck %s

module attributes {byteir.llvm_module} {
  func.func @Unknown0(%arg0: memref<32xf32>, %arg1: memref<32xf32>) attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    scf.for %arg2 = %c0 to %c32 step %c1 {
      %0 = memref.load %arg0[%arg2] : memref<32xf32>
      %1 = math.tanh %0 : f32
      memref.store %1, %arg1[%arg2] : memref<32xf32>
    }
    return
  }
  // CHECK-LABEL: llvm.func @tanhf
  //   CHECK: llvm.call @tanhf
}