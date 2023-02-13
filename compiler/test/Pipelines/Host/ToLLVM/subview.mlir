// RUN: byteir-opt --to-llvm %s | FileCheck %s

module attributes {byteir.llvm_module} {
  func.func @subview(%arg0: memref<32x128xi32>) -> memref <32x64xi32, strided<[128, 1]>> attributes {llvm.emit_c_interface} {
    %0 = memref.subview %arg0[0, 0] [32, 64] [1, 1] : memref<32x128xi32> to memref <32x64xi32, strided<[128, 1]>>
    return %0: memref <32x64xi32, strided<[128, 1]>>
  }
  // CHECK-LABEL: llvm.func @subview
  //   CHECK: llvm.mlir.undef : !llvm.struct
  // CHECK-LABEL: llvm.func @_mlir_ciface_subview
}