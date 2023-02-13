// RUN: byteir-opt %s -linalg-op-tile="tile-sizes=2,4" -canonicalize-ext | FileCheck %s

func.func @scan_2d_tensor(%0: tensor<16x32xi32>) -> tensor<16x32xi32> {
  %c0 = tensor.empty() : tensor<32xi32>
  %1 = tensor.empty() : tensor<16x32xi32>
  %2:2 = linalg_ext.scan
    {__internal_linalg_transform__ = "__byteir_tile__"}
    dimension(0) inclusive(true)
    ins(%0 : tensor<16x32xi32>) outs(%1, %c0 : tensor<16x32xi32>, tensor<32xi32>) {
    ^bb0(%arg0 : i32, %arg1 : i32):
      %sum = arith.addi %arg0, %arg1 : i32
      linalg_ext.yield %sum : i32
  } -> tensor<16x32xi32>, tensor<32xi32>
  return %2#0 : tensor<16x32xi32>
}
//CHECK-LABEL: func.func @scan_2d_tensor
//CHECK: scf.for
//CHECK:   scf.for
//CHECK:     linalg_ext.scan
//CHECK:     scf.yield
//CHECK:   } {__byteir_parallel__}
//CHECK:   scf.yield
//CHECK: } 
//CHECK: return

func.func @scan_2d_memref(%0: memref<16x32xi32>, %1: memref<16x32xi32>) {
  %c0 = memref.alloc() : memref<32xi32>
  linalg_ext.scan
    {__internal_linalg_transform__ = "__byteir_tile__"}
    dimension(0) inclusive(true)
    ins(%0 : memref<16x32xi32>) outs(%1, %c0 : memref<16x32xi32>, memref<32xi32>) {
    ^bb0(%arg0 : i32, %arg1 : i32):
      %sum = arith.addi %arg0, %arg1 : i32
      linalg_ext.yield %sum : i32
  }
  return
}
//CHECK-LABEL: func.func @scan_2d_memref
//CHECK: scf.for
//CHECK:   scf.for
//CHECK:     linalg_ext.scan
//CHECK:   } {__byteir_parallel__}
//CHECK: } 
//CHECK: return
