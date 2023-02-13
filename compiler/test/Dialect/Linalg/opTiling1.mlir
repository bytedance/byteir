// RUN: byteir-opt %s -linalg-op-tile="tile-sizes=4" -canonicalize-ext | FileCheck %s

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
//CHECK:   linalg_ext.scan
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
//CHECK:   linalg_ext.scan
//CHECK: } 
//CHECK: return

func.func @softmax_tensor(%arg0: tensor<1024x64xf32>) -> (tensor<1024x64xf32>) {
  %0 = tensor.empty() : tensor<1024x64xf32>
  %1 = tensor.empty() : tensor<1024xf32>
  %2 = tensor.empty() : tensor<1024xf32>
  %3 = tensor.empty() : tensor<1024xf32>
  %4:4 = linalg_ext.softmax
    {__internal_linalg_transform__ = "__byteir_tile__"}
    dimension(1) 
    ins(%arg0 : tensor<1024x64xf32>) outs(%0, %1, %2, %3 : tensor<1024x64xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>) : tensor<1024x64xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>
  return %4#0 : tensor<1024x64xf32>
}
//CHECK-LABEL: func.func @softmax_tensor
//CHECK: scf.for
//CHECK:   linalg_ext.softmax
//CHECK:   scf.yield
//CHECK: } {__byteir_parallel__}
//CHECK: return

func.func @softmax_memref(%arg0: memref<1024x64xf32>) -> (memref<1024x64xf32>) {
  %0 = memref.alloc() : memref<1024x64xf32>
  %1 = memref.alloc() : memref<1024xf32>
  %2 = memref.alloc() : memref<1024xf32>
  %3 = memref.alloc() : memref<1024xf32>
  linalg_ext.softmax
    {__internal_linalg_transform__ = "__byteir_tile__"}
    dimension(1)
    ins(%arg0 : memref<1024x64xf32>) outs(%0, %1, %2, %3 : memref<1024x64xf32>, memref<1024xf32>, memref<1024xf32>, memref<1024xf32>)
  return %0 : memref<1024x64xf32>
}
//CHECK-LABEL: func.func @softmax_memref
//CHECK: scf.for
//CHECK:   linalg_ext.softmax
//CHECK: } {__byteir_parallel__}
//CHECK: return
