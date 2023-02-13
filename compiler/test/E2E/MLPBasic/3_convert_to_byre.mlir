// RUN: byteir-opt %s -convert-to-byre -cse | FileCheck %s


func.func private @Unknown0(%arg0: memref<32xf32>, %arg1: memref<128x32xf32>) -> memref<128x32xf32> attributes { byre_compute_name = "AddOp"}  {
  %0 = memref.alloc() : memref<128x32xf32>
  "lmhlo.broadcast_in_dim"(%arg0, %0) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (memref<32xf32>, memref<128x32xf32>) -> ()
  %1 = memref.alloc() : memref<128x32xf32>
  "lmhlo.add"(%arg1, %0, %1) : (memref<128x32xf32>, memref<128x32xf32>, memref<128x32xf32>) -> ()
  return %1 : memref<128x32xf32>
}

func.func @mlp(%arg0: memref<128x64xf32> {__placeholder__byre.argname = "A"}, %arg1: memref<64x32xf32> {__placeholder__byre.argname = "B"}, %arg2: memref<32xf32> {__placeholder__byre.argname = "C"}) -> (memref<128x32xf32> {__placeholder__byre.argname = "D"}) attributes { __placeholder__byre.entry_point} {
  %0 = memref.alloc() : memref<128x32xf32>
  "lmhlo.dot"(%arg0, %arg1, %0) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (memref<128x64xf32>, memref<64x32xf32>, memref<128x32xf32>) -> ()
  %1 = call @Unknown0(%arg2, %0) : (memref<32xf32>, memref<128x32xf32>) -> memref<128x32xf32>
  return %1 : memref<128x32xf32>
}
// CHECK:     byre.compute @MatmulOp
// CHECK:     byre.compute @AddOp

