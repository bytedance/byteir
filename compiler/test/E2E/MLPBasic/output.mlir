// RUN: byteir-opt %s | FileCheck %s

module attributes {byre.container_module}  {
  func.func @mlp_no_broadcast(%arg0: memref<128x64xf32> {byre.argname = "A", byre.argtype = 1 : i32}, %arg1: memref<64x32xf32> {byre.argname = "B", byre.argtype = 1 : i32}, %arg2: memref<128x32xf32> {byre.argname = "C", byre.argtype = 1 : i32}, %arg3: memref<128x32xf32> {byre.argname = "D", byre.argtype = 2 : i32}) attributes {byre.entry_point} {
    %0 = memref.alloc() : memref<128x32xf32>
    byre.compute @MatmulOp(%arg0, %arg1, %0) : memref<128x64xf32>, memref<64x32xf32>, memref<128x32xf32>
    byre.compute @AddOp(%0, %arg2, %arg3) : memref<128x32xf32>, memref<128x32xf32>, memref<128x32xf32>
    return
  }
  // CHECK-LABEL: func.func @mlp_no_broadcast
}

