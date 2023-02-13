// RUN: byteir-opt -convert-lmhlo-to-byre %s | FileCheck %s

module attributes {byre.container_module} {
// CHECK: module attributes {byre.container_module}  {
  func.func @mhlo_add(%arg0: memref<4xf32> {byre.argname = "A", byre.argtype = 1 : i32}, %arg1: memref<4xf32> {byre.argname = "B", byre.argtype = 1 : i32}, %arg2: memref<4xf32> {byre.argname = "C", byre.argtype = 2 : i32}) attributes { byre.entry_point } {
    "lmhlo.add"(%arg0, %arg1, %arg2) : (memref<4xf32>, memref<4xf32>, memref<4xf32>) -> ()
    return
  }
// CHECK-LABEL: func.func @mhlo_add
//   CHECK:  byre.compute @AddOp(%arg0, %arg1, %arg2)
//     CHECK-SAME: memory_effects = [1 : i32, 1 : i32, 2 : i32]
  func.func @lace_reshape(%arg0: memref<1x1024xi64> {byre.argname = "A", byre.argtype = 1 : i32}, %arg1: memref<32x32xi64> {byre.argname = "B", byre.argtype = 2 : i32}) attributes { byre.entry_point } {
    %0 = "lace.reshape"(%arg0) : (memref<1x1024xi64>) -> memref<32x32xi64>
    "lmhlo.add"(%0, %0, %arg1) : (memref<32x32xi64>, memref<32x32xi64>, memref<32x32xi64>) -> ()
    return
  }
// CHECK-LABEL: func.func @lace_reshape
//   CHECK:  byre.alias
//     CHECK-SAME: offset = 0
//   CHECK:  byre.compute @AddOp

 func.func @lace_slice(%arg0: memref<1x512xi64> {byre.argname = "A", byre.argtype = 1 : i32}, %arg1: memref<1x128xi64> {byre.argname = "B", byre.argtype = 2 : i32}) attributes { byre.entry_point } {
    %0 = "lace.slice"(%arg0) {limit_indices = dense<[1, 256]> : tensor<2xi64>, start_indices = dense<[0, 128]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (memref<1x512xi64>) -> memref<1x128xi64>
    "lmhlo.add"(%0, %0, %arg1) : (memref<1x128xi64>, memref<1x128xi64>, memref<1x128xi64>) -> ()
    return
  }
// CHECK-LABEL: func.func @lace_slice
//   CHECK:  byre.alias
//     CHECK-SAME: offset = 1024
//   CHECK:  byre.compute @AddOp

 func.func @lace_string_eq(%arg0: memref<!ace.string> {byre.argname = "A", byre.argtype = 1 : i32}, %arg1: memref<i1> {byre.argname = "B", byre.argtype = 2 : i32}) attributes { byre.entry_point } {
    %0 = memref.alloc() : memref<!ace.string>
    "lace.constant"(%0) {value = dense<"constant"> : tensor<!ace.string>} : (memref<!ace.string>) -> ()
    "lace.custom_call"(%arg0, %0, %arg1) {call_target_name="tf.Equal", operand_segment_sizes = array<i32: 2,1>} : (memref<!ace.string>, memref<!ace.string>, memref<i1>) -> ()
    return
  }
// CHECK-LABEL: func.func @lace_string_eq
//   CHECK-NEXT: %[[Constant:.*]] = memref.alloc
//   CHECK-NEXT: byre.compute @FillOp(%[[Constant]])
//     CHECK-SAME: value = dense<"constant">
//   CHECK-NEXT: byre.compute @tf.Equal(%arg0, %[[Constant]], %arg1)
}
