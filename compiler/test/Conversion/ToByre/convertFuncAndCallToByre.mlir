// RUN: byteir-opt --convert-func-and-call-to-byre %s | FileCheck %s

module {
// CHECK: module attributes {byre.container_module}  {

  func.func @mhlo_add(%arg0: memref<4xf32> {__placeholder__byre.argname = "A"}, %arg1: memref<4xf32> {__placeholder__byre.argname = "B"}) -> (memref<4xf32> {__placeholder__byre.argname = "C"}) attributes { __placeholder__byre.entry_point } {
    %0 = call @some_func(%arg0, %arg1) : (memref<4xf32>, memref<4xf32>) -> memref<4xf32>
    return %0 : memref<4xf32>
  }

  func.func private @some_func(%arg0: memref<4xf32>, %arg1: memref<4xf32>) -> memref<4xf32> attributes { byre_compute_name = "customAddOp"}  {
    %0 = memref.alloc() : memref<4xf32>
    "lmhlo.add"(%arg0, %arg1, %0) : (memref<4xf32>, memref<4xf32>, memref<4xf32>) -> ()
    %1 = memref.alloc() : memref<4xf32>
    "lmhlo.add"(%arg0, %0, %1) : (memref<4xf32>, memref<4xf32>, memref<4xf32>) -> ()
    return %1 : memref<4xf32>
  }

// CHECK:   func.func @mhlo_add(%[[ARG_0:.*]]: memref<4xf32> {byre.argname = "A", byre.argtype = 1 : i32}, %[[ARG_1:.*]]: memref<4xf32> {byre.argname = "B", byre.argtype = 1 : i32}, %[[ARG_2:.*]]: memref<4xf32> {byre.argname = "C", byre.argtype = 2 : i32}) attributes {byre.entry_point} {
// CHECK:     byre.compute @customAddOp(%[[ARG_0]], %[[ARG_1]], %[[ARG_2]])
//   CHECK-SAME: memory_effects = [1 : i32, 1 : i32, 2 : i32]
//   CHECK-SAME: memref<4xf32>, memref<4xf32>, memref<4xf32>
// CHECK:     return

  func.func @mhlo_add_2(%arg0: memref<4xf32> {__placeholder__byre.argname = "A"}, %arg1: memref<4xf32> {__placeholder__byre.argname = "B"}) -> (memref<4xf32> {__placeholder__byre.argname = "C"}, memref<4xf32> {__placeholder__byre.argname = "D"}) attributes { __placeholder__byre.entry_point } {
    %0:2 = call @some_func_2(%arg0, %arg1) : (memref<4xf32>, memref<4xf32>) -> (memref<4xf32>, memref<4xf32>)
    return %0#0, %0#1 : memref<4xf32>, memref<4xf32>
  }

  func.func private @some_func_2(%arg0: memref<4xf32>, %arg1: memref<4xf32>) -> (memref<4xf32>, memref<4xf32>) attributes { byre_compute_name = "customAddOp2"}  {
    %0 = memref.alloc() : memref<4xf32>
    "lmhlo.add"(%arg0, %arg1, %0) : (memref<4xf32>, memref<4xf32>, memref<4xf32>) -> ()
    %1 = memref.alloc() : memref<4xf32>
    "lmhlo.add"(%arg0, %0, %1) : (memref<4xf32>, memref<4xf32>, memref<4xf32>) -> ()
    return %0, %1 : memref<4xf32>, memref<4xf32>
  }

// CHECK:   func.func @mhlo_add_2(%[[ARG_0:.*]]: memref<4xf32> {byre.argname = "A", byre.argtype = 1 : i32}, %[[ARG_1:.*]]: memref<4xf32> {byre.argname = "B", byre.argtype = 1 : i32}, %[[ARG_2:.*]]: memref<4xf32> {byre.argname = "C", byre.argtype = 2 : i32}, %[[ARG_3:.*]]: memref<4xf32> {byre.argname = "D", byre.argtype = 2 : i32}) attributes {byre.entry_point} {
// CHECK:     byre.compute @customAddOp2(%[[ARG_0]], %[[ARG_1]], %[[ARG_2]], %[[ARG_3]])
//   CHECK-SAME: memory_effects = [1 : i32, 1 : i32, 2 : i32, 2 : i32]
//   CHECK-SAME: memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>
// CHECK:     return

  func.func @test_pass_through(%arg0: memref<4xf32> {__placeholder__byre.argname = "A"}, %arg1: memref<4xf32> {__placeholder__byre.argname = "B"}) -> (memref<4xf32> {__placeholder__byre.argname = "C"}, memref<4xf32> {__placeholder__byre.argname = "D"}) attributes { __placeholder__byre.entry_point } {
    %0:2 = call @some_func_3(%arg0, %arg1) : (memref<4xf32>, memref<4xf32>) -> (memref<4xf32>, memref<4xf32>)
    return %0#0, %0#1 : memref<4xf32>, memref<4xf32>
  }

  func.func private @some_func_3(%arg0: memref<4xf32>, %arg1: memref<4xf32>) -> (memref<4xf32>, memref<4xf32>) attributes { byre_compute_name = "customAddOp3", arg_offsets = [0 : i32, 2 : i32, 1 : i32], passthrough_arg = [3 : i32, 2 : i32]}

// CHECK:   func.func @test_pass_through(%[[ARG_0:.*]]: memref<4xf32> {byre.argname = "A", byre.argtype = 1 : i32}, %[[ARG_1:.*]]: memref<4xf32> {byre.argname = "B", byre.argtype = 1 : i32}, %[[ARG_2:.*]]: memref<4xf32> {byre.argname = "C", byre.argtype = 2 : i32}, %[[ARG_3:.*]]: memref<4xf32> {byre.argname = "D", byre.argtype = 2 : i32}) attributes {byre.entry_point} {
// CHECK:     byre.compute @customAddOp3(%[[ARG_0]], %[[ARG_2]], %[[ARG_1]])
//   CHECK-SAME: memory_effects = [1 : i32, 2 : i32, 1 : i32]
//   CHECK-SAME: memref<4xf32>, memref<4xf32>, memref<4xf32>
// CHECK:     byre.copy(%[[ARG_2]], %[[ARG_3]]) : memref<4xf32>, memref<4xf32>
// CHECK:     return

}



