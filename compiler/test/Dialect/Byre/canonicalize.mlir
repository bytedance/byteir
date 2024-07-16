// RUN: byteir-opt --canonicalize %s | FileCheck %s

module attributes {byre.container_module} {
  func.func @collapse_alias(%arg0: memref<512xf32> {byre.argtype = 1: i32, byre.argname = "A"}, %arg1: memref<512xf32> {byre.argtype = 2: i32, byre.argname = "B"}) attributes {byre.entry_point} {
// CHECK-LABEL: func.func @collapse_alias
    %0 = "byre.alias"(%arg0) {offset = 128 : i64} : (memref<512xf32>) -> memref<256xf32>
    %1 = "byre.alias"(%0) {offset = 32 : i64} : (memref<256xf32>) -> memref<128xf32>
// CHECK: byre.alias
//   CHECK-SAME: offset = 160
    %2 = memref.alloc() : memref<512xf32>
    %3 = "byre.alias"(%2) {offset = 128 : i64} : (memref<512xf32>) -> memref<256xf32>
    %4 = "byre.alias"(%3) {offset = 32 : i64} : (memref<256xf32>) -> memref<128xf32>
// CHECK: byre.alias
//   CHECK-SAME: offset = 160
    byre.compute @SomeOp(%1, %4, %arg1) : memref<128xf32>, memref<128xf32>, memref<512xf32>
    return
  }

  func.func @collapse_alias_bitwidth(%arg0: memref<3x64xf32, "cuda"> {byre.argname = "Input0", byre.argtype = 1 : i32}, %arg1: memref<1x64xf32, "cuda"> {byre.argname = "Output0", byre.argtype = 2 : i32}) attributes {byre.entry_point} {
// CHECK-LABEL: func.func @collapse_alias_bitwidth
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() : memref<768xi8, "cuda">
    %0 = "byre.alias"(%alloc) <{offset = 0 : i64}> : (memref<768xi8, "cuda">) -> memref<3x64xf32, "cuda">
// CHECK: byre.alias
//   CHECK-SAME: offset = 0
    byre.compute @PTXOp(%arg0, %0) {BlockSize.x = 256 : i32, GridSize.x = 1 : i32, arg_ranks = [2 : i32, 2 : i32], call_convention = "bare_ptr", kernel_name = "Unknown0", memory_effects = [1 : i32, 2 : i32]} : memref<3x64xf32, "cuda">, memref<3x64xf32, "cuda">
    %1 = "byre.alias"(%0) <{offset = 0 : i64}> : (memref<3x64xf32, "cuda">) -> memref<1x64xf32, strided<[64, 1]>, "cuda">
    %2 = "byre.alias"(%0) <{offset = 0 : i64}> : (memref<3x64xf32, "cuda">) -> memref<1x64xf32, strided<[64, 1], offset: 64>, "cuda">
    %3 = "byre.alias"(%0) <{offset = 0 : i64}> : (memref<3x64xf32, "cuda">) -> memref<1x64xf32, strided<[64, 1], offset: 128>, "cuda">
    %4 = "byre.alias"(%1) <{offset = 0 : i64}> : (memref<1x64xf32, strided<[64, 1]>, "cuda">) -> memref<1x64xf32, "cuda">
// CHECK: byre.alias
//   CHECK-SAME: offset = 0
    %5 = "byre.alias"(%2) <{offset = 64 : i64}> : (memref<1x64xf32, strided<[64, 1], offset: 64>, "cuda">) -> memref<1x64xf32, "cuda">
// CHECK: byre.alias
//   CHECK-SAME: offset = 256
    %6 = "byre.alias"(%3) <{offset = 128 : i64}> : (memref<1x64xf32, strided<[64, 1], offset: 128>, "cuda">) -> memref<1x64xf32, "cuda">
// CHECK: byre.alias
//   CHECK-SAME: offset = 512
    byre.compute @PTXOp(%4, %5, %6, %arg1) {BlockSize.x = 256 : i32, GridSize.x = 1 : i32, arg_ranks = [2 : i32, 2 : i32, 2 : i32, 2 : i32], call_convention = "bare_ptr", kernel_name = "Unknown1", memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]} : memref<1x64xf32, "cuda">, memref<1x64xf32, "cuda">, memref<1x64xf32, "cuda">, memref<1x64xf32, "cuda">
    return
  }

  func.func @remove_identity_alias(%arg0: memref<256xf32> {byre.argtype = 1: i32, byre.argname = "A"}, %arg1: memref<256xf32> {byre.argtype = 2: i32, byre.argname = "B"}) attributes {byre.entry_point} {
// CHECK-LABEL: func.func @remove_identity_alias
    %0 = "byre.alias"(%arg0) {offset = 0 : i64} : (memref<256xf32>) -> memref<256xf32>
    byre.compute @SomeOp(%0, %arg1) : memref<256xf32>, memref<256xf32>
// CHECK-NEXT: byre.compute @SomeOp(%arg0, %arg1)
    return
  }

  func.func @test_group_copy(%arg0 : memref<100x?xf32> {byre.argtype = 1: i32, byre.argname = "A"}, %arg1 : memref<100x?xf32> {byre.argtype = 2: i32, byre.argname = "B"}, %arg2 : memref<200x?xf32> {byre.argtype = 1: i32, byre.argname = "C"}, %arg3 : memref<200x?xf32> {byre.argtype = 2: i32, byre.argname = "D"}) attributes {byre.entry_point} {
// CHECK-LABEL: func.func @test_group_copy
    "byre.group_copy"(%arg0, %arg2, %arg1, %arg2) {callee = "d2h_array"} : (memref<100x?xf32>, memref<200x?xf32>, memref<100x?xf32>, memref<200x?xf32>) -> ()
// CHECK-NEXT: byre.copy(%arg0, %arg1) {callee = "d2h"}
    return
  }
}
