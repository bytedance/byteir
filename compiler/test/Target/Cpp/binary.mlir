// RUN: byteir-translate -emit-cpp %s | FileCheck %s

// CHECK-LABEL: binary_int
// CHECK-SAME: (int32_t [[V1:[^ ]*]], int32_t [[V2:[^ ]*]])
func.func @binary_int(%arg0 : i32, %arg1 : i32) -> i32 {
  // CHECK-NEXT: int32_t [[V3:[^ ]*]] = [[V1]] + [[V2]];
  %0 = arith.addi %arg0, %arg1: i32
  // CHECK-NEXT: int32_t [[V4:[^ ]*]] = [[V1]] - [[V3]];
  %1 = arith.subi %arg0, %0: i32
  // CHECK-NEXT: int32_t [[V5:[^ ]*]] = [[V3]] * [[V4]];
  %2 = arith.muli %0, %1: i32
  // CHECK-NEXT: int32_t [[V6:[^ ]*]] = [[V5]] / [[V4]];
  %3 = arith.divsi %2, %1: i32
  // CHECK-NEXT: int32_t [[V7:[^ ]*]] = [[V6]] % [[V5]];
  %4 = arith.remsi %3, %2: i32
  // CHECK-NEXT: int32_t [[V8:[^ ]*]] = [[V7]] << [[V6]];
  %5 = arith.shli %4, %3: i32
  // CHECK-NEXT: int32_t [[V9:[^ ]*]] = [[V8]] >> [[V7]];
  %6 = arith.shrsi %5, %4: i32
  // CHECK-NEXT: int32_t [[V10:[^ ]*]] = [[V9]] & [[V8]];
  %7 = arith.andi %6, %5: i32
  // CHECK-NEXT: bool [[V11:[^ ]*]] = [[V1]] < [[V2]];
  %8 = arith.cmpi "slt", %arg0, %arg1 : i32
  // CHECK-NEXT: bool [[V12:[^ ]*]] = [[V1]] == [[V2]];
  %9 = arith.cmpi "eq", %arg0, %arg1 : i32
  // CHECK-NEXT: bool [[V12:[^ ]*]] = [[V1]] != [[V2]];
  %10 = arith.cmpi "ne", %arg0, %arg1 : i32
  // CHECK-NEXT: bool [[V14:[^ ]*]] = [[V1]] > [[V2]];
  %11 = arith.cmpi "sgt", %arg0, %arg1 : i32
  return %7 : i32
}

// CHECK-LABEL: binary_float
// CHECK-SAME: (float [[V1:[^ ]*]], float [[V2:[^ ]*]])
func.func @binary_float(%arg0 : f32, %arg1 : f32) -> f32 {
  // CHECK-NEXT: float [[V3:[^ ]*]] = [[V1]] + [[V2]];
  %0 = arith.addf %arg0, %arg1: f32
  // CHECK-NEXT: float [[V4:[^ ]*]] = [[V1]] - [[V3]];
  %1 = arith.subf %arg0, %0: f32
  // CHECK-NEXT: float [[V5:[^ ]*]] = [[V3]] * [[V4]];
  %2 = arith.mulf %0, %1: f32
  // CHECK-NEXT: float [[V6:[^ ]*]] = [[V5]] / [[V4]];
  %3 = arith.divf %2, %1: f32
  // CHECK-NEXT: bool [[V7:[^ ]*]] = [[V1]] == [[V2]];
  %4 = arith.cmpf "oeq", %arg0, %arg1 : f32
  // CHECK-NEXT: bool [[V8:[^ ]*]] = [[V1]] != [[V2]];
  %5 = arith.cmpf "one", %arg0, %arg1 : f32
  // CHECK-NEXT: bool [[V9:[^ ]*]] = [[V1]] < [[V2]];
  %6 = arith.cmpf "olt", %arg0, %arg1 : f32
  // CHECK-NEXT: bool [[V10:[^ ]*]] = [[V1]] > [[V2]];
  %7 = arith.cmpf "ogt", %arg0, %arg1 : f32
  return %3 : f32
}

// CHECK-LABEL: binary_bool
// CHECK-SAME: (bool [[V1:[^ ]*]], bool [[V2:[^ ]*]])
func.func @binary_bool(%arg0 : i1, %arg1 : i1) -> i1 {
  // CHECK-NEXT: bool [[V3:[^ ]*]] = [[V1]] & [[V2]];
  %0 = arith.andi %arg0, %arg1: i1
  // CHECK-NEXT: bool [[V4:[^ ]*]] = [[V1]] | [[V3]];
  %1 = arith.ori %arg0, %0: i1
  // CHECK-NEXT: bool [[V5:[^ ]*]] = [[V3]] ^ [[V4]];
  %2 = arith.xori  %0, %1: i1
  return %2 : i1
}
