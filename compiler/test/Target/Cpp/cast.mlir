// RUN: byteir-translate -emit-cpp %s | FileCheck %s

// CHECK-LABEL: test_cast
// CHECK-SAME: (int32_t [[V1:[^ ]*]], uint32_t [[V2:[^ ]*]], size_t [[V3:[^ ]*]])
func.func @test_cast(%arg0 : i32, %arg1 : ui32, %arg2: index) -> i32 {
  // CHECK-NEXT: float [[V4:[^ ]*]] = (float)([[V1]]);
  %0 = arith.sitofp %arg0: i32 to f32
  // CHECK-NEXT: int32_t [[V5:[^ ]*]] = (int32_t)([[V3]]);
  %1 = arith.index_cast %arg2: index to i32
  // CHECK-NEXT: float [[V6:[^ ]*]] = (float)([[V2]]);
  %2 = builtin.unrealized_conversion_cast %arg1: ui32 to f32
  // CHECK-NEXT: int32_t [[V7:[^ ]*]] = (int32_t)([[V4]]);
  %3 = arith.fptosi %0: f32 to i32 
  return %3 : i32
}

