// RUN: byteir-translate -emit-cpp %s | FileCheck %s

// CHECK-LABEL: func_external
// CHECK-SAME: (int32_t, int32_t)
func.func private @func_external(%arg0 : i32, %arg1 : i32)

// CHECK-LABEL: func_int
// CHECK-SAME: (int32_t [[V1:[^ ]*]], int32_t [[V2:[^ ]*]])
func.func @func_int(%arg0 : i32, %arg1 : i32) -> () {
  return
}

// CHECK-LABEL: func_memref
// CHECK-SAME: (float*, int32_t)
func.func private @func_memref(%arg0 : memref<2x3xf32>, %arg1 : i32)