// RUN: byteir-opt %s -generic-device-config="anchor-attr=__byteir_test_device__ compute-name=TestDeviceOp" -byre-opt | FileCheck %s

// CHECK: module attributes {byre.container_module} 
func.func @main(%arg0: memref<1x97xf32>, %arg1: memref<1x6xf32>, %arg2: memref<1x6xf32>) -> memref<1x6xf32> {
  %0 = call @device_func(%arg0, %arg2) : (memref<1x97xf32>, memref<1x6xf32>) -> memref<1x6xf32>
  return %0 : memref<1x6xf32>
}
func.func private @device_func(memref<1x97xf32>, memref<1x6xf32>) -> memref<1x6xf32> attributes {__byteir_test_device__}

// CHECK-LABEL: func.func @main
// CHECK-SAME: attributes {byre.entry_point}
// CHECK-NEXT: byre.compute @TestDeviceOp
//   CHECK-DAG: kernel_name = "device_func"
//   CHECK-DAG: memory_effects = [1 : i32, 1 : i32, 2 : i32]
