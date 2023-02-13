// RUN: byteir-opt %s -generic-device-config="anchor-attr=__byteir_test_device__ compute-name=TestDeviceOp" | FileCheck %s

func.func private @device_func(memref<1x97xf32>, memref<1x6xf32>) -> memref<1x6xf32> attributes {__byteir_test_device__}
// CHECK-LABEL: func.func private @device_func
// CHECK-SAME: attributes {__byre__kernel_name = "device_func", __byteir_test_device__, arg_offsets = [0 : i32, 1 : i32, 2 : i32], byre_compute_name = "TestDeviceOp", byre_force_compute_name}