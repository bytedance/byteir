// RUN: byteir-opt %s -set-op-space="entry-func=main space=cpu" -set-arg-space="entry-func=main all-space=cpu" --allow-unregistered-dialect | FileCheck %s

module attributes {byre.container_module} {
  func.func @main(%arg0: memref<128x64xf32> {byre.argname = "in0", byre.argtype = 1: i32}, %arg1: memref<64x32xf32> {byre.argname = "in1", byre.argtype = 1: i32}, %arg2: memref<32xf32> {byre.argname = "in2", byre.argtype = 1: i32}, %arg3: memref<128x32xf32> {byre.argname = "out0", byre.argtype = 2: i32}) attributes {byre.entry_point} {
    %0 = memref.alloc() : memref<128x64xf16>
    byre.compute @ConvertOp(%arg0, %0) {memory_effects = [1 : i32, 2 : i32]} : memref<128x64xf32>, memref<128x64xf16>
    %1 = memref.alloc() : memref<64x32xf16>
    byre.compute @ConvertOp(%arg1, %1) {memory_effects = [1 : i32, 2 : i32]} : memref<64x32xf32>, memref<64x32xf16>
    %2 = memref.alloc() : memref<32xf16>
    byre.compute @ConvertOp(%arg2, %2) {memory_effects = [1 : i32, 2 : i32]} : memref<32xf32>, memref<32xf16>
    %3 = call @func_with_bufferization() : () -> memref<1x97xf32>
    %4 = call @mlp_device(%0, %1, %2, %3) : (memref<128x64xf16>, memref<64x32xf16>, memref<32xf16>, memref<1x97xf32>) -> memref<128x32xf16>
    byre.compute @ConvertOp(%4, %arg3) {memory_effects = [1 : i32, 2 : i32]} : memref<128x32xf16>, memref<128x32xf32>
    return
  }
  func.func private @mlp_device(memref<128x64xf16>, memref<64x32xf16>, memref<32xf16>, memref<1x97xf32>) -> memref<128x32xf16> attributes {device = "test"}

  func.func private @func_with_bufferization() -> memref<1x97xf32> attributes {device = "cpu"} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %2 = mhlo.constant dense<[1, 97]> : tensor<2xi64>
    %3 = "mhlo.rng"(%0, %1, %2) {device = "host", rng_distribution = #mhlo.rng_distribution<UNIFORM>} : (tensor<f32>, tensor<f32>, tensor<2xi64>) -> tensor<1x97xf32>
    %4 = bufferization.to_memref %3 {device = "cpu"} : memref<1x97xf32>
    return %4 : memref<1x97xf32>
  }
}
// CHECK-LABEL: func.func @main(%arg0: memref<128x64xf32, "cpu"> {byre.argname = "in0", byre.argtype = 1 : i32}, %arg1: memref<64x32xf32, "cpu"> {byre.argname = "in1", byre.argtype = 1 : i32}, %arg2: memref<32xf32, "cpu"> {byre.argname = "in2", byre.argtype = 1 : i32}, %arg3: memref<128x32xf32, "cpu"> {byre.argname = "out0", byre.argtype = 2 : i32}) attributes {byre.entry_point} {
// CHECK-NEXT:    %alloc = memref.alloc() : memref<128x64xf16, "test">
// CHECK-NEXT:    %alloc_0 = memref.alloc() : memref<128x64xf16, "cpu">
// CHECK-NEXT:    byre.compute @ConvertOp(%arg0, %alloc_0) {device = "cpu", memory_effects = [1 : i32, 2 : i32]} : memref<128x64xf32, "cpu">, memref<128x64xf16, "cpu">
// CHECK-NEXT:    memref.copy %alloc_0, %alloc : memref<128x64xf16, "cpu"> to memref<128x64xf16, "test">
// CHECK-NEXT:    %alloc_1 = memref.alloc() : memref<64x32xf16, "test">
// CHECK-NEXT:    %alloc_2 = memref.alloc() : memref<64x32xf16, "cpu">
// CHECK-NEXT:    byre.compute @ConvertOp(%arg1, %alloc_2) {device = "cpu", memory_effects = [1 : i32, 2 : i32]} : memref<64x32xf32, "cpu">, memref<64x32xf16, "cpu">
// CHECK-NEXT:    memref.copy %alloc_2, %alloc_1 : memref<64x32xf16, "cpu"> to memref<64x32xf16, "test">
// CHECK-NEXT:    %alloc_3 = memref.alloc() : memref<32xf16, "test">
// CHECK-NEXT:    %alloc_4 = memref.alloc() : memref<32xf16, "cpu">
// CHECK-NEXT:    byre.compute @ConvertOp(%arg2, %alloc_4) {device = "cpu", memory_effects = [1 : i32, 2 : i32]} : memref<32xf32, "cpu">, memref<32xf16, "cpu">
// CHECK-NEXT:    memref.copy %alloc_4, %alloc_3 : memref<32xf16, "cpu"> to memref<32xf16, "test">
// CHECK-NEXT:    %0 = call @func_with_bufferization() : () -> memref<1x97xf32, "cpu">
// CHECK-NEXT:    %alloc_5 = memref.alloc() : memref<1x97xf32, "test">
// CHECK-NEXT:    memref.copy %0, %alloc_5 : memref<1x97xf32, "cpu"> to memref<1x97xf32, "test">
// CHECK-NEXT:    %1 = call @mlp_device(%alloc, %alloc_1, %alloc_3, %alloc_5) : (memref<128x64xf16, "test">, memref<64x32xf16, "test">, memref<32xf16, "test">, memref<1x97xf32, "test">) -> memref<128x32xf16, "test">
// CHECK-NEXT:    %alloc_6 = memref.alloc() : memref<128x32xf16, "cpu">
// CHECK-NEXT:    memref.copy %1, %alloc_6 : memref<128x32xf16, "test"> to memref<128x32xf16, "cpu">
// CHECK-NEXT:    byre.compute @ConvertOp(%alloc_6, %arg3) {device = "cpu", memory_effects = [1 : i32, 2 : i32]} : memref<128x32xf16, "cpu">, memref<128x32xf32, "cpu">
// CHECK-NEXT:    return
// CHECK-NEXT:  }

// CHECK-LABEL: func.func private @mlp_device(memref<128x64xf16, "test">, memref<64x32xf16, "test">, memref<32xf16, "test">, memref<1x97xf32, "test">) -> memref<128x32xf16, "test"> attributes {device = "test"}

// CHECK-LABEL: func.func private @func_with_bufferization() -> memref<1x97xf32, "cpu"> attributes {device = "cpu"} {
// CHECK-NEXT:    %0 = mhlo.constant {device = "cpu"} dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:    %1 = mhlo.constant {device = "cpu"} dense<1.000000e+00> : tensor<f32>
// CHECK-NEXT:    %2 = mhlo.constant {device = "cpu"} dense<[1, 97]> : tensor<2xi64>
// CHECK-NEXT:    %3 = "mhlo.rng"(%0, %1, %2) <{rng_distribution = #mhlo.rng_distribution<UNIFORM>}> {device = "host"} : (tensor<f32>, tensor<f32>, tensor<2xi64>) -> tensor<1x97xf32>
// CHECK-NEXT:    %4 = bufferization.to_memref %3 {device = "cpu"} : memref<1x97xf32, "cpu">
// CHECK-NEXT:    return %4 : memref<1x97xf32, "cpu">
// CHECK-NEXT:  }
