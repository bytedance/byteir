// RUN: byteir-opt %s -set-op-space="entry-func=main space=cpu" -set-arg-space="entry-func=main all-space=cpu" | FileCheck %s

module {
  func.func @main(%arg0: memref<128x64xf32>, %arg1: memref<64x32xf32>, %arg2: memref<32xf32>) -> memref<128x32xf32> {
    %0 = memref.alloc() : memref<128x64xf16>
    "lmhlo.convert"(%arg0, %0) : (memref<128x64xf32>, memref<128x64xf16>) -> ()
    %1 = memref.alloc() : memref<64x32xf16>
    "lmhlo.convert"(%arg1, %1) : (memref<64x32xf32>, memref<64x32xf16>) -> ()
    %2 = memref.alloc() : memref<32xf16>
    "lmhlo.convert"(%arg2, %2) : (memref<32xf32>, memref<32xf16>) -> ()
    %3 = call @func_with_bufferization() : () -> memref<1x97xf32>
    %4 = call @mlp_device(%0, %1, %2, %3) : (memref<128x64xf16>, memref<64x32xf16>, memref<32xf16>, memref<1x97xf32>) -> memref<128x32xf16>
    %5 = memref.alloc() : memref<128x32xf32>
    "lmhlo.convert"(%4, %5) : (memref<128x32xf16>, memref<128x32xf32>) -> ()
    return %5 : memref<128x32xf32>
  }
  func.func private @mlp_device(memref<128x64xf16>, memref<64x32xf16>, memref<32xf16>, memref<1x97xf32>) -> memref<128x32xf16> attributes {device = "test"}

  func.func private @func_with_bufferization() -> memref<1x97xf32> attributes {device = "cpu"} {
    %0 = memref.alloc() : memref<f32>
    "lmhlo.constant"(%0) {device = "cpu", value = dense<0.000000e+00> : tensor<f32>} : (memref<f32>) -> ()
    %1 = bufferization.to_tensor %0 {device = "cpu"} : memref<f32>
    %2 = memref.alloc() : memref<f32>
    "lmhlo.constant"(%2) {device = "cpu", value = dense<1.000000e+00> : tensor<f32>} : (memref<f32>) -> ()
    %3 = bufferization.to_tensor %2 {device = "cpu"} : memref<f32>
    %4 = memref.alloc() : memref<2xi64>
    "lmhlo.constant"(%4) {device = "cpu", value = dense<[1, 97]> : tensor<2xi64>} : (memref<2xi64>) -> ()
    %5 = bufferization.to_tensor %4 {device = "cpu"} : memref<2xi64>
    %6 = "mhlo.rng"(%1, %3, %5) {device = "host", rng_distribution = #mhlo.rng_distribution<UNIFORM>} : (tensor<f32>, tensor<f32>, tensor<2xi64>) -> tensor<1x97xf32>
    %7 = bufferization.to_memref %6 {device = "cpu"} : memref<1x97xf32>
    return %7 : memref<1x97xf32>
  }
}
// CHECK-LABEL: func.func @main(%arg0: memref<128x64xf32, "cpu">, %arg1: memref<64x32xf32, "cpu">, %arg2: memref<32xf32, "cpu">) -> memref<128x32xf32, "cpu"> {
// CHECK-NEXT:    %alloc = memref.alloc() : memref<128x64xf16, "test">
// CHECK-NEXT:    %alloc_0 = memref.alloc() : memref<128x64xf16, "cpu">
// CHECK-NEXT:    "lmhlo.convert"(%arg0, %alloc_0) {device = "cpu"} : (memref<128x64xf32, "cpu">, memref<128x64xf16, "cpu">) -> ()
// CHECK-NEXT:    memref.copy %alloc_0, %alloc : memref<128x64xf16, "cpu"> to memref<128x64xf16, "test">
// CHECK-NEXT:    %alloc_1 = memref.alloc() : memref<64x32xf16, "test">
// CHECK-NEXT:    %alloc_2 = memref.alloc() : memref<64x32xf16, "cpu">
// CHECK-NEXT:    "lmhlo.convert"(%arg1, %alloc_2) {device = "cpu"} : (memref<64x32xf32, "cpu">, memref<64x32xf16, "cpu">) -> ()
// CHECK-NEXT:    memref.copy %alloc_2, %alloc_1 : memref<64x32xf16, "cpu"> to memref<64x32xf16, "test">
// CHECK-NEXT:    %alloc_3 = memref.alloc() : memref<32xf16, "test">
// CHECK-NEXT:    %alloc_4 = memref.alloc() : memref<32xf16, "cpu">
// CHECK-NEXT:    "lmhlo.convert"(%arg2, %alloc_4) {device = "cpu"} : (memref<32xf32, "cpu">, memref<32xf16, "cpu">) -> ()
// CHECK-NEXT:    memref.copy %alloc_4, %alloc_3 : memref<32xf16, "cpu"> to memref<32xf16, "test">
// CHECK-NEXT:    %0 = call @func_with_bufferization() : () -> memref<1x97xf32, "cpu">
// CHECK-NEXT:    %alloc_5 = memref.alloc() : memref<1x97xf32, "test">
// CHECK-NEXT:    memref.copy %0, %alloc_5 : memref<1x97xf32, "cpu"> to memref<1x97xf32, "test">
// CHECK-NEXT:    %1 = call @mlp_device(%alloc, %alloc_1, %alloc_3, %alloc_5) : (memref<128x64xf16, "test">, memref<64x32xf16, "test">, memref<32xf16, "test">, memref<1x97xf32, "test">) -> memref<128x32xf16, "test">
// CHECK-NEXT:    %alloc_6 = memref.alloc() : memref<128x32xf32, "cpu">
// CHECK-NEXT:    %alloc_7 = memref.alloc() : memref<128x32xf16, "cpu">
// CHECK-NEXT:    memref.copy %1, %alloc_7 : memref<128x32xf16, "test"> to memref<128x32xf16, "cpu">
// CHECK-NEXT:    "lmhlo.convert"(%alloc_7, %alloc_6) {device = "cpu"} : (memref<128x32xf16, "cpu">, memref<128x32xf32, "cpu">) -> ()
// CHECK-NEXT:    return %alloc_6 : memref<128x32xf32, "cpu">
// CHECK-NEXT:  }

// CHECK-LABEL: func.func private @mlp_device(memref<128x64xf16, "test">, memref<64x32xf16, "test">, memref<32xf16, "test">, memref<1x97xf32, "test">) -> memref<128x32xf16, "test"> attributes {device = "test"}

// CHECK-LABEL: func.func private @func_with_bufferization() -> memref<1x97xf32, "cpu"> attributes {device = "cpu"} {
// CHECK-NEXT:    %alloc = memref.alloc() : memref<f32, "cpu">
// CHECK-NEXT:    "lmhlo.constant"(%alloc) <{value = dense<0.000000e+00> : tensor<f32>}> {device = "cpu"} : (memref<f32, "cpu">) -> ()
// CHECK-NEXT:    %0 = bufferization.to_tensor %alloc {device = "cpu"} : memref<f32, "cpu">
// CHECK-NEXT:    %alloc_0 = memref.alloc() : memref<f32, "cpu">
// CHECK-NEXT:    "lmhlo.constant"(%alloc_0) <{value = dense<1.000000e+00> : tensor<f32>}> {device = "cpu"} : (memref<f32, "cpu">) -> ()
// CHECK-NEXT:    %1 = bufferization.to_tensor %alloc_0 {device = "cpu"} : memref<f32, "cpu">
// CHECK-NEXT:    %alloc_1 = memref.alloc() : memref<2xi64, "cpu">
// CHECK-NEXT:    "lmhlo.constant"(%alloc_1) <{value = dense<[1, 97]> : tensor<2xi64>}> {device = "cpu"} : (memref<2xi64, "cpu">) -> ()
// CHECK-NEXT:    %2 = bufferization.to_tensor %alloc_1 {device = "cpu"} : memref<2xi64, "cpu">
// CHECK-NEXT:    %3 = "mhlo.rng"(%0, %1, %2) {device = "host", rng_distribution = #mhlo.rng_distribution<UNIFORM>} : (tensor<f32>, tensor<f32>, tensor<2xi64>) -> tensor<1x97xf32>
// CHECK-NEXT:    %4 = bufferization.to_memref %3 {device = "cpu"} : memref<1x97xf32, "cpu">
// CHECK-NEXT:    return %4 : memref<1x97xf32, "cpu">
// CHECK-NEXT:  }
