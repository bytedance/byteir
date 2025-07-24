// RUN: byteir-opt %s -canonicalize-ext | FileCheck %s

func.func private @fold_reduce_add_f() -> tensor<16xf32> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = mhlo.constant dense<5.000000e+00> : tensor<1024x16x512xf32>
    %2 = mhlo.reduce(%1 init: %0) applies mhlo.add across dimensions = [0, 2] : (tensor<1024x16x512xf32>, tensor<f32>) -> tensor<16xf32>
    return %2 : tensor<16xf32>
}
// CHECK-LABEL: fold_reduce_add_f
// CHECK: mhlo.constant dense<2.621440e+06>
// CHECK-NOT: mhlo.reduce

func.func private @fold_reduce_add_d() -> tensor<16xf64> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f64>
    %1 = mhlo.constant dense<5.000000e+00> : tensor<1024x16x512xf64>
    %2 = mhlo.reduce(%1 init: %0) applies mhlo.add across dimensions = [0, 2] : (tensor<1024x16x512xf64>, tensor<f64>) -> tensor<16xf64>
    return %2 : tensor<16xf64>
}
// CHECK-LABEL: fold_reduce_add_d
// CHECK: mhlo.constant dense<2.621440e+06>
// CHECK-NOT: mhlo.reduce

func.func private @fold_reduce_add_i() -> tensor<16xi32> {
    %0 = mhlo.constant dense<0> : tensor<i32>
    %1 = mhlo.constant dense<5> : tensor<16x1024xi32>
    %2 = mhlo.reduce(%1 init: %0) applies mhlo.add across dimensions = [1] : (tensor<16x1024xi32>, tensor<i32>) -> tensor<16xi32>
    return %2 : tensor<16xi32>
}
// CHECK-LABEL: fold_reduce_add_i
// CHECK: mhlo.constant dense<5120>
// CHECK-NOT: mhlo.reduce

func.func private @fold_reduce_mul_f() -> tensor<16xf32> {
    %0 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %1 = mhlo.constant dense<2.000000e+00> : tensor<16x2x4xf32>
    %2 = mhlo.reduce(%1 init: %0) applies mhlo.multiply across dimensions = [1, 2] : (tensor<16x2x4xf32>, tensor<f32>) -> tensor<16xf32>
    return %2 : tensor<16xf32>
}
// CHECK-LABEL: fold_reduce_mul_f
// CHECK: mhlo.constant dense<2.560000e+02>
// CHECK-NOT: mhlo.reduce

func.func private @fold_reduce_mul_i() -> tensor<16xi32> {
    %0 = mhlo.constant dense<1> : tensor<i32>
    %1 = mhlo.constant dense<2> : tensor<16x16xi32>
    %2 = mhlo.reduce(%1 init: %0) applies mhlo.multiply across dimensions = [0] : (tensor<16x16xi32>, tensor<i32>) -> tensor<16xi32>
    return %2 : tensor<16xi32>
}
// CHECK-LABEL: fold_reduce_mul_i
// CHECK: mhlo.constant dense<65536>
// CHECK-NOT: mhlo.reduce

func.func private @fold_reduce_min_f() -> tensor<16xf32> {
    %0 = mhlo.constant dense<0x7F800000> : tensor<f32>
    %1 = mhlo.constant dense<5.000000e+00> : tensor<1024x16xf32>
    %2 = mhlo.reduce(%1 init: %0) applies mhlo.minimum across dimensions = [0] : (tensor<1024x16xf32>, tensor<f32>) -> tensor<16xf32>
    return %2 : tensor<16xf32>
}
// CHECK-LABEL: fold_reduce_min_f
// CHECK: mhlo.constant dense<5.000000e+00>
// CHECK-NOT: mhlo.reduce

func.func private @fold_reduce_max_i() -> tensor<16xi32> {
    %0 = mhlo.constant dense<-2147483648> : tensor<i32>
    %1 = mhlo.constant dense<5> : tensor<1024x512x16xi32>
    %2 = mhlo.reduce(%1 init: %0) applies mhlo.maximum across dimensions = [0, 1] : (tensor<1024x512x16xi32>, tensor<i32>) -> tensor<16xi32>
    return %2 : tensor<16xi32>
}
// CHECK-LABEL: fold_reduce_max_i
// CHECK: mhlo.constant dense<5>
// CHECK-NOT: mhlo.reduce