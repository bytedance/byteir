// RUN: byteir-opt %s -hlo-fusion-to-linalg="target="cpu" arch="x86_64"" | FileCheck %s

func.func @mhlo_convert_f32_i32(%arg0: tensor<2x3xf32>) -> tensor<2x3xi32> {
    %0 = mhlo.convert %arg0 : (tensor<2x3xf32>) -> tensor<2x3xi32>
    return %0 : tensor<2x3xi32>
}
// CHECK-LABEL: mhlo_convert_f32_i32
// CHECK: linalg.map
// CHECK: arith.cmpf
// CHECK: arith.fptosi
// CHECK: arith.select
