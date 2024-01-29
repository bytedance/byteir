// RUN: %python -m byteir.tools.cat_numerical_test --before-pass-file %s --backend=ait | FileCheck %s

func.func @permute021(%arg0 : tensor<32x16x128xf32>) -> tensor<32x128x16xf32> attributes {__byteir_cat_fusion__} {
  %0 = "mhlo.transpose"(%arg0) {permutation = dense<[0, 2, 1]> : tensor<3xi64>} : (tensor<32x16x128xf32>) -> tensor<32x128x16xf32>
  return %0 : tensor<32x128x16xf32>
}

// CHECK: numerical test pass
