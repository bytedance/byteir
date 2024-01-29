// RUN: %python -m byteir.tools.cat_numerical_test --before-pass-file %s --backend=ait | FileCheck %s

func.func @concat(%arg0 : tensor<8x256x768xf32>, %arg1 : tensor<8x256x768xf32>, %arg2 : tensor<8x256x768xf32>) -> tensor<8x256x2304xf32> attributes {__byteir_cat_fusion__} {
  %0 = "mhlo.concatenate"(%arg0, %arg1, %arg2) {dimension = 2 : i64} : (tensor<8x256x768xf32>, tensor<8x256x768xf32>, tensor<8x256x768xf32>) -> tensor<8x256x2304xf32>
  return %0 : tensor<8x256x2304xf32>
}

// CHECK: numerical test pass

