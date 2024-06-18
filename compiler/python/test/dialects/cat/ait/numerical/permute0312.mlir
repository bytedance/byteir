// RUN: %python -m byteir.tools.cat_executor %s --backend=ait | FileCheck %s

func.func @permute0312(%arg0 : tensor<1x16x32x128xf32>) -> tensor<1x128x16x32xf32> attributes {__byteir_cat_fusion__} {
  %0 = "mhlo.transpose"(%arg0) {permutation = dense<[0, 3, 1, 2]> : tensor<4xi64>} : (tensor<1x16x32x128xf32>) -> tensor<1x128x16x32xf32>
  return %0 : tensor<1x128x16x32xf32>
}

// CHECK: cat ait numerical test pass
