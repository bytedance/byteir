// RUN: %python -m byteir.tools.cat_executor %s --backend=ait | FileCheck %s

func.func @permute10(%arg0 : tensor<128x64xf32>) -> tensor<64x128xf32> attributes {__byteir_cat_fusion__} {
  %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<128x64xf32>) -> tensor<64x128xf32>
  return %0 : tensor<64x128xf32>
}

// CHECK: cat ait numerical test pass
