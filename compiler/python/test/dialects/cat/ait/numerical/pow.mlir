// RUN: %python -m byteir.tools.cat_executor %s --backend=ait | FileCheck %s

func.func @pow(%arg0 : tensor<1x16x128xf32>, %arg1 : tensor<1x16x128xf32>) -> tensor<1x16x128xf32> attributes {__byteir_cat_fusion__} {
  %0 = mhlo.power %arg0, %arg1 : tensor<1x16x128xf32>
  return %0 : tensor<1x16x128xf32>
}

// CHECK: cat ait numerical test pass
