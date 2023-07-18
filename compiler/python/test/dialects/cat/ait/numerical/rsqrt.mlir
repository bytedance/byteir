// RUN: %python -m byteir.dialects.cat.numerical_test --before-pass-file %s --backend=ait | FileCheck %s

func.func @rsqrt(%arg0 : tensor<8x1024xf32>) -> tensor<8x1024xf32> attributes {__byteir_cat_fusion__} {
  %0 = mhlo.rsqrt %arg0 : tensor<8x1024xf32>
  return %0 : tensor<8x1024xf32>
}

// CHECK: numerical test pass

