// RUN: %python -m byteir.dialects.cat.numerical_test --before-pass-file %s --backend=ait | FileCheck %s

func.func @softmax_f16(%arg0 : tensor<1x12x1024x1024xf16>) -> tensor<1x12x1024x1024xf16> attributes {__byteir_cat_fusion__} {
  %0 = mhlo.custom_call @byteir.softmax(%arg0) {backend_config = "", byteir_attrs = {axis = 3 : i64}} : (tensor<1x12x1024x1024xf16>) -> tensor<1x12x1024x1024xf16>
  return %0 : tensor<1x12x1024x1024xf16>
}

// CHECK: numerical test pass
