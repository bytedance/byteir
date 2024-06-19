// RUN: %python -m byteir.tools.cat_executor %s --mode=profile --backend=ait | FileCheck %s

func.func @main(%arg0 : tensor<128x64xf16>, %arg1 : tensor<64x32xf16>) -> tensor<128x32xf16> attributes {__byteir_cat_fusion__} {
  %0 = "mhlo.dot"(%arg0, %arg1) : (tensor<128x64xf16>, tensor<64x32xf16>) -> tensor<128x32xf16>
  return %0 : tensor<128x32xf16>
}

// CHECK: cat ait profile finish
