// RUN: %python -m byteir.tools.cat_numerical_test --before-pass-file %s --backend=ait | FileCheck %s

func.func @where(%arg0 : tensor<8x12x256x256xi1>, %arg1 : tensor<8x12x256x256xf32>, %arg2 : tensor<8x12x256x256xf32>) -> tensor<8x12x256x256xf32> attributes {__byteir_cat_fusion__} {
  %0 = mhlo.select %arg0, %arg1, %arg2 : tensor<8x12x256x256xi1>, tensor<8x12x256x256xf32>
  return %0 : tensor<8x12x256x256xf32>
}

// CHECK: numerical test pass

