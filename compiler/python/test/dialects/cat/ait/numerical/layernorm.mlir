// RUN: %python -m byteir.tools.cat_executor %s --backend=ait | FileCheck %s

func.func @layer_norm(%arg0 : tensor<1x16x4096xf32>, %arg1 : tensor<4096xf32>, %arg2 : tensor<4096xf32>) -> tensor<1x16x4096xf32> attributes {__byteir_cat_fusion__} {
  %0:3 = "mhlo.custom_call"(%arg0, %arg1, %arg2) {api_version = 1 : i32, backend_config = "", byteir_attrs = {axis = [2], epsilon = 1.000000e-05 : f64}, call_target_name = "byteir.layer_norm", called_computations = [], has_side_effect = false} : (tensor<1x16x4096xf32>, tensor<4096xf32>, tensor<4096xf32>) -> (tensor<1x16x4096xf32>, tensor<1x16x1xf32>, tensor<1x16x1xf32>)
  return %0#0 : tensor<1x16x4096xf32>
}

// CHECK: cat ait numerical test pass
