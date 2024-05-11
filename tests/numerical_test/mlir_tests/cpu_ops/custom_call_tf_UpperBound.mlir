func.func @custom_call_tf_UpperBound(%arg0 : tensor<1x2560xf16>) -> tensor<1x2560xi32> { 
  %0 = stablehlo.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 6.000000e+00, 1.000000e+01, 2.000000e+01, 5.000000e+01]]> : tensor<1x8xf16>
  %1 = "stablehlo.custom_call"(%0, %arg0) {
    call_target_name = "tf.UpperBound",
    has_side_effect = false,
    backend_config = "",
    byteir_attrs = {},
    api_version = 1 : i32,
    called_computations = [@tf.UpperBound]
  } : (tensor<1x8xf16>, tensor<1x2560xf16>) -> tensor<1x2560xi32>
  func.return %1 : tensor<1x2560xi32>
}
