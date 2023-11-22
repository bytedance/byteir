module {
  func.func @contain_string(%arg0: tensor<1x1x!tf_type.string>, %arg1: tensor<1x10xf32>, %arg2: tensor<1x10xf32>) -> tensor<1x10xf32> {
    %0 = call @contain_string_host(%arg0, %arg1, %arg2) : (tensor<1x1x!tf_type.string>, tensor<1x10xf32>, tensor<1x10xf32>) -> tensor<1x10xf32>
    return %0 : tensor<1x10xf32>
  }
  func.func @contain_string_host(%arg0: tensor<1x1x!tf_type.string>, %arg1: tensor<1x10xf32>, %arg2: tensor<1x10xf32>) -> tensor<1x10xf32> attributes {__byteir_host_device__, device = "host"} {
    %0 = "tf.Const"() {device = "host", value = dense<"string_to_compare"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
    %1 = "tf.Equal"(%arg0, %0) {device = "host", incompatible_shape_error = true} : (tensor<1x1x!tf_type.string>, tensor<!tf_type.string>) -> tensor<1x1xi1>
    %2 = mhlo.reshape %1 : (tensor<1x1xi1>) -> tensor<i1>
    %3 = mhlo.select %2, %arg1, %arg2 : tensor<i1>, tensor<1x10xf32>
    %4 = mhlo.custom_call @pytorch.add_hbm(%3, %arg1) {backend_config = "", byteir_attrs = {}, device = "hbmpim"} : (tensor<1x10xf32>, tensor<1x10xf32>) -> tensor<1x10xf32>
    return %4 : tensor<1x10xf32>
  }
}

