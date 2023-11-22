module {
  func.func @contain_string(%arg0: tensor<1x1x!tf_type.string>, %arg1: tensor<1x10xf32>, %arg2: tensor<1x10xf32>) -> tensor<1x10xf32> {
    %cst = "tf.Const"() {device = "host", value = dense<"string_to_compare"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
    %0 = "tf.Equal"(%arg0, %cst) {device = "host", incompatible_shape_error = true} : (tensor<1x1x!tf_type.string>, tensor<!tf_type.string>) -> tensor<1x1xi1>
    %1 = "mhlo.reshape"(%0) : (tensor<1x1xi1>) -> tensor<i1>
    %2 = "mhlo.select"(%1, %arg1, %arg2) : (tensor<i1>, tensor<1x10xf32>, tensor<1x10xf32>) -> tensor<1x10xf32>
    %3 = mhlo.custom_call @pytorch.add_hbm(%2, %arg1) {backend_config = "", byteir_attrs = {}, device = "hbmpim"} : (tensor<1x10xf32>,tensor<1x10xf32>) -> tensor<1x10xf32>
    return %3 : tensor<1x10xf32>
  }
}

