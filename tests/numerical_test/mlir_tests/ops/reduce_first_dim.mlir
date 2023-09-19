func.func @reduce_sum(%arg0 : tensor<256x2304xf32>) -> tensor<2304xf32> {
    %cst = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = mhlo.reduce(%arg0 init: %cst) across dimensions = [0] : (tensor<256x2304xf32>, tensor<f32>) -> tensor<2304xf32>
     reducer(%arg1: tensor<f32>, %arg2: tensor<f32>)  {
      %1 = mhlo.add %arg1, %arg2 : tensor<f32>
      mhlo.return %1 : tensor<f32>
    }
    return %0 : tensor<2304xf32>
}
