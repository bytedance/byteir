func.func @compare_LT_f64(%arg0 : tensor<256x1xf64>, %arg1 : tensor<256x1xf64>) -> tensor<256x1xi1> { 
  %0 = "stablehlo.compare"(%arg0, %arg1) {
    comparison_direction = #stablehlo<comparison_direction NE>,
    compare_type = #stablehlo<comparison_type FLOAT>
  } : (tensor<256x1xf64>, tensor<256x1xf64>) -> tensor<256x1xi1>
  func.return %0 : tensor<256x1xi1>
}
