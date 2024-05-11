func.func @compare_LT_i32(%arg0 : tensor<256x1xi32>, %arg1 : tensor<256x1xi32>) -> tensor<256x1xi1> { 
  %0 = "stablehlo.compare"(%arg0, %arg1) {
    comparison_direction = #stablehlo<comparison_direction NE>,
    compare_type = #stablehlo<comparison_type SIGNED>
  } : (tensor<256x1xi32>, tensor<256x1xi32>) -> tensor<256x1xi1>
  func.return %0 : tensor<256x1xi1>
}
