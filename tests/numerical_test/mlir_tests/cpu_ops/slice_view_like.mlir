func.func @main(%arg0: tensor<512x200xf32>, %arg1: tensor<512x200xf32>) -> (tensor<128x2x100xf32>, tensor<128x2x100xf32>, tensor<1x100xf32>, tensor<1x100xf32>, tensor<512x200xf32>) {
    %0 = stablehlo.slice %arg0 [0:128, 0:200] : (tensor<512x200xf32>) -> tensor<128x200xf32>
    %1 = stablehlo.slice %arg1 [10:138, 0:200] : (tensor<512x200xf32>) -> tensor<128x200xf32>
    %2 = stablehlo.reshape %0 : (tensor<128x200xf32>) -> tensor<128x2x100xf32>
    %3 = stablehlo.reshape %1 : (tensor<128x200xf32>) -> tensor<128x2x100xf32>
    %4 = stablehlo.slice %arg0 [0:1, 0:100] : (tensor<512x200xf32>) -> tensor<1x100xf32>
    %5 = stablehlo.slice %arg1 [10:11, 100:200] : (tensor<512x200xf32>) -> tensor<1x100xf32>
    %6 = stablehlo.add %arg0, %arg1 : tensor<512x200xf32>
    return %2, %3, %4, %5, %6 : tensor<128x2x100xf32>, tensor<128x2x100xf32>, tensor<1x100xf32>, tensor<1x100xf32>, tensor<512x200xf32>
}
