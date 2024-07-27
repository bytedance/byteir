func.func @bmm_crr(%arg0 : tensor<1x256x4096xf32>, %arg1 : tensor<256x11008xf32>) -> tensor<4096x11008xf32> {
    %0 = mhlo.reshape %arg0 : (tensor<1x256x4096xf32>) -> tensor<256x4096xf32>
    %1 = "mhlo.transpose"(%0) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<256x4096xf32>) -> tensor<4096x256xf32>
    %2 = "mhlo.dot"(%1, %arg1) : (tensor<4096x256xf32>, tensor<256x11008xf32>) -> tensor<4096x11008xf32>
    return %2: tensor<4096x11008xf32>
}

