func.func @insert_slice(%arg0 : tensor<1x32x256x64xf16>, %arg1 : tensor<1x32x256x64xf16>) -> tensor<1x32x256x128xf16> {
    %cst = mhlo.constant dense<0.000000e+00> : tensor<1x32x256x128xf16>
    %inserted_slice_0 = tensor.insert_slice %arg0 into %cst[0, 0, 0, 64] [1, 32, 256, 64] [1, 1, 1, 1] : tensor<1x32x256x64xf16> into tensor<1x32x256x128xf16>
    %inserted_slice_1 = tensor.insert_slice %arg1 into %inserted_slice_0[0, 0, 0, 0] [1, 32, 256, 64] [1, 1, 1, 1] : tensor<1x32x256x64xf16> into tensor<1x32x256x128xf16>
    return %inserted_slice_1 : tensor<1x32x256x128xf16>
}
