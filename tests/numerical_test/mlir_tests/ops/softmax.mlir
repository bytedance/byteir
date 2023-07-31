func.func @softmax(%arg0 : tensor<8x12x256x256xf32>) -> tensor<8x12x256x256xf32> {
    %0 = mhlo.custom_call @byteir.softmax(%arg0) {backend_config = "", byteir_attrs = {axis = 3 : i64}} : (tensor<8x12x256x256xf32>) -> tensor<8x12x256x256xf32>
    return %0 : tensor<8x12x256x256xf32>
}
