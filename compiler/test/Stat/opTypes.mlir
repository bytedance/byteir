// RUN: byteir-stat -op-cnt %s | FileCheck %s -check-prefix=DEFAULT
// RUN: byteir-stat -op-cnt -func-name="tf_add" %s | FileCheck %s -check-prefix=FUNCNAME
// RUN: byteir-stat -op-cnt -top-only %s | FileCheck %s -check-prefix=TOPONLY
// RUN: byteir-stat -op-cnt -func-name="tf_add" -top-only %s | FileCheck %s -check-prefix=FUNCNAMETOPONLY

module {
  func.func private @some_callee(%arg0 : tensor<2x4xf32>, %arg1 : tensor<2x4xf32>) -> (tensor<2x4xf32>) {
    %0 = "tf.Add"(%arg0, %arg1) : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    %1 = "tf.Mul"(%0, %arg0) : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    %2 = "mhlo.add"(%1, %arg0) : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    %11 = mhlo.convert %1 : (tensor<2x4xf32>) -> tensor<2x4xf16>
    %21 = mhlo.convert %2 : (tensor<2x4xf32>) -> tensor<2x4xf16>
    %3 = "mhlo.add"(%11, %21) : (tensor<2x4xf16>, tensor<2x4xf16>) -> tensor<2x4xf16>
    %31 = mhlo.custom_call @byteir.softmax(%3) {backend_config = "", byteir_attrs = {axis = 1 : i64}} : (tensor<2x4xf16>) -> tensor<2x4xf32>
    return %31 : tensor<2x4xf32>
  }
  func.func @tf_add(%arg0 : tensor<2x4xf32>, %arg1 : tensor<2x4xf32>) -> (tensor<2x4xf32>) {
    %0 = "tf.Add"(%arg0, %arg1) : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    %1 = "tf.Mul"(%0, %arg0) : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    %2 = "mhlo.add"(%1, %arg0) : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    %3 = "mhlo.add"(%1, %2) : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    %4 = "mhlo.fusion"(%arg0, %arg1) ( {
        %6 = "mhlo.add" (%arg0, %arg1) : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
        %7 = "mhlo.add" (%arg0, %6) : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
        "mhlo.return"(%6) : (tensor<2x4xf32>) -> ()
      }) : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    %5 = call @some_callee(%3, %4) : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    return %5 : tensor<2x4xf32>
  }
}
// DEFAULT: func.call 1 f32 f32
// DEFAULT: func.func 2 NA  NA
// DEFAULT: func.return 2 f32 NA
// DEFAULT: mhlo.add 6 f16,f32 f16,f32
// DEFAULT: mhlo.convert 2 f32 f16
// DEFAULT: mhlo.custom_call 1 f16 f32
// DEFAULT: mhlo.fusion 1 f32 f32
// DEFAULT: mhlo.return 1 f32 NA
// DEFAULT: tf.Add 2 f32 f32
// DEFAULT: tf.Mul 2 f32 f32

// FUNCNAME: func.call 1 f32 f32
// FUNCNAME: func.func 1
// FUNCNAME: func.return 1 f32 NA
// FUNCNAME: mhlo.add 4 f32 f32
// FUNCNAME: mhlo.fusion 1 f32 f32
// FUNCNAME: mhlo.return 1 f32 NA
// FUNCNAME: tf.Add 1 f32 f32
// FUNCNAME: tf.Mul 1 f32 f32

// TOPONLY: func.call 1 f32 f32
// TOPONLY: func.return 2 f32 NA
// TOPONLY: mhlo.add 4 f16,f32 f16,f32
// TOPONLY: mhlo.convert 2 f32 f16
// TOPONLY: mhlo.custom_call 1 f16 f32
// TOPONLY: mhlo.fusion 1 f32 f32
// TOPONLY: tf.Add 2 f32 f32
// TOPONLY: tf.Mul 2 f32 f32

// FUNCNAMETOPONLY: func.call 1 f32 f32
// FUNCNAMETOPONLY: func.return 1 f32 NA
// FUNCNAMETOPONLY: mhlo.add 2 f32 f32
// FUNCNAMETOPONLY: mhlo.fusion 1 f32 f32
// FUNCNAMETOPONLY: tf.Add 1 f32 f32
// FUNCNAMETOPONLY: tf.Mul 1 f32 f32
