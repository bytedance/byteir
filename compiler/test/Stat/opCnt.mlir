// RUN: byteir-stat -op-cnt %s | FileCheck %s -check-prefix=DEFAULT
// RUN: byteir-stat -op-cnt -func-name="tf_add" %s | FileCheck %s -check-prefix=FUNCNAME
// RUN: byteir-stat -op-cnt -top-only %s | FileCheck %s -check-prefix=TOPONLY
// RUN: byteir-stat -op-cnt -func-name="tf_add" -top-only %s | FileCheck %s -check-prefix=FUNCNAMETOPONLY

module {
  func.func private @some_callee(%arg0 : tensor<2x4xf32>, %arg1 : tensor<2x4xf32>) -> (tensor<2x4xf32>) {
    %0 = "tf.Add"(%arg0, %arg1) : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    %1 = "tf.Mul"(%0, %arg0) : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    %2 = "mhlo.add"(%1, %arg0) : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    %3 = "mhlo.add"(%1, %2) : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32> 
    return %3 : tensor<2x4xf32>
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
// DEFAULT: func.call 1
// DEFAULT: func.func 2
// DEFAULT: func.return 2
// DEFAULT: mhlo.add 6
// DEFAULT: mhlo.fusion 1
// DEFAULT: mhlo.return 1
// DEFAULT: tf.Add 2
// DEFAULT: tf.Mul 2

// FUNCNAME: func.call 1
// FUNCNAME: func.func 1
// FUNCNAME: func.return 1
// FUNCNAME: mhlo.add 4
// FUNCNAME: mhlo.fusion 1
// FUNCNAME: mhlo.return 1
// FUNCNAME: tf.Add 1
// FUNCNAME: tf.Mul 1

// TOPONLY: func.call 1
// TOPONLY: func.return 2
// TOPONLY: mhlo.add 4
// TOPONLY: mhlo.fusion 1
// TOPONLY: tf.Add 2
// TOPONLY: tf.Mul 2

// FUNCNAMETOPONLY: func.call 1
// FUNCNAMETOPONLY: func.return 1
// FUNCNAMETOPONLY: mhlo.add 2
// FUNCNAMETOPONLY: mhlo.fusion 1
// FUNCNAMETOPONLY: tf.Add 1
// FUNCNAMETOPONLY: tf.Mul 1
