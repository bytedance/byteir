// RUN: byteir-opt %s -allow-unregistered-dialect -set-arg-shape="dim=0 size=3 entry-func-name=tf_add arg-attr-name=__placeholder__byre.argname" | FileCheck %s

func.func @tf_add(%arg0 : tensor<?x4xf32> {__placeholder__byre.argname = "A"}, %arg1 : tensor<?x4xf32> {__placeholder__byre.argname = "B"}) -> (tensor<*xf32> {__placeholder__byre.argname = "C"}) attributes { __placeholder__byre.entry_point} {
    %res = "tf.Add"(%arg0, %arg1) : (tensor<?x4xf32>, tensor<?x4xf32>) -> tensor<*xf32>
    return %res : tensor<*xf32>
}
// CHECK-LABEL: func.func @tf_add
// CHECK-NEXT: %[[RES0:.*]] = "tf.Add"(%arg0, %arg1) : (tensor<3x4xf32>, tensor<3x4xf32>) -> tensor<*xf32>
