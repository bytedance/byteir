// RUN: byteir-opt %s -fuse-matmul-epilogue | FileCheck %s

func.func @dot_element_epilog(%arg0 : tensor<4x4xf32>, %arg1 : tensor<4x4xf32>, %arg2 : tensor<4x4xf32>, %arg3 : tensor<4x4xf32>) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
  %0 = "mhlo.dot"(%arg0, %arg1) {precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  %1 = "mhlo.add"(%arg2, %0) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  %2 = "mhlo.abs"(%1) : (tensor<4x4xf32>) -> tensor<4x4xf32>
  %3 = "mhlo.add"(%arg3, %2) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  %4 = "mhlo.dot"(%3, %arg1) {precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  %5 = "mhlo.add"(%3, %4) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  return %3, %5 : tensor<4x4xf32>, tensor<4x4xf32>
}
// CHECK-LABEL: func.func @dot_element_epilog
// CHECK-NEXT:  mhlo.fusion
// CHECK-NEXT:    mhlo.dot
// CHECK-NEXT:    mhlo.add
// CHECK-NEXT:    mhlo.abs
// CHECK-NEXT:    mhlo.add
// CHECK-NEXT:    mhlo.return
// CHECK: {__byteir_matmul_epilogue_fusion__}
// CHECK:       mhlo.fusion
// CHECK-NEXT:    mhlo.dot
// CHECK-NEXT:    mhlo.add
// CHECK-NEXT:    mhlo.return
// CHECK: {__byteir_matmul_epilogue_fusion__}
// CHECK:  return

func.func @dot_element_epilog_with_previous(%arg0 : tensor<4x4xf32>, %arg1 : tensor<4x4xf32>, %arg2 : tensor<4x4xf32>, %arg3 : tensor<4x4xf32>) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
  %0 = "mhlo.add"(%arg0, %arg1) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  %1 = "mhlo.dot"(%arg2, %arg3) {precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  %2 = "mhlo.add"(%0, %1) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  return %0, %2 : tensor<4x4xf32>, tensor<4x4xf32>
}
// CHECK-LABEL: func.func @dot_element_epilog_with_previous
// CHECK-NEXT:  mhlo.add
// CHECK-NEXT:  mhlo.fusion
// CHECK-NEXT:    mhlo.dot
// CHECK-NEXT:    mhlo.add
// CHECK-NEXT:    mhlo.return
// CHECK: {__byteir_matmul_epilogue_fusion__}
// CHECK:  return

func.func @dot_element_epilog_with_next(%arg0 : tensor<4x4xf32>, %arg1 : tensor<4x4xf32>, %arg2 : tensor<4x4xf32>, %arg3 : tensor<4x4xf32>) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
  %0 = "mhlo.dot"(%arg2, %arg3) {precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  %1 = "mhlo.add"(%arg0, %arg1) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  %2 = "mhlo.add"(%1, %0) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  return %1, %2 : tensor<4x4xf32>, tensor<4x4xf32>
}
// CHECK-LABEL: func.func @dot_element_epilog_with_next
// CHECK-NEXT:  mhlo.add
// CHECK-NEXT:  mhlo.fusion
// CHECK-NEXT:    mhlo.dot
// CHECK-NEXT:    mhlo.add
// CHECK-NEXT:    mhlo.return
// CHECK: {__byteir_matmul_epilogue_fusion__}
// CHECK:  return

func.func @dot_element_prolog(%arg0 : tensor<4x4xf32>, %arg1 : tensor<4x4xf32>, %arg2 : tensor<4x4xf32>, %arg3 : tensor<4x4xf32>) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
  %0 = "mhlo.add"(%arg0, %arg1) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  %1 = "mhlo.dot"(%0, %arg2) {precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  return %0, %1 : tensor<4x4xf32>, tensor<4x4xf32>
}
// CHECK-LABEL: func.func @dot_element_prolog
// CHECK-NEXT:  mhlo.add
// CHECK-NEXT:  mhlo.fusion
// CHECK-NEXT:    mhlo.dot
// CHECK-NEXT:    mhlo.return
// CHECK: {__byteir_matmul_epilogue_fusion__}
// CHECK-NEXT:  return
