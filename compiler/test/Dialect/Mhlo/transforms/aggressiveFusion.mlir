// RUN: byteir-opt %s -hlo-aggressive-fusion | FileCheck %s
// RUN: byteir-opt %s -hlo-aggressive-fusion="disable-fusion" | FileCheck %s --check-prefix CHECK-NOFUSION

func.func @mhlo_aggressive_fusion(%arg0 : tensor<32x32xf32>, %arg1 : tensor<32xi64>, %arg2 : tensor<32x32xf32>) -> tensor<32x32xf32> {
  %0 = "mhlo.add"(%arg0, %arg0) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  %1 = "mhlo.torch_index_select"(%0, %arg1) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<32x32xf32>, tensor<32xi64>) -> tensor<32x32xf32>
  %2 = "mhlo.add"(%1, %arg2) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  return %2 : tensor<32x32xf32>
}
// CHECK-LABEL: func.func @mhlo_aggressive_fusion
// CHECK-NEXT:  mhlo.fusion
// CHECK-NEXT:    mhlo.add
// CHECK-NEXT:    mhlo.torch_index_select
// CHECK-NEXT:    mhlo.add
// CHECK-NEXT:    mhlo.return
// CHECK: {__byteir_hlo_aggressive_fusion__}
// CHECK:  return

// CHECK-NOFUSION-LABEL: func.func @mhlo_aggressive_fusion
// CHECK-NOFUSION-NEXT:  mhlo.fusion
// CHECK-NOFUSION-NEXT:    mhlo.add
// CHECK-NOFUSION-NEXT:    mhlo.return
// CHECK-NOFUSION-NEXT:  {__byteir_hlo_aggressive_fusion__}
// CHECK-NOFUSION-NEXT:  mhlo.fusion
// CHECK-NOFUSION-NEXT:    mhlo.torch_index_select
// CHECK-NOFUSION-NEXT:    mhlo.return
// CHECK-NOFUSION-NEXT:  {__byteir_hlo_aggressive_fusion__}
// CHECK-NOFUSION-NEXT:  mhlo.fusion
// CHECK-NOFUSION-NEXT:    mhlo.add
// CHECK-NOFUSION-NEXT:    mhlo.return
// CHECK-NOFUSION-NEXT:  {__byteir_hlo_aggressive_fusion__}
// CHECK-NOFUSION-NEXT:  return

func.func @reshape_add(%arg0: tensor<2xf32>, %arg1: tensor<2x1xf32>) -> (tensor<2x1xf32>) {
  %0 = mhlo.reshape %arg0 : (tensor<2xf32>) -> tensor<2x1xf32>
  %1 = mhlo.add %0, %arg1 : tensor<2x1xf32>
  return %1 : tensor<2x1xf32>
}
// CHECK-LABEL: func.func @reshape_add
// CHECK-NEXT:  mhlo.reshape
// CHECK-NEXT:  mhlo.fusion
// CHECK-NEXT:    mhlo.add
// CHECK-NEXT:    mhlo.return
// CHECK: {__byteir_hlo_aggressive_fusion__}
// CHECK:  return

// CHECK-NOFUSION-LABEL: func.func @reshape_add
// CHECK-NOFUSION-NEXT:  mhlo.fusion
// CHECK-NOFUSION-NEXT:    mhlo.reshape
// CHECK-NOFUSION-NEXT:    mhlo.add
// CHECK-NOFUSION-NEXT:    mhlo.return
// CHECK-NOFUSION-NEXT: {__byteir_hlo_aggressive_fusion__}
// CHECK-NOFUSION-NEXT:  return

func.func @single_reshape(%arg0: tensor<2xf32>) -> tensor<2x1xf32> {
  %0 = mhlo.reshape %arg0 : (tensor<2xf32>) -> tensor<2x1xf32>
  return %0 : tensor<2x1xf32>
}
// CHECK-LABEL: func.func @single_reshape
// CHECK-NOT:  mhlo.fusion
// CHECK:  mhlo.reshape
// CHECK:  return
