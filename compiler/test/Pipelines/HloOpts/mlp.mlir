// RUN: byteir-opt %s -hlo-graph-opt -hlo-fusion-opt="outline-single-elemwise-op" | FileCheck %s

// CHECK: func.func private @Unknown
// CHECK:   mhlo.log
// CHECK: func.func @main
module @IrToMhlo.91 {
  func.func @main(%arg0: tensor<50xf32>, %arg1: tensor<50x784xf32>, %arg2: tensor<50xf32>, %arg3: tensor<50x50xf32>, %arg4: tensor<10xf32>, %arg5: tensor<10x50xf32>, %arg6: tensor<32x1x28x28xf32>) -> tuple<tensor<32x10xf32>, tensor<32x50xf32>, tensor<32x50xf32>> {
    %0 = call @aten.view.20(%arg6) : (tensor<32x1x28x28xf32>) -> tensor<32x784xf32>
    %1 = call @aten.permute.16(%arg1) {xla_shape = "f32[784,50]{0,1}"} : (tensor<50x784xf32>) -> tensor<784x50xf32>
    %2 = call @aten.addmm.24(%0, %1, %arg0) : (tensor<32x784xf32>, tensor<784x50xf32>, tensor<50xf32>) -> tensor<32x50xf32>
    %3 = call @aten.relu.35(%2) : (tensor<32x50xf32>) -> tensor<32x50xf32>
    %4 = call @aten.permute.12(%arg3) {xla_shape = "f32[50,50]{0,1}"} : (tensor<50x50xf32>) -> tensor<50x50xf32>
    %5 = call @aten.addmm.41(%3, %4, %arg2) : (tensor<32x50xf32>, tensor<50x50xf32>, tensor<50xf32>) -> tensor<32x50xf32>
    %6 = call @aten.relu.52(%5) : (tensor<32x50xf32>) -> tensor<32x50xf32>
    %7 = call @aten.permute.8(%arg5) {xla_shape = "f32[50,10]{0,1}"} : (tensor<10x50xf32>) -> tensor<50x10xf32>
    %8 = call @aten.addmm.58(%6, %7, %arg4) : (tensor<32x50xf32>, tensor<50x10xf32>, tensor<10xf32>) -> tensor<32x10xf32>
    %9 = call @aten.log_softmax.77(%8) : (tensor<32x10xf32>) -> tensor<32x10xf32>
    %10 = "mhlo.tuple"(%9, %3, %6) {xla_shape = "(f32[32,10]{1,0}, f32[32,50]{1,0}, f32[32,50]{1,0})"} : (tensor<32x10xf32>, tensor<32x50xf32>, tensor<32x50xf32>) -> tuple<tensor<32x10xf32>, tensor<32x50xf32>, tensor<32x50xf32>>
    return %10 : tuple<tensor<32x10xf32>, tensor<32x50xf32>, tensor<32x50xf32>>
  }
  func.func private @aten.view.20(%arg0: tensor<32x1x28x28xf32>) -> tensor<32x784xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<32x1x28x28xf32>) -> tensor<32x784xf32>
    return %0 : tensor<32x784xf32>
  }
  func.func private @aten.permute.16(%arg0: tensor<50x784xf32>) -> tensor<784x50xf32> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0]> : tensor<2xi64>, xla_shape = "f32[784,50]{0,1}"} : (tensor<50x784xf32>) -> tensor<784x50xf32>
    return %0 : tensor<784x50xf32>
  }
  func.func private @aten.addmm.24(%arg0: tensor<32x784xf32>, %arg1: tensor<784x50xf32>, %arg2: tensor<50xf32>) -> tensor<32x50xf32> {
    %0 = "mhlo.dot"(%arg0, %arg1) {precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<32x784xf32>, tensor<784x50xf32>) -> tensor<32x50xf32>
    %1 = "mhlo.reshape"(%arg2) : (tensor<50xf32>) -> tensor<1x50xf32>
    %2 = "mhlo.broadcast_in_dim"(%1) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x50xf32>) -> tensor<1x50xf32>
    %3 = "mhlo.reshape"(%2) : (tensor<1x50xf32>) -> tensor<50xf32>
    %4 = "mhlo.broadcast_in_dim"(%3) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<50xf32>) -> tensor<32x50xf32>
    %5 = mhlo.add %0, %4 : tensor<32x50xf32>
    return %5 : tensor<32x50xf32>
  }
  func.func private @aten.relu.35(%arg0: tensor<32x50xf32>) -> tensor<32x50xf32> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<32x50xf32>
    %2 = mhlo.maximum %arg0, %1 : tensor<32x50xf32>
    return %2 : tensor<32x50xf32>
  }
  func.func private @aten.permute.12(%arg0: tensor<50x50xf32>) -> tensor<50x50xf32> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0]> : tensor<2xi64>, xla_shape = "f32[50,50]{0,1}"} : (tensor<50x50xf32>) -> tensor<50x50xf32>
    return %0 : tensor<50x50xf32>
  }
  func.func private @aten.addmm.41(%arg0: tensor<32x50xf32>, %arg1: tensor<50x50xf32>, %arg2: tensor<50xf32>) -> tensor<32x50xf32> {
    %0 = "mhlo.dot"(%arg0, %arg1) {precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<32x50xf32>, tensor<50x50xf32>) -> tensor<32x50xf32>
    %1 = "mhlo.reshape"(%arg2) : (tensor<50xf32>) -> tensor<1x50xf32>
    %2 = "mhlo.broadcast_in_dim"(%1) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x50xf32>) -> tensor<1x50xf32>
    %3 = "mhlo.reshape"(%2) : (tensor<1x50xf32>) -> tensor<50xf32>
    %4 = "mhlo.broadcast_in_dim"(%3) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<50xf32>) -> tensor<32x50xf32>
    %5 = mhlo.add %0, %4 : tensor<32x50xf32>
    return %5 : tensor<32x50xf32>
  }
  func.func private @aten.relu.52(%arg0: tensor<32x50xf32>) -> tensor<32x50xf32> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<32x50xf32>
    %2 = mhlo.maximum %arg0, %1 : tensor<32x50xf32>
    return %2 : tensor<32x50xf32>
  }
  func.func private @aten.permute.8(%arg0: tensor<10x50xf32>) -> tensor<50x10xf32> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0]> : tensor<2xi64>, xla_shape = "f32[50,10]{0,1}"} : (tensor<10x50xf32>) -> tensor<50x10xf32>
    return %0 : tensor<50x10xf32>
  }
  func.func private @aten.addmm.58(%arg0: tensor<32x50xf32>, %arg1: tensor<50x10xf32>, %arg2: tensor<10xf32>) -> tensor<32x10xf32> {
    %0 = "mhlo.dot"(%arg0, %arg1) {precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<32x50xf32>, tensor<50x10xf32>) -> tensor<32x10xf32>
    %1 = "mhlo.reshape"(%arg2) : (tensor<10xf32>) -> tensor<1x10xf32>
    %2 = "mhlo.broadcast_in_dim"(%1) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x10xf32>) -> tensor<1x10xf32>
    %3 = "mhlo.reshape"(%2) : (tensor<1x10xf32>) -> tensor<10xf32>
    %4 = "mhlo.broadcast_in_dim"(%3) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<10xf32>) -> tensor<32x10xf32>
    %5 = mhlo.add %0, %4 : tensor<32x10xf32>
    return %5 : tensor<32x10xf32>
  }
  func.func private @aten.log_softmax.77(%arg0: tensor<32x10xf32>) -> tensor<32x10xf32> {
    %0 = mhlo.constant dense<0xFF800000> : tensor<f32>
    %1 = mhlo.reduce(%arg0 init: %0) across dimensions = [1] : (tensor<32x10xf32>, tensor<f32>) -> tensor<32xf32>
     reducer(%arg1: tensor<f32>, %arg2: tensor<f32>)  {
      %10 = mhlo.maximum %arg1, %arg2 : tensor<f32>
      "mhlo.return"(%10) : (tensor<f32>) -> ()
    }
    %2 = "mhlo.broadcast_in_dim"(%1) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<32xf32>) -> tensor<32x10xf32>
    %3 = mhlo.subtract %arg0, %2 : tensor<32x10xf32>
    %4 = "mhlo.exponential"(%3) : (tensor<32x10xf32>) -> tensor<32x10xf32>
    %5 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = mhlo.reduce(%4 init: %5) across dimensions = [1] : (tensor<32x10xf32>, tensor<f32>) -> tensor<32xf32>
     reducer(%arg1: tensor<f32>, %arg2: tensor<f32>)  {
      %10 = mhlo.add %arg1, %arg2 : tensor<f32>
      "mhlo.return"(%10) : (tensor<f32>) -> ()
    }
    %7 = "mhlo.log"(%6) : (tensor<32xf32>) -> tensor<32xf32>
    %8 = "mhlo.broadcast_in_dim"(%7) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<32xf32>) -> tensor<32x10xf32>
    %9 = mhlo.subtract %3, %8 : tensor<32x10xf32>
    return %9 : tensor<32x10xf32>
  }
}

