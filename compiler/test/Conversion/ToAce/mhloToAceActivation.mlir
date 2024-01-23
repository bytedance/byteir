// RUN: byteir-opt %s -convert-mhlo-to-ace | FileCheck %s


func.func @relu(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = mhlo.constant dense<0.000000e+00> : tensor<4x4xf32>
  %1 = mhlo.maximum %arg0, %0 : tensor<4x4xf32>
  return %1 : tensor<4x4xf32>
}
// CHECK-LABEL: func.func @relu(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
// CHECK-NEXT:    %0 = "ace.activate"(%arg0) <{act_func = "relu"}> : (tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-NEXT:    return %0 : tensor<4x4xf32>
// CHECK-NEXT:  }

func.func @relu6(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = mhlo.constant dense<6.000000e+00> : tensor<f32>
  %2 = "mhlo.clamp"(%0, %arg0, %1) : (tensor<f32>, tensor<4x4xf32>, tensor<f32>) -> tensor<4x4xf32>
  return %2 : tensor<4x4xf32>
}
// CHECK-LABEL:  func.func @relu6(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
// CHECK-NEXT:    %0 = "ace.activate"(%arg0) <{act_func = "relu6"}> : (tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-NEXT:    return %0 : tensor<4x4xf32>
// CHECK-NEXT:  }

func.func @leaky_relu(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = mhlo.constant dense<2.000000e-01> : tensor<4x4xf32>
  %1 = mhlo.constant dense<0.000000e+00> : tensor<4x4xf32>
  %2 = mhlo.multiply %arg0, %0 : tensor<4x4xf32>
  %3 = "mhlo.compare"(%arg0, %1) {comparison_direction = #mhlo<comparison_direction GT>} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xi1>
  %4 = "mhlo.select"(%3, %arg0, %2) : (tensor<4x4xi1>, tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  return %4 : tensor<4x4xf32>
}
// CHECK-LABEL: func.func @leaky_relu(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
// CHECK-NEXT:    %0 = "ace.activate"(%arg0) <{act_func = "leaky_relu"}> {alpha = 2.000000e-01 : f32} : (tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-NEXT:    return %0 : tensor<4x4xf32>
// CHECK-NEXT:  }
