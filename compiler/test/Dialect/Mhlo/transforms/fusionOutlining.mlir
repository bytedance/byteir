// RUN: byteir-opt -fusion-outlining %s | FileCheck %s

func.func @mhlo_add(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  %0 = "mhlo.fusion"(%arg0, %arg1) ( {
    %1 = mhlo.add %arg0, %arg1 : tensor<4xf32>
    %2 = mhlo.add %arg0, %1 : tensor<4xf32>
    "mhlo.return"(%2) : (tensor<4xf32>) -> ()
  }) {some_attr} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}
// CHECK-LABEL: func.func private @Unknown0
// CHECK-SAME: {some_attr}
// CHECK:   %[[VAR_0:.*]] = mhlo.add %{{.*}}, %{{.*}} : tensor<4xf32>
// CHECK:   %[[VAR_1:.*]] = mhlo.add %{{.*}}, %[[VAR_0]] : tensor<4xf32>
// CHECK:   return %[[VAR_1]] : tensor<4xf32>

// CHECK-LABEL: func.func @mhlo_add
// CHECK:   %[[VAR_0:.*]] = call @Unknown0(%{{.*}}, %{{.*}}) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
// CHECK:   return %[[VAR_0]] : tensor<4xf32>

func.func @mhlo_add_2(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  %0 = "mhlo.fusion"(%arg0, %arg1) ( {
    %1 = mhlo.add %arg0, %arg1 : tensor<4xf32>
    %2 = mhlo.add %arg0, %1 : tensor<4xf32>
    "mhlo.return"(%2) : (tensor<4xf32>) -> ()
  }) {byre_compute_name = "TestFunc", some_attr} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}
// CHECK-LABEL: func.func private @TestFunc1
// CHECK-SAME: {byre_compute_name = "TestFunc", some_attr}
// CHECK:   %[[VAR_0:.*]] = mhlo.add %{{.*}}, %{{.*}} : tensor<4xf32>
// CHECK:   %[[VAR_1:.*]] = mhlo.add %{{.*}}, %[[VAR_0]] : tensor<4xf32>
// CHECK:   return %[[VAR_1]] : tensor<4xf32>

// CHECK-LABEL: func.func @mhlo_add_2
// CHECK:   %[[VAR_0:.*]] = call @TestFunc1(%{{.*}}, %{{.*}}) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
// CHECK:   return %[[VAR_0]] : tensor<4xf32>

func.func @mhlo_add_3(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  %0 = "mhlo.fusion"(%arg0, %arg1) ( {
    %1 = mhlo.add %arg0, %arg1 : tensor<4xf32>
    %2 = mhlo.add %arg0, %1 : tensor<4xf32>
    "mhlo.return"(%2) : (tensor<4xf32>) -> ()
  }) {byre_compute_name = "TestFunc"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}
// CHECK-LABEL: func.func @mhlo_add_3
// CHECK:   %[[VAR_0:.*]] = call @TestFunc2(%{{.*}}, %{{.*}}) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
// CHECK:   return %[[VAR_0]] : tensor<4xf32>

func.func @mhlo_add_outer_scope_cst(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  %0 = mhlo.constant dense<0.000000e+00> : tensor<4xf32>
  %1 = "mhlo.fusion"(%arg0, %arg1) ( {
    %2 = mhlo.add %arg0, %0 : tensor<4xf32>
    %3 = mhlo.add %arg1, %2 : tensor<4xf32>
    "mhlo.return"(%3) : (tensor<4xf32>) -> ()
  }) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %1 : tensor<4xf32>
}
// CHECK-LABEL: func.func private @Unknown3
// CEHCK:  %[[VAR_0:.*]] = mhlo.constant dense<0.000000e+00> : tensor<4xf32>
// CHECK:  %[[VAR_1:.*]] = mhlo.add %{{.*}}, %[[VAR_0]] : tensor<4xf32>
// CHECK:  %[[VAR_2:.*]] = mhlo.add %{{.*}}, %[[VAR_1]] : tensor<4xf32>
// CHECK:  return %[[VAR_2]] : tensor<4xf32>

// CHECK-LABEL: func.func @mhlo_add_outer_scope_cst
// CHECK:   %[[VAR_0:.*]] = call @Unknown3(%{{.*}}, %{{.*}}) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
// CHECK:   return %[[VAR_0]] : tensor<4xf32>

func.func @mhlo_share_constant(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  %0 = mhlo.constant dense<0.000000e+00> : tensor<4xf32>
  %1 = "mhlo.fusion"(%arg0, %arg1) ( {
    %2 = mhlo.add %arg0, %0 : tensor<4xf32>
    %3 = mhlo.add %arg1, %2 : tensor<4xf32>
    "mhlo.return"(%3) : (tensor<4xf32>) -> ()
  }) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  %2 = "mhlo.fusion"(%arg0, %arg1) ( {
    %2 = mhlo.subtract %arg0, %0 : tensor<4xf32>
    %3 = mhlo.subtract %arg1, %2 : tensor<4xf32>
    "mhlo.return"(%3) : (tensor<4xf32>) -> ()
  }) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %1, %2 : tensor<4xf32>, tensor<4xf32>
}
// CHECK-LABEL: func.func private @Unknown4
// CEHCK:  %[[VAR_0:.*]] = mhlo.constant dense<0.000000e+00> : tensor<4xf32>
// CHECK:  %[[VAR_1:.*]] = mhlo.add %{{.*}}, %[[VAR_0:.*]] : tensor<4xf32>
// CHECK:  %[[VAR_2:.*]] = mhlo.add %{{.*}}, %[[VAR_1]] : tensor<4xf32>
// CHECK:  return %[[VAR_2]] : tensor<4xf32>

// CHECK-LABEL: func.func private @Unknown5
// CEHCK:  %[[VAR_0:.*]] = mhlo.constant dense<0.000000e+00> : tensor<4xf32>
// CHECK:  %[[VAR_1:.*]] = mhlo.subtract %{{.*}}, %[[VAR_0]] : tensor<4xf32>
// CHECK:  %[[VAR_2:.*]] = mhlo.subtract %{{.*}}, %[[VAR_1]] : tensor<4xf32>
// CHECK:  return %[[VAR_2]] : tensor<4xf32>

// CHECK-LABEL: func.func @mhlo_share_constant
// CHECK:   %[[VAR_0:.*]] = call @Unknown4(%{{.*}}, %{{.*}}) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
// CHECK:   %[[VAR_1:.*]] = call @Unknown5(%{{.*}}, %{{.*}}) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
// CHECK:   return %[[VAR_0]], %[[VAR_1]] : tensor<4xf32>, tensor<4xf32>
