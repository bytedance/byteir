// RUN: byteir-opt %s --set-assuming-always-true --canonicalize | FileCheck %s
func.func @case0(%arg0: tensor<?x35xf32, {bounded_shape = [4, 35]}>, %arg1: tensor<?x32xf32, {bounded_shape = [4, 32]}>, %arg2: tensor<?x52xf32, {bounded_shape = [4, 52]}>) -> tensor<?x52xf32, {bounded_shape = [4, 52]}> {
  %c0 = arith.constant 0 : index
  %c52 = arith.constant 52 : index
  %0 = tensor.dim %arg0, %c0 : tensor<?x35xf32, {bounded_shape = [4, 35]}>
  %1 = tensor.from_elements %0, %c52 : tensor<2xindex>
  %2 = tensor.dim %arg1, %c0 : tensor<?x32xf32, {bounded_shape = [4, 32]}>
  %3 = tensor.from_elements %2, %c52 : tensor<2xindex>
  "shape_ext.tie"(%arg2, %2) : (tensor<?x52xf32, {bounded_shape = [4, 52]}>, index) -> ()
  %4 = shape.cstr_broadcastable %1, %3 : tensor<2xindex>, tensor<2xindex>
  %5 = shape.assuming %4 -> (tensor<?x52xf32, {bounded_shape = [4, 52]}>) {
    %6 = shape.broadcast %1, %3 : tensor<2xindex>, tensor<2xindex> -> tensor<2xindex>
    %7 = "mhlo.dynamic_broadcast_in_dim"(%arg2, %6) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<?x52xf32, {bounded_shape = [4, 52]}>, tensor<2xindex>) -> tensor<?x52xf32, {bounded_shape = [4, 52]}>
    %8 = tensor.extract %6[%c0] : tensor<2xindex>
    "shape_ext.tie"(%7, %8) : (tensor<?x52xf32, {bounded_shape = [4, 52]}>, index) -> ()
    %9 = mhlo.multiply %7, %arg2 : tensor<?x52xf32, {bounded_shape = [4, 52]}>
    "shape_ext.tie"(%9, %8) : (tensor<?x52xf32, {bounded_shape = [4, 52]}>, index) -> ()
    shape.assuming_yield %9 : tensor<?x52xf32, {bounded_shape = [4, 52]}>
  }
  return %5 : tensor<?x52xf32, {bounded_shape = [4, 52]}>
}

//CHECK-LABEL: func.func @case0
//CHECK-NOT:  shape.assuming
