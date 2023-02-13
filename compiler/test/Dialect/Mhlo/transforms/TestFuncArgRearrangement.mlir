// RUN: byteir-opt %s -test-rearrange-func-arg -canonicalize-ext | FileCheck %s

func.func private @test_func1(%arg0: tensor<4x2xf32>, %arg1: tensor<4x3xf32>, %arg2: tensor<4x4xf32>, %arg3: tensor<4x5xf32>, %arg4: tensor<4x2x3xf32>, %arg5: tensor<4x7xf32>) -> (tensor<128x2xf32>, tensor<128x3xf32>, tensor<128x4xf32>, tensor<128x5xf32>, tensor<128x2x3xf32>) attributes {__byteir_unit_test__ = {arg = [["pack", 3, 0, 2], ["identity", 1], ["pack2d", 5, 4]], result = [["pack", 2, 0], ["identity", 1], ["pack2d", 4, 3]]}, other_attr_1, test_rewrite_from}
// CHECK-LABEL: func.func private @test_func1
// CHECK-SAME: (tensor<4x11xf32>, tensor<4x3xf32>, tensor<4x13xf32>) -> (tensor<128x6xf32>, tensor<128x3xf32>, tensor<128x11xf32>) attributes {other_attr_1, test_rewrite_to}

func.func @main1(%arg0: tensor<4x2xf32>, %arg1: tensor<4x3xf32>, %arg2: tensor<4x4xf32>, %arg3: tensor<4x5xf32>, %arg4: tensor<4x2x3xf32>, %arg5: tensor<4x7xf32>) -> (tensor<128x2xf32>, tensor<128x3xf32>, tensor<128x4xf32>, tensor<128x5xf32>, tensor<128x2x3xf32>) attributes {__byteir_unit_test__ = {arg = [["pack", 3, 0, 2], ["identity", 1], ["pack2d", 5, 4]], result = [["pack", 2, 0], ["identity", 1], ["pack2d", 4, 3]]}, other_attr_2, test_rewrite_from} {
  %0:5 = call @test_func1(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5) : (tensor<4x2xf32>, tensor<4x3xf32>, tensor<4x4xf32>, tensor<4x5xf32>, tensor<4x2x3xf32>, tensor<4x7xf32>) -> (tensor<128x2xf32>, tensor<128x3xf32>, tensor<128x4xf32>, tensor<128x5xf32>, tensor<128x2x3xf32>)
  return %0#0, %0#1, %0#2, %0#3, %0#4 : tensor<128x2xf32>, tensor<128x3xf32>, tensor<128x4xf32>, tensor<128x5xf32>, tensor<128x2x3xf32>
}
// CHECK-LABEL: func.func @main1
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4x11xf32>, %[[ARG1:.*]]: tensor<4x3xf32>, %[[ARG2:.*]]: tensor<4x13xf32>) -> (tensor<128x6xf32>, tensor<128x3xf32>, tensor<128x11xf32>) attributes {other_attr_2, test_rewrite_to}
// CHECK-NEXT: %[[V0:.*]]:3 = call @test_func1(%[[ARG0]], %[[ARG1]], %[[ARG2]]) : (tensor<4x11xf32>, tensor<4x3xf32>, tensor<4x13xf32>) -> (tensor<128x6xf32>, tensor<128x3xf32>, tensor<128x11xf32>)
// CHECK-NEXT: return %[[V0:.*]]#0, %[[V0:.*]]#1, %[[V0:.*]]#2 : tensor<128x6xf32>, tensor<128x3xf32>, tensor<128x11xf32>

func.func private @test_func_duplicate(%arg0: tensor<4x2xf32>, %arg1: tensor<4x3xf32>, %arg2: tensor<4x4xf32>) -> (tensor<128x2xf32>, tensor<128x3xf32>, tensor<128x4xf32>) attributes {__byteir_unit_test__ = {arg = [["pack", 1, 0], ["identity", 1], ["pack", 1, 2]], result = [["pack", 0, 1], ["identity", 1], ["pack", 2, 1]]}, other_attr_1}
// CHECK-LABEL: func.func private @test_func_duplicate
// CHECK-SAME: (tensor<4x5xf32>, tensor<4x3xf32>, tensor<4x7xf32>) -> (tensor<128x5xf32>, tensor<128x3xf32>, tensor<128x7xf32>) attributes {other_attr_1}

func.func @main_duplicate(%arg0: tensor<4x2xf32>, %arg1: tensor<4x3xf32>, %arg2: tensor<4x4xf32>) -> (tensor<128x2xf32>, tensor<128x3xf32>, tensor<128x4xf32>) attributes {__byteir_unit_test__ = {arg = [["pack", 1, 0], ["identity", 1], ["pack", 1, 2]], result = [["pack", 0, 1], ["identity", 1], ["pack", 2, 1]]}, other_attr_2} {
  %0:3 = call @test_func_duplicate(%arg0, %arg1, %arg2) : (tensor<4x2xf32>, tensor<4x3xf32>, tensor<4x4xf32>) -> (tensor<128x2xf32>, tensor<128x3xf32>, tensor<128x4xf32>)
  return %0#0, %0#1, %0#2 : tensor<128x2xf32>, tensor<128x3xf32>, tensor<128x4xf32>
}
// CHECK-LABEL: func.func @main_duplicate
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4x5xf32>, %[[ARG1:.*]]: tensor<4x3xf32>, %[[ARG2:.*]]: tensor<4x7xf32>) -> (tensor<128x5xf32>, tensor<128x3xf32>, tensor<128x7xf32>) attributes {other_attr_2}
// CHECK-NEXT: %[[V0:.*]]:3 = call @test_func_duplicate(%[[ARG0]], %[[ARG1]], %[[ARG2]]) : (tensor<4x5xf32>, tensor<4x3xf32>, tensor<4x7xf32>) -> (tensor<128x5xf32>, tensor<128x3xf32>, tensor<128x7xf32>)
// CHECK-NEXT: return %[[V0:.*]]#0, %[[V0:.*]]#1, %[[V0:.*]]#2 : tensor<128x5xf32>, tensor<128x3xf32>, tensor<128x7xf32>

func.func private @test_func_duplicate_2(%arg0: tensor<4x2xf32>, %arg1: tensor<4x3xf32>, %arg2: tensor<4x4xf32>, %arg3: tensor<4x5xf32>, %arg4: tensor<4x6xf32>, %arg5: tensor<4x7xf32>) -> (tensor<128x2xf32>, tensor<128x3xf32>, tensor<128x4xf32>) attributes {__byteir_unit_test__ = {arg = [["pack", 0, 1, 2, 3], ["pack", 0, 1, 4, 5], ["identity", 2]], result = [["pack", 0, 1], ["identity", 1], ["pack", 2, 1]]}, other_attr_1}
// CHECK-LABEL: func.func private @test_func_duplicate_2
// CHECK-SAME: (tensor<4x14xf32>, tensor<4x18xf32>, tensor<4x4xf32>) -> (tensor<128x5xf32>, tensor<128x3xf32>, tensor<128x7xf32>) attributes {other_attr_1}

func.func @main_duplicate_2(%arg0: tensor<4x2xf32>, %arg1: tensor<4x3xf32>, %arg2: tensor<4x4xf32>, %arg3: tensor<4x5xf32>, %arg4: tensor<4x6xf32>, %arg5: tensor<4x7xf32>) -> (tensor<128x2xf32>, tensor<128x3xf32>, tensor<128x4xf32>) attributes {__byteir_unit_test__ = {arg = [["pack", 0, 1, 2, 3], ["pack", 0, 1, 4, 5], ["identity", 2]], result = [["pack", 0, 1], ["identity", 1], ["pack", 2, 1]]}, other_attr_2} {
  %0:3 = call @test_func_duplicate_2(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5) : (tensor<4x2xf32>, tensor<4x3xf32>, tensor<4x4xf32>, tensor<4x5xf32>, tensor<4x6xf32>, tensor<4x7xf32>) -> (tensor<128x2xf32>, tensor<128x3xf32>, tensor<128x4xf32>)
  return %0#0, %0#1, %0#2 : tensor<128x2xf32>, tensor<128x3xf32>, tensor<128x4xf32>
}
// CHECK-LABEL: func.func @main_duplicate_2
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4x14xf32>, %[[ARG1:.*]]: tensor<4x18xf32>, %[[ARG2:.*]]: tensor<4x4xf32>) -> (tensor<128x5xf32>, tensor<128x3xf32>, tensor<128x7xf32>) attributes {other_attr_2}
// CHECK-NEXT: %[[V0:.*]]:3 = call @test_func_duplicate_2(%[[ARG0]], %[[ARG1]], %[[ARG2]]) : (tensor<4x14xf32>, tensor<4x18xf32>, tensor<4x4xf32>) -> (tensor<128x5xf32>, tensor<128x3xf32>, tensor<128x7xf32>)
// CHECK-NEXT: return %[[V0:.*]]#0, %[[V0:.*]]#1, %[[V0:.*]]#2 : tensor<128x5xf32>, tensor<128x3xf32>, tensor<128x7xf32>

func.func private @test_func_duplicate_3(%arg0: tensor<4x2xf32>, %arg1: tensor<4x3xf32>, %arg2: tensor<4x4xf32>, %arg3: tensor<4x5xf32>, %arg4: tensor<4x6xf32>, %arg5: tensor<4x7xf32>) -> (tensor<128x2xf32>, tensor<128x3xf32>, tensor<128x4xf32>) attributes {__byteir_unit_test__ = {arg = [["pack", 0, 1, 2, 3], ["identity", 2], ["pack", 0, 1, 4, 5]], result = [["identity", 1], ["pack", 0, 1], ["pack", 2, 1]]}, other_attr_1}
// CHECK-LABEL: func.func private @test_func_duplicate_3
// CHECK-SAME: (tensor<4x14xf32>, tensor<4x4xf32>, tensor<4x18xf32>) -> (tensor<128x3xf32>, tensor<128x5xf32>, tensor<128x7xf32>) attributes {other_attr_1}

func.func @main_duplicate_3(%arg0: tensor<4x2xf32>, %arg1: tensor<4x3xf32>, %arg2: tensor<4x4xf32>, %arg3: tensor<4x5xf32>, %arg4: tensor<4x6xf32>, %arg5: tensor<4x7xf32>) -> (tensor<128x2xf32>, tensor<128x3xf32>, tensor<128x4xf32>) attributes {__byteir_unit_test__ = {arg = [["pack", 0, 1, 2, 3], ["identity", 2], ["pack", 0, 1, 4, 5]], result = [["identity", 1], ["pack", 0, 1], ["pack", 2, 1]]}, other_attr_2} {
  %0:3 = call @test_func_duplicate_3(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5) : (tensor<4x2xf32>, tensor<4x3xf32>, tensor<4x4xf32>, tensor<4x5xf32>, tensor<4x6xf32>, tensor<4x7xf32>) -> (tensor<128x2xf32>, tensor<128x3xf32>, tensor<128x4xf32>)
  return %0#0, %0#1, %0#2 : tensor<128x2xf32>, tensor<128x3xf32>, tensor<128x4xf32>
}
// CHECK-LABEL: func.func @main_duplicate_3
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4x14xf32>, %[[ARG1:.*]]: tensor<4x4xf32>, %[[ARG2:.*]]: tensor<4x18xf32>) -> (tensor<128x3xf32>, tensor<128x5xf32>, tensor<128x7xf32>) attributes {other_attr_2}
// CHECK-NEXT: %[[V0:.*]]:3 = call @test_func_duplicate_3(%[[ARG0]], %[[ARG1]], %[[ARG2]]) : (tensor<4x14xf32>, tensor<4x4xf32>, tensor<4x18xf32>) -> (tensor<128x3xf32>, tensor<128x5xf32>, tensor<128x7xf32>)
// CHECK-NEXT: return %[[V0:.*]]#0, %[[V0:.*]]#1, %[[V0:.*]]#2 : tensor<128x3xf32>, tensor<128x5xf32>, tensor<128x7xf32>


func.func private @test_func_duplicate_4(%arg0: tensor<4x2xf32>, %arg1: tensor<4x3xf32>, %arg2: tensor<4x4x4xf32>, %arg3: tensor<4x5xf32>, %arg4: tensor<4x6xf32>, %arg5: tensor<4x7xf32>) -> (tensor<128x2xf32>, tensor<128x3xf32>, tensor<128x4xf32>) attributes {__byteir_unit_test__ = {arg = [["pack2d", 0, 1, 2, 3], ["identity", 2], ["pack", 0, 1, 4, 5]], result = [["identity", 1], ["pack", 0, 1], ["pack", 2, 1]]}, other_attr_1}
// CHECK-LABEL: func.func private @test_func_duplicate_4
// CHECK-SAME: (tensor<4x26xf32>, tensor<4x4x4xf32>, tensor<4x18xf32>) -> (tensor<128x3xf32>, tensor<128x5xf32>, tensor<128x7xf32>) attributes {other_attr_1}

func.func @main_duplicate_4(%arg0: tensor<4x2xf32>, %arg1: tensor<4x3xf32>, %arg2: tensor<4x4x4xf32>, %arg3: tensor<4x5xf32>, %arg4: tensor<4x6xf32>, %arg5: tensor<4x7xf32>) -> (tensor<128x2xf32>, tensor<128x3xf32>, tensor<128x4xf32>) attributes {__byteir_unit_test__ = {arg = [["pack2d", 0, 1, 2, 3], ["identity", 2], ["pack", 0, 1, 4, 5]], result = [["identity", 1], ["pack", 0, 1], ["pack", 2, 1]]}, other_attr_2} {
  %0:3 = call @test_func_duplicate_4(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5) : (tensor<4x2xf32>, tensor<4x3xf32>, tensor<4x4x4xf32>, tensor<4x5xf32>, tensor<4x6xf32>, tensor<4x7xf32>) -> (tensor<128x2xf32>, tensor<128x3xf32>, tensor<128x4xf32>)
  return %0#0, %0#1, %0#2 : tensor<128x2xf32>, tensor<128x3xf32>, tensor<128x4xf32>
}
// CHECK-LABEL: func.func @main_duplicate_4
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4x26xf32>, %[[ARG1:.*]]: tensor<4x4x4xf32>, %[[ARG2:.*]]: tensor<4x18xf32>) -> (tensor<128x3xf32>, tensor<128x5xf32>, tensor<128x7xf32>) attributes {other_attr_2}
// CHECK-NEXT: %[[V0:.*]]:3 = call @test_func_duplicate_4(%[[ARG0]], %[[ARG1]], %[[ARG2]]) : (tensor<4x26xf32>, tensor<4x4x4xf32>, tensor<4x18xf32>) -> (tensor<128x3xf32>, tensor<128x5xf32>, tensor<128x7xf32>)
// CHECK-NEXT: return %[[V0:.*]]#0, %[[V0:.*]]#1, %[[V0:.*]]#2 : tensor<128x3xf32>, tensor<128x5xf32>, tensor<128x7xf32>

func.func @main_use_before_declare(%arg0: tensor<4x2xf32>, %arg1: tensor<4x3xf32>, %arg2: tensor<4x4xf32>, %arg3: tensor<4x5xf32>, %arg4: tensor<4x6xf32>, %arg5: tensor<4x7xf32>) -> (tensor<128x2xf32>, tensor<128x3xf32>, tensor<128x4xf32>) attributes {__byteir_unit_test__ = {arg = [["pack", 0, 1, 2, 3], ["identity", 2], ["pack", 0, 1, 4, 5]], result = [["identity", 1], ["pack", 0, 1], ["pack", 2, 1]]}, other_attr_2} {
  %0:3 = call @test_use_before_declare(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5) : (tensor<4x2xf32>, tensor<4x3xf32>, tensor<4x4xf32>, tensor<4x5xf32>, tensor<4x6xf32>, tensor<4x7xf32>) -> (tensor<128x2xf32>, tensor<128x3xf32>, tensor<128x4xf32>)
  return %0#0, %0#1, %0#2 : tensor<128x2xf32>, tensor<128x3xf32>, tensor<128x4xf32>
}
// CHECK-LABEL: func.func @main_use_before_declare
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4x14xf32>, %[[ARG1:.*]]: tensor<4x4xf32>, %[[ARG2:.*]]: tensor<4x18xf32>) -> (tensor<128x3xf32>, tensor<128x5xf32>, tensor<128x7xf32>) attributes {other_attr_2}
// CHECK-NEXT: %[[V0:.*]]:3 = call @test_use_before_declare(%[[ARG0]], %[[ARG1]], %[[ARG2]]) : (tensor<4x14xf32>, tensor<4x4xf32>, tensor<4x18xf32>) -> (tensor<128x3xf32>, tensor<128x5xf32>, tensor<128x7xf32>)
// CHECK-NEXT: return %[[V0:.*]]#0, %[[V0:.*]]#1, %[[V0:.*]]#2 : tensor<128x3xf32>, tensor<128x5xf32>, tensor<128x7xf32>

func.func private @test_use_before_declare(%arg0: tensor<4x2xf32>, %arg1: tensor<4x3xf32>, %arg2: tensor<4x4xf32>, %arg3: tensor<4x5xf32>, %arg4: tensor<4x6xf32>, %arg5: tensor<4x7xf32>) -> (tensor<128x2xf32>, tensor<128x3xf32>, tensor<128x4xf32>) attributes {__byteir_unit_test__ = {arg = [["pack", 0, 1, 2, 3], ["identity", 2], ["pack", 0, 1, 4, 5]], result = [["identity", 1], ["pack", 0, 1], ["pack", 2, 1]]}, other_attr_1}
// CHECK-LABEL: func.func private @test_use_before_declare
// CHECK-SAME: (tensor<4x14xf32>, tensor<4x4xf32>, tensor<4x18xf32>) -> (tensor<128x3xf32>, tensor<128x5xf32>, tensor<128x7xf32>) attributes {other_attr_1}
