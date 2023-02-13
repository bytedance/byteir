// RUN: byteir-opt %s --split-input-file --test-print-shape-analysis | FileCheck %s

func.func @test_transpose_static(%arg0 : tensor<3x4x5xi32>) -> () {
  %0 = arith.constant dense<[2, 1, 0]> : tensor<3xi32>
  %1 = "tosa.transpose"(%arg0, %0) : (tensor<3x4x5xi32>, tensor<3xi32>)  -> (tensor<?x?x?xi32>)
  %2 = "tosa.add"(%arg0, %arg0) : (tensor<3x4x5xi32>, tensor<3x4x5xi32>) -> tensor<?x?x?xi32>
  return
}
// CHECK: for operation : %cst = arith.constant dense<[2, 1, 0]> : tensor<3xi32>, inferred values are:
// CHECK: 	dense<[2, 1, 0]> : tensor<3xi32>
// CHECK: for operation : %0 = "tosa.transpose"(%arg0, %cst) : (tensor<3x4x5xi32>, tensor<3xi32>) -> tensor<?x?x?xi32>, inferred shapes are:
// CHECK: 	tensor<5x4x3xi32>
// CHECK: for operation : %0 = "tosa.transpose"(%arg0, %cst) : (tensor<3x4x5xi32>, tensor<3xi32>) -> tensor<?x?x?xi32>, inferred values are:
// CHECK: 	<UNKNOWN>
// CHECK: for operation : %1 = "tosa.add"(%arg0, %arg0) : (tensor<3x4x5xi32>, tensor<3x4x5xi32>) -> tensor<?x?x?xi32>, inferred shapes are:
// CHECK: 	tensor<3x4x5xi32>
// CHECK: for operation : %1 = "tosa.add"(%arg0, %arg0) : (tensor<3x4x5xi32>, tensor<3x4x5xi32>) -> tensor<?x?x?xi32>, inferred values are:
// CHECK: 	<UNKNOWN>

// -----

func.func @test_transpose_dynamic(%arg : tensor<3x4x5xi32>) -> () {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  %c0_index = arith.constant 0 : index
  %new_shape = tensor.from_elements %c1, %c2, %c0 : tensor<3xi32>
  %1 = "tosa.transpose"(%arg, %new_shape) : (tensor<3x4x5xi32>, tensor<3xi32>)  -> (tensor<?x?x?xi32>)
  %2 = "tosa.add"(%1, %1) : (tensor<?x?x?xi32>, tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
  %3 = shape.shape_of %arg : tensor<3x4x5xi32> -> tensor<3xindex>
  %4 = shape.shape_of %2 : tensor<?x?x?xi32> -> tensor<3xindex>
  %5 = tensor.extract %4[%c0_index] : tensor<3xindex>
  %6 = tensor.dim %2, %c0_index : tensor<?x?x?xi32>
  %7 = arith.addi %5, %6: index
  return
}
// CHECK: for operation : %c0_i32 = arith.constant 0 : i32, inferred values are:
// CHECK: 	0 : i32
// CHECK: for operation : %c1_i32 = arith.constant 1 : i32, inferred values are:
// CHECK: 	1 : i32
// CHECK: for operation : %c2_i32 = arith.constant 2 : i32, inferred values are:
// CHECK: 	2 : i32
// CHECK: for operation : %c0 = arith.constant 0 : index, inferred values are:
// CHECK: 	0 : index
// CHECK: for operation : %from_elements = tensor.from_elements %c1_i32, %c2_i32, %c0_i32 : tensor<3xi32>, inferred values are:
// CHECK: 	dense<[1, 2, 0]> : tensor<3xi32>
// CHECK: for operation : %0 = "tosa.transpose"(%arg0, %from_elements) : (tensor<3x4x5xi32>, tensor<3xi32>) -> tensor<?x?x?xi32>, inferred shapes are:
// CHECK: 	tensor<4x5x3xi32>
// CHECK: for operation : %0 = "tosa.transpose"(%arg0, %from_elements) : (tensor<3x4x5xi32>, tensor<3xi32>) -> tensor<?x?x?xi32>, inferred values are:
// CHECK: 	<UNKNOWN>
// CHECK: for operation : %1 = "tosa.add"(%0, %0) : (tensor<?x?x?xi32>, tensor<?x?x?xi32>) -> tensor<?x?x?xi32>, inferred shapes are:
// CHECK: 	tensor<4x5x3xi32>
// CHECK: for operation : %1 = "tosa.add"(%0, %0) : (tensor<?x?x?xi32>, tensor<?x?x?xi32>) -> tensor<?x?x?xi32>, inferred values are:
// CHECK: 	<UNKNOWN>
// CHECK: for operation : %2 = shape.shape_of %arg0 : tensor<3x4x5xi32> -> tensor<3xindex>, inferred values are:
// CHECK: 	dense<[3, 4, 5]> : tensor<3xindex>
// CHECK: for operation : %3 = shape.shape_of %1 : tensor<?x?x?xi32> -> tensor<3xindex>, inferred values are:
// CHECK: 	dense<[4, 5, 3]> : tensor<3xindex>
// CHECK: for operation : %extracted = tensor.extract %3[%c0] : tensor<3xindex>, inferred values are:
// CHECK: 	4 : index
// CHECK: for operation : %dim = tensor.dim %1, %c0 : tensor<?x?x?xi32>, inferred values are:
// CHECK: 	4 : index
// CHECK: for operation : %4 = arith.addi %extracted, %dim : index, inferred values are:
// CHECK: 	8 : index

// -----

func.func private @test_dynamic_callgraph_inner(%arg: tensor<?x?x?xi32>) -> tensor<?x?x?xi32> {
  %0 = "tosa.add"(%arg, %arg) : (tensor<?x?x?xi32>, tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
// CHECK: for operation : %0 = "tosa.add"(%arg0, %arg0) : (tensor<?x?x?xi32>, tensor<?x?x?xi32>) -> tensor<?x?x?xi32>, inferred shapes are:
// CHECK: 	tensor<4x5x3xi32>
  return %0: tensor<?x?x?xi32>
}


func.func @test_dynamic_callgraph(%arg : tensor<3x4x5xi32>) -> () {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  %new_shape = tensor.from_elements %c1, %c2, %c0 : tensor<3xi32>
  %0 = "tosa.transpose"(%arg, %new_shape) : (tensor<3x4x5xi32>, tensor<3xi32>)  -> (tensor<?x?x?xi32>)
  %1 = call @test_dynamic_callgraph_inner(%0) : (tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
  %2 = shape.shape_of %1 : tensor<?x?x?xi32> -> tensor<3xindex>
// CHECK: for operation : %2 = shape.shape_of %1 : tensor<?x?x?xi32> -> tensor<3xindex>, inferred values are:
// CHECK: 	dense<[4, 5, 3]> : tensor<3xindex>
  return
}

// -----

func.func @test_dynamic_with_cf(%arg0: tensor<3x4x5xi32>, %arg1: i1) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c2_i32 = arith.constant 2 : i32
  %from_elements = tensor.from_elements %c1_i32, %c2_i32, %c0_i32 : tensor<3xi32>
  %0 = "tosa.transpose"(%arg0, %from_elements) : (tensor<3x4x5xi32>, tensor<3xi32>) -> tensor<?x?x?xi32>
  cf.cond_br %arg1, ^bb1(%0 : tensor<?x?x?xi32>), ^bb2(%0 : tensor<?x?x?xi32>)
^bb1(%1: tensor<?x?x?xi32>):  // pred: ^bb0
  %2 = "tosa.add"(%1, %1) : (tensor<?x?x?xi32>, tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
// CHECK: for operation : %2 = "tosa.add"(%1, %1) : (tensor<?x?x?xi32>, tensor<?x?x?xi32>) -> tensor<?x?x?xi32>, inferred shapes are:
// CHECK: 	tensor<4x5x3xi32>
  cf.br ^bb2(%2 : tensor<?x?x?xi32>)
^bb2(%3: tensor<?x?x?xi32>):  // 2 preds: ^bb0, ^bb1
  %4 = shape.shape_of %3 : tensor<?x?x?xi32> -> tensor<3xindex>
// CHECK: for operation : %4 = shape.shape_of %3 : tensor<?x?x?xi32> -> tensor<3xindex>, inferred values are:
// CHECK: 	dense<[4, 5, 3]> : tensor<3xindex>
  return
}

// -----

func.func @test_cf_join_same(%arg0: tensor<3x4x5xi32>, %arg1: i1) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c2_i32 = arith.constant 2 : i32
  %from_elements = tensor.from_elements %c1_i32, %c2_i32, %c0_i32 : tensor<3xi32>
  %0 = "tosa.transpose"(%arg0, %from_elements) : (tensor<3x4x5xi32>, tensor<3xi32>) -> tensor<?x?x?xi32>
  cf.cond_br %arg1, ^bb1(%0 : tensor<?x?x?xi32>), ^bb2(%0 : tensor<?x?x?xi32>)
^bb1(%1: tensor<?x?x?xi32>):  // pred: ^bb0
  %2 = "tosa.add"(%1, %1) : (tensor<?x?x?xi32>, tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
// CHECK: for operation : %2 = "tosa.add"(%1, %1) : (tensor<?x?x?xi32>, tensor<?x?x?xi32>) -> tensor<?x?x?xi32>, inferred shapes are:
// CHECK: 	tensor<4x5x3xi32>
  cf.br ^bb3(%2 : tensor<?x?x?xi32>)
^bb2(%3: tensor<?x?x?xi32>):  // pred: ^bb0
  %4 = "tosa.sub"(%3, %3) : (tensor<?x?x?xi32>, tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
// CHECK: for operation : %4 = "tosa.sub"(%3, %3) : (tensor<?x?x?xi32>, tensor<?x?x?xi32>) -> tensor<?x?x?xi32>, inferred shapes are:
// CHECK: 	tensor<4x5x3xi32>
  cf.br ^bb3(%4 : tensor<?x?x?xi32>)
^bb3(%5: tensor<?x?x?xi32>):  // 2 preds: ^bb1, ^bb2
  %6 = shape.shape_of %5 : tensor<?x?x?xi32> -> tensor<3xindex>
// CHECK: for operation : %6 = shape.shape_of %5 : tensor<?x?x?xi32> -> tensor<3xindex>, inferred values are:
// CHECK: 	dense<[4, 5, 3]> : tensor<3xindex>
  return
}

// -----

func.func @test_cf_join_partial_shape(%arg0: tensor<3x4x5xi32>, %arg1: tensor<3x4x6xi32>, %arg2: i1) {
  %0 = "tosa.identity"(%arg0) : (tensor<3x4x5xi32>) -> tensor<?x?x?xi32>
// CHECK: for operation : %0 = "tosa.identity"(%arg0) : (tensor<3x4x5xi32>) -> tensor<?x?x?xi32>, inferred shapes are:
// CHECK:   tensor<3x4x5xi32>
  %1 = "tosa.identity"(%arg1) : (tensor<3x4x6xi32>) -> tensor<?x?x?xi32>
// CHECK: for operation : %1 = "tosa.identity"(%arg1) : (tensor<3x4x6xi32>) -> tensor<?x?x?xi32>, inferred shapes are:
// CHECK:   tensor<3x4x6xi32>
  cf.cond_br %arg2, ^bb1(%0 : tensor<?x?x?xi32>), ^bb1(%1 : tensor<?x?x?xi32>)
^bb1(%2: tensor<?x?x?xi32>):  // 2 preds: ^bb0, ^bb0
  %3 = "tosa.add"(%2, %2) : (tensor<?x?x?xi32>, tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
// CHECK: for operation : %3 = "tosa.add"(%2, %2) : (tensor<?x?x?xi32>, tensor<?x?x?xi32>) -> tensor<?x?x?xi32>, inferred shapes are:
// CHECK: 	tensor<3x4x?xi32>
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %3, %c0 : tensor<?x?x?xi32>
// CHECK: for operation : %dim = tensor.dim %3, %c0 : tensor<?x?x?xi32>, inferred values are:
// CHECK: 	3 : index
  return
}

// -----

func.func private @test_callgraph_join_same_inner(%arg: tensor<?x?x?xi32>) -> tensor<?x?x?xi32> {
  %0 = "tosa.add"(%arg, %arg) : (tensor<?x?x?xi32>, tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
// CHECK: for operation : %0 = "tosa.add"(%arg0, %arg0) : (tensor<?x?x?xi32>, tensor<?x?x?xi32>) -> tensor<?x?x?xi32>, inferred shapes are:
// CHECK: 	tensor<4x5x3xi32>
  return %0: tensor<?x?x?xi32>
}

func.func @test_callgraph_join_same(%arg : tensor<3x4x5xi32>) -> () {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  %new_shape = tensor.from_elements %c1, %c2, %c0 : tensor<3xi32>
  %1 = "tosa.transpose"(%arg, %new_shape) : (tensor<3x4x5xi32>, tensor<3xi32>)  -> (tensor<?x?x?xi32>)
  %2 = call @test_callgraph_join_same_inner(%1) : (tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
  %3 = call @test_callgraph_join_same_inner(%1) : (tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
  return
}

// -----

func.func private @test_callgraph_join_partial_shape_inner(%arg0: tensor<?x?x?xi32>) -> tensor<?x?x?xi32> {
  %0 = "tosa.add"(%arg0, %arg0) : (tensor<?x?x?xi32>, tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
// CHECK: for operation : %0 = "tosa.add"(%arg0, %arg0) : (tensor<?x?x?xi32>, tensor<?x?x?xi32>) -> tensor<?x?x?xi32>, inferred shapes are:
// CHECK:   tensor<3x4x?xi32>
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %0, %c0 : tensor<?x?x?xi32>
// CHECK: for operation : %dim = tensor.dim %0, %c0 : tensor<?x?x?xi32>, inferred values are:
// CHECK: 	3 : index
  return %0 : tensor<?x?x?xi32>
}
func.func @test_callgraph_join_partial_shape(%arg0: tensor<3x4x5xi32>, %arg1: tensor<3x4x6xi32>) {
  %0 = "tosa.identity"(%arg0) : (tensor<3x4x5xi32>) -> tensor<?x?x?xi32>
// CHECK: for operation : %0 = "tosa.identity"(%arg0) : (tensor<3x4x5xi32>) -> tensor<?x?x?xi32>, inferred shapes are:
// CHECK:   tensor<3x4x5xi32>
  %1 = "tosa.identity"(%arg1) : (tensor<3x4x6xi32>) -> tensor<?x?x?xi32>
// CHECK: for operation : %1 = "tosa.identity"(%arg1) : (tensor<3x4x6xi32>) -> tensor<?x?x?xi32>, inferred shapes are:
// CHECK:   tensor<3x4x6xi32>
  %2 = call @test_callgraph_join_partial_shape_inner(%0) : (tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
  %3 = call @test_callgraph_join_partial_shape_inner(%1) : (tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
  return
}
