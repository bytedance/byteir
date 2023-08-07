// RUN: byteir-opt -hlo-fusion-to-linalg -cse %s | FileCheck %s

func.func @convert_rng_static() -> tensor<8x1024x768xf32> attributes {__placeholder__byre.entry_point} {
    %0 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %1 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = byre.compute @GetSeed() -> tensor<i64>
    %3 = byre.compute @GetOffset() -> tensor<i64>
    %4 = mhlo.custom_call @byteir.rng_uniform(%1, %0, %2, %3) {backend_config = ""} : (tensor<f32>, tensor<f32>, tensor<i64>, tensor<i64>) -> tensor<8x1024x768xf32>
    return %4 : tensor<8x1024x768xf32>
  }
// CHECK-LABEL: func.func @convert_rng_static
// CHECK-DAG: arith.constant
// CHECK-DAG: arith.constant
// CHECK-DAG: byre.compute
// CHECK-DAG: byre.compute
// CHECK-DAG: tensor.empty
// CHECK-NEXT: linalg.generic
// CHECK-DAG:  ^{{.*}}(%[[MIN:.+]]: f32, %[[MAX:.+]]: f32, %[[SEED:.+]]: i64, %[[OFFSET:.+]]: i64, %[[OUT:.+]]: f32
// CHECK-DAG:    %[[I32SEED:.+]] = arith.trunci %[[SEED]] : i64 to i32
// CHECK-DAG:    %[[I32OFFSET:.+]] = arith.trunci %[[OFFSET]] : i64 to i32
// CHECK-DAG:    %[[KSEED:.+]] = arith.addi %[[I32SEED]], %[[I32OFFSET]] : i32
// CHECK-DAG:    %[[CST0:.+]] = arith.constant 1103515245 : i32
// CHECK-DAG:    %[[CST1:.+]] = arith.constant 12345 : i32
// CHECK-DAG:    %[[CST2:.+]] = arith.constant 2.32830644E-10 : f32
// CHECK-DAG:    %[[IDX0:.+]] = linalg.index 0 : index
// CHECK-DAG:    %[[IDX0_CAST:.+]] = arith.index_cast %[[IDX0]] : index to i32
// CHECK-DAG:    %[[IDX1:.+]] = linalg.index 1 : index
// CHECK-DAG:    %[[IDX1_CAST:.+]] = arith.index_cast %[[IDX1]] : index to i32
// CHECK-DAG:    %[[IDX2:.+]] = linalg.index 2 : index
// CHECK-DAG:    %[[IDX2_CAST:.+]] = arith.index_cast %[[IDX2]] : index to i32
// CHECK-DAG:    %[[START:.+]] = arith.addi %[[IDX0_CAST]], %[[KSEED]] : i32
// CHECK-DAG:    %[[VAL1:.+]] = arith.muli %[[START]], %[[CST0]] : i32
// CHECK-DAG:    %[[VAL2:.+]] = arith.addi %[[VAL1]], %[[CST1]] : i32
// CHECK-DAG:    %[[VAL3:.+]] = arith.addi %[[IDX1_CAST]], %[[VAL2]] : i32
// CHECK-DAG:    %[[VAL4:.+]] = arith.muli %[[VAL3]], %[[CST0]] : i32
// CHECK-DAG:    %[[VAL5:.+]] = arith.addi %[[VAL4]], %[[CST1]] : i32
// CHECK-DAG:    %[[VAL6:.+]] = arith.addi %[[IDX2_CAST]], %[[VAL5]] : i32
// CHECK-DAG:    %[[VAL7:.+]] = arith.muli %[[VAL6]], %[[CST0]] : i32
// CHECK-DAG:    %[[VAL8:.+]] = arith.addi %[[VAL7]], %[[CST1]] : i32
// CHECK-DAG:    %[[DIFF:.+]] = arith.subf %[[MAX]], %[[MIN]] : f32
// CHECK-DAG:    %[[FACT:.+]] = arith.mulf %[[DIFF]], %[[CST2]] : f32
// CHECK-DAG:    %[[VAL8_CAST:.+]] = arith.uitofp %[[VAL8]] : i32 to f32
// CHECK-DAG:    %[[VAL6:.+]] = arith.mulf %[[VAL8_CAST]], %[[FACT]] : f32
// CHECK-DAG:    %[[VAL7:.+]] = arith.addf %[[VAL6]], %[[MIN]] : f32
// CHECK-NEXT:   linalg.yield %[[VAL7]] : f32
// CHECK-NEXT: -> tensor<8x1024x768xf32>

func.func @convert_rng_dynamic(%arg0: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> attributes {__placeholder__byre.entry_point} {
    %0 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %1 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = shape.shape_of %arg0 : tensor<?x?x?xf32> -> tensor<3xindex>
    %3 = arith.index_cast %2 : tensor<3xindex> to tensor<3xi64>
    %4 = byre.compute @GetSeed() -> tensor<i64>
    %5 = byre.compute @GetOffset() -> tensor<i64>
    %6 = mhlo.custom_call @byteir.rng_uniform(%1, %0, %4, %5, %3) {backend_config = ""} : (tensor<f32>, tensor<f32>, tensor<i64>, tensor<i64>, tensor<3xi64>) -> tensor<?x?x?xf32>
    return %6 : tensor<?x?x?xf32>
  }

// CHECK-LABEL: func.func @convert_rng_dynamic
// CHECK-DAG: arith.constant
// CHECK-DAG: arith.constant
// CHECK-DAG: byre.compute
// CHECK-DAG: byre.compute
// CHECK-DAG: arith.index_cast
// CHECK-DAG: arith.constant
// CHECK-DAG: tensor.extract
// CHECK-DAG: arith.constant
// CHECK-DAG: tensor.extract
// CHECK-DAG: arith.constant
// CHECK-DAG: tensor.extract
// CHECK-DAG: tensor.empty
// CHECK-NEXT: linalg.generic
// CHECK-DAG:  ^{{.*}}(%[[MIN:.+]]: f32, %[[MAX:.+]]: f32, %[[SEED:.+]]: i64, %[[OFFSET:.+]]: i64, %[[OUT:.+]]: f32
// CHECK-DAG:    %[[I32SEED:.+]] = arith.trunci %[[SEED]] : i64 to i32
// CHECK-DAG:    %[[I32OFFSET:.+]] = arith.trunci %[[OFFSET]] : i64 to i32
// CHECK-DAG:    %[[KSEED:.+]] = arith.addi %[[I32SEED]], %[[I32OFFSET]] : i32
// CHECK-DAG:    %[[CST0:.+]] = arith.constant 1103515245 : i32
// CHECK-DAG:    %[[CST1:.+]] = arith.constant 12345 : i32
// CHECK-DAG:    %[[CST2:.+]] = arith.constant 2.32830644E-10 : f32
// CHECK-DAG:    %[[IDX0:.+]] = linalg.index 0 : index
// CHECK-DAG:    %[[IDX0_CAST:.+]] = arith.index_cast %[[IDX0]] : index to i32
// CHECK-DAG:    %[[IDX1:.+]] = linalg.index 1 : index
// CHECK-DAG:    %[[IDX1_CAST:.+]] = arith.index_cast %[[IDX1]] : index to i32
// CHECK-DAG:    %[[IDX2:.+]] = linalg.index 2 : index
// CHECK-DAG:    %[[IDX2_CAST:.+]] = arith.index_cast %[[IDX2]] : index to i32
// CHECK-DAG:    %[[START:.+]] = arith.addi %[[IDX0_CAST]], %[[KSEED]] : i32
// CHECK-DAG:    %[[VAL1:.+]] = arith.muli %[[START]], %[[CST0]] : i32
// CHECK-DAG:    %[[VAL2:.+]] = arith.addi %[[VAL1]], %[[CST1]] : i32
// CHECK-DAG:    %[[VAL3:.+]] = arith.addi %[[IDX1_CAST]], %[[VAL2]] : i32
// CHECK-DAG:    %[[VAL4:.+]] = arith.muli %[[VAL3]], %[[CST0]] : i32
// CHECK-DAG:    %[[VAL5:.+]] = arith.addi %[[VAL4]], %[[CST1]] : i32
// CHECK-DAG:    %[[VAL6:.+]] = arith.addi %[[IDX2_CAST]], %[[VAL5]] : i32
// CHECK-DAG:    %[[VAL7:.+]] = arith.muli %[[VAL6]], %[[CST0]] : i32
// CHECK-DAG:    %[[VAL8:.+]] = arith.addi %[[VAL7]], %[[CST1]] : i32
// CHECK-DAG:    %[[DIFF:.+]] = arith.subf %[[MAX]], %[[MIN]] : f32
// CHECK-DAG:    %[[FACT:.+]] = arith.mulf %[[DIFF]], %[[CST2]] : f32
// CHECK-DAG:    %[[VAL8_CAST:.+]] = arith.uitofp %[[VAL8]] : i32 to f32
// CHECK-DAG:    %[[VAL6:.+]] = arith.mulf %[[VAL8_CAST]], %[[FACT]] : f32
// CHECK-DAG:    %[[VAL7:.+]] = arith.addf %[[VAL6]], %[[MIN]] : f32
// CHECK-NEXT:   linalg.yield %[[VAL7]] : f32
// CHECK-NEXT: -> tensor<?x?x?xf32>
