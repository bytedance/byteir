// RUN: byteir-opt %s -unroll="unroll-full" -cse -canonicalize-ext | FileCheck %s -check-prefix=UNROLLFULL
// RUN: byteir-opt %s -unroll="unroll-factor=3" -cse  -canonicalize-ext | FileCheck %s -check-prefix=UNROLL3

func.func @two_tripcount(%arg0 : memref<?xf32>) {
  %0 = arith.constant 7.0 : f32
  %lb = arith.constant 0 : index
  %ub = arith.constant 63 : index
  %step = arith.constant 32 : index
  scf.for %i0 = %lb to %ub step %step {
    memref.store %0, %arg0[%i0] : memref<?xf32>
  } {__byteir_unroll__}
  return
}
// UNROLLFULL-LABEL: func.func @two_tripcount
// UNROLLFULL-SAME: (%[[ARG0:[a-zA-Z0-9]+]]: memref<?xf32>)
// UNROLLFULL: %[[Vcst:.*]] = arith.constant 7.000000e+00 : f32
// UNROLLFULL: %[[V0:.*]] = arith.constant 0 : index
// UNROLLFULL: %[[V32:.*]] = arith.constant 32 : index
// UNROLLFULL: memref.store %[[Vcst]], %[[ARG0]][%[[V0]]] : memref<?xf32>
// UNROLLFULL: memref.store %[[Vcst]], %[[ARG0]][%[[V32]]] : memref<?xf32>

// UNROLL3-LABEL: func.func @two_tripcount
// UNROLL3: scf.for
// UNROLL3:   memref.store


func.func @many_tripcount(%arg0 : memref<?xf32>) {
  %0 = arith.constant 7.0 : f32
  %lb = arith.constant 0 : index
  %ub = arith.constant 14 : index
  %step = arith.constant 2 : index
  scf.for %i0 = %lb to %ub step %step {
    memref.store %0, %arg0[%i0] : memref<?xf32>
  } {__byteir_unroll__}
  return
}
// UNROLLFULL-LABEL: func.func @many_tripcount
// UNROLLFULL-SAME: (%[[ARG0:[a-zA-Z0-9]+]]: memref<?xf32>)
// UNROLLFULL-DAG: %[[V4:.*]] = arith.constant 4 : index
// UNROLLFULL-DAG: %[[V6:.*]] = arith.constant 6 : index
// UNROLLFULL-DAG: %[[V8:.*]] = arith.constant 8 : index
// UNROLLFULL-DAG: %[[V10:.*]] = arith.constant 10 : index
// UNROLLFULL-DAG: %[[V12:.*]] = arith.constant 12 : index
// UNROLLFULL-DAG: %[[Vcst:.*]] = arith.constant 7.000000e+00 : f32
// UNROLLFULL-DAG: %[[V0:.*]] = arith.constant 0 : index
// UNROLLFULL-DAG: %[[V2:.*]] = arith.constant 2 : index
// UNROLLFULL: memref.store %[[Vcst]], %[[ARG0]][%[[V0]]] : memref<?xf32>
// UNROLLFULL: memref.store %[[Vcst]], %[[ARG0]][%[[V2]]] : memref<?xf32>
// UNROLLFULL: memref.store %[[Vcst]], %[[ARG0]][%[[V4]]] : memref<?xf32>
// UNROLLFULL: memref.store %[[Vcst]], %[[ARG0]][%[[V6]]] : memref<?xf32>
// UNROLLFULL: memref.store %[[Vcst]], %[[ARG0]][%[[V8]]] : memref<?xf32>
// UNROLLFULL: memref.store %[[Vcst]], %[[ARG0]][%[[V10]]] : memref<?xf32>
// UNROLLFULL: memref.store %[[Vcst]], %[[ARG0]][%[[V12]]] : memref<?xf32>


// UNROLL3-LABEL: func.func @many_tripcount
// UNROLL3-SAME: (%[[ARG0:[a-zA-Z0-9]+]]: memref<?xf32>)
// UNROLL3-DAG: %[[V4:.*]] = arith.constant 4 : index
// UNROLL3-DAG: %[[Vcst:.*]] = arith.constant 7.000000e+00 : f32
// UNROLL3-DAG: %[[V0:.*]] = arith.constant 0 : index
// UNROLL3-DAG: %[[V2:.*]] = arith.constant 2 : index
// UNROLL3-DAG: %[[V12:.*]] = arith.constant 12 : index
// UNROLL3-DAG: %[[V6:.*]] = arith.constant 6 : index
// UNROLL3: scf.for
// UNROLL3-SAME: %[[ARG1:[a-zA-Z0-9]+]] = %[[V0]] to %[[V12]] step %[[V6]] {
// UNROLL3:   memref.store %[[Vcst]], %[[ARG0]][%[[ARG1]]] : memref<?xf32>
// UNROLL3: %[[C0:.*]] = arith.addi %[[ARG1]], %[[V2]] : index
// UNROLL3:   memref.store %[[Vcst]], %[[ARG0]][%[[C0]]] : memref<?xf32>
// UNROLL3: %[[C1:.*]] = arith.addi %[[ARG1]], %[[V4]] : index
// UNROLL3:   memref.store %[[Vcst]], %[[ARG0]][%[[C1]]] : memref<?xf32>
// UNROLL3: }
// UNROLL3: memref.store %[[Vcst]], %[[ARG0]][%[[V12]]] : memref<?xf32>

#map = affine_map<(d0) -> (-d0 + 63, 32)>
func.func @fuse_unary(%arg0: tensor<63x128xf32>, %arg1: tensor<63x128xf32>) -> tensor<63x128xf32> {
  %c32 = arith.constant 32 : index
  %c0 = arith.constant 0 : index
  %c63 = arith.constant 63 : index
  %0 = scf.for %arg2 = %c0 to %c63 step %c32 iter_args(%arg3 = %arg1) -> (tensor<63x128xf32>) {
    %1 = affine.min #map(%arg2)
    %extracted_slice = tensor.extract_slice %arg0[%arg2, 0] [%1, 128] [1, 1] : tensor<63x128xf32> to tensor<?x128xf32>
    %extracted_slice_0 = tensor.extract_slice %arg3[%arg2, 0] [%1, 128] [1, 1] : tensor<63x128xf32> to tensor<?x128xf32>
    %2 = linalg.elemwise_unary ins(%extracted_slice : tensor<?x128xf32>) outs(%extracted_slice_0 : tensor<?x128xf32>) -> tensor<?x128xf32>
    %inserted_slice = tensor.insert_slice %2 into %arg3[%arg2, 0] [%1, 128] [1, 1] : tensor<?x128xf32> into tensor<63x128xf32>
    scf.yield %inserted_slice : tensor<63x128xf32>
  } {__byteir_unroll__}
  return %0 : tensor<63x128xf32>
}

// UNROLLFULL-LABEL: func.func @fuse_unary
// UNROLLFULL: tensor.extract_slice
// UNROLLFULL: tensor.extract_slice
// UNROLLFULL: linalg.elemwise_unary
// UNROLLFULL-SAME: tensor<32x128xf32>) -> tensor<32x128xf32>
// UNROLLFULL: tensor.insert_slice
// UNROLLFULL: tensor.extract_slice
// UNROLLFULL: tensor.extract_slice
// UNROLLFULL: linalg.elemwise_unary
// UNROLLFULL-SAME: tensor<31x128xf32>) -> tensor<31x128xf32>
// UNROLLFULL: tensor.insert_slice
