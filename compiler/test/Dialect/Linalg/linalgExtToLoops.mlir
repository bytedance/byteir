// RUN: byteir-opt %s -convert-linalg-ext-to-loops --split-input-file | FileCheck %s

func.func @scatter(%src: memref<2x3x32x64xf32>, %indices: memref<100x2xi64>, %update: memref<100x32x64xf32>) {
  linalg_ext.scatter
    ins(%indices, %update: memref<100x2xi64>, memref<100x32x64xf32>)
    outs(%src: memref<2x3x32x64xf32>) {
      ^bb0(%arg0: f32, %arg1: f32):
        %0 = arith.addf %arg0, %arg1 : f32
        linalg_ext.yield %0 : f32
    }
  return
}

// CHECK-LABEL: func @scatter
//  CHECK-SAME:     %[[SRC:[a-zA-Z0-9]+]]: memref<2x3x32x64xf32>
//  CHECK-SAME:     %[[INDICES:[a-zA-Z0-9]+]]: memref<100x2xi64>
//  CHECK-SAME:     %[[UPDATE:[a-zA-Z0-9]+]]: memref<100x32x64xf32>
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[C100:.+]] = arith.constant 100 : index
//   CHECK-DAG:   %[[C32:.+]] = arith.constant 32 : index
//   CHECK-DAG:   %[[C64:.+]] = arith.constant 64 : index
//       CHECK:   scf.for %[[IV0:[a-zA-Z0-9]+]] = %[[C0]] to %[[C100]] step %[[C1]]
//       CHECK:     scf.for %[[IV1:[a-zA-Z0-9]+]] = %[[C0]] to %[[C32]] step %[[C1]]
//       CHECK:       scf.for %[[IV2:[a-zA-Z0-9]+]] = %[[C0]] to %[[C64]] step %[[C1]]
//   CHECK-DAG:         %[[INDEX0:.+]] = memref.load %[[INDICES]][%[[IV0]], %[[C0]]]
//   CHECK-DAG:         %[[INDEX0_CAST:.+]] = arith.index_cast %[[INDEX0]] : i64 to index
//   CHECK-DAG:         %[[INDEX1:.+]] = memref.load %[[INDICES]][%[[IV0]], %[[C1]]]
//   CHECK-DAG:         %[[INDEX1_CAST:.+]] = arith.index_cast %[[INDEX1]] : i64 to index
//   CHECK-DAG:         %[[LHS:.+]] = memref.load %[[SRC]][%[[INDEX0_CAST]], %[[INDEX1_CAST]], %[[IV1]], %[[IV2]]]
//   CHECK-DAG:         %[[RHS:.+]] = memref.load %[[UPDATE]][%[[IV0]], %[[IV1]], %[[IV2]]]
//       CHECK:         %[[ADDF:.+]] = arith.addf %[[LHS]], %[[RHS]]
//       CHECK:         memref.store %[[ADDF]], %[[SRC]][%[[INDEX0_CAST]], %[[INDEX1_CAST]], %[[IV1]], %[[IV2]]]
