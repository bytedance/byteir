// RUN: byteir-opt %s -transform-dialect-interpreter="erase-after" | FileCheck %s

func.func @conv(%arg0 : memref<?x?xf32>, %arg1 : memref<?x?xf32>, %arg2 : memref<?x?xf32>) {
  linalg.conv_2d ins(%arg0, %arg1 : memref<?x?xf32>, memref<?x?xf32>) outs(%arg2 : memref<?x?xf32>)
  return
}

transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["linalg.conv_2d"]} in %arg1
    %1, %loop:2 = transform.structured.tile %0 [2, 3]
}

// CHECK-LABEL: func @conv
//       CHECK:   scf.for
//       CHECK:     scf.for
//   CHECK-NOT: transform.sequence
