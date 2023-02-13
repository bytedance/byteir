// RUN: byteir-opt -test-print-use-range -split-input-file %s | FileCheck %s

// CHECK-LABEL: Testing : func_empty
func.func @func_empty() {
  return
}
//      CHECK:  ---- UserangeAnalysis -----
// CHECK-NEXT:  ---- Id and Operation Begins -----
// CHECK-NEXT:  ---- Id and Operation Ends -----
// CHECK-NEXT:  ---------------------------

// -----

// CHECK-LABEL: Testing : useRangeGap
// CHECK: ---- Id and Operation Begins -----
// CHECK-NEXT: ID: 7, Op: "lmhlo.negate"(%arg1, %alloc)
// CHECK-NEXT: ID: 9, Op: "lmhlo.negate"(%arg1, %alloc_0)
// CHECK-NEXT: ID: 13, Op: "lmhlo.negate"(%arg2, %alloc)
// CHECK-NEXT: ID: 15, Op: "lmhlo.negate"(%arg2, %alloc_0)
func.func @useRangeGap(%arg0: i1, %arg1: memref<2xf32>, %arg2: memref<2xf32>)
{
  %0 = memref.alloc() : memref<2xf32>
  %1 = memref.alloc() : memref<2xf32>
  cf.cond_br %arg0, ^bb1, ^bb2
^bb1:
  "lmhlo.negate"(%arg1, %0) : (memref<2xf32>, memref<2xf32>) -> ()
  "lmhlo.negate"(%arg1, %1) : (memref<2xf32>, memref<2xf32>) -> ()
  cf.br ^bb3
^bb2:
  "lmhlo.negate"(%arg2, %0) : (memref<2xf32>, memref<2xf32>) -> ()
  "lmhlo.negate"(%arg2, %1) : (memref<2xf32>, memref<2xf32>) -> ()
  cf.br ^bb3
^bb3:
  return
}
//      CHECK:  Value: %alloc {{ *}}
// CHECK-NEXT:  Userange: {(7, 7), (13, 13)}
//      CHECK:  Value: %alloc_0 {{ *}}
// CHECK-NEXT:  Userange: {(9, 9), (15, 15)}

// -----

// CHECK-LABEL: Testing : useRangeGapWithView
func.func @useRangeGapWithView(%arg0: i1, %arg1: memref<2xf32>, %arg2: memref<2xf32>)
{
  %c0 = arith.constant 0 : index
  %0 = memref.alloc() : memref<2xf32>
  %1 = memref.alloc() : memref<8xi8>
  %2 = memref.view %1[%c0][] : memref<8xi8> to memref<2xf32>
  cf.cond_br %arg0, ^bb1, ^bb2
^bb1:
  "lmhlo.negate"(%arg1, %0) : (memref<2xf32>, memref<2xf32>) -> ()
  "lmhlo.negate"(%arg1, %2) : (memref<2xf32>, memref<2xf32>) -> ()
  cf.br ^bb3
^bb2:
  "lmhlo.negate"(%arg2, %0) : (memref<2xf32>, memref<2xf32>) -> ()
  "lmhlo.negate"(%arg2, %2) : (memref<2xf32>, memref<2xf32>) -> ()
  cf.br ^bb3
^bb3:
  return
}
//      CHECK:  Value: %alloc {{ *}}
// CHECK-NEXT:  Userange: {(11, 11), (17, 17)}
//      CHECK:  Value: %alloc_0 {{ *}}
// CHECK-NEXT:  Userange: {(7, 7), (11, 13), (17, 19), (7, 9)}
//      CHECK:  Value: %view {{ *}}
// CHECK-NEXT:  Userange: {(11, 13), (17, 19), (7, 9)}

// -----

// CHECK-LABEL: Testing : loopWithNestedRegion
func.func @loopWithNestedRegion(%arg0: i1, %arg1: memref<2xf32>, %arg2: memref<2xf32>)
{
  %0 = memref.alloc() : memref<2xf32>
  %1 = memref.alloc() : memref<2xf32>
  %2 = memref.alloc() : memref<2xf32>
  %3 = memref.alloc() : memref<2xf32>
  cf.br ^bb1
^bb1:
  %4 = scf.if %arg0 -> (memref<2xf32>) {
    "lmhlo.negate"(%arg1, %0) : (memref<2xf32>, memref<2xf32>) -> ()
    scf.yield %2 : memref<2xf32>
  } else {
    "lmhlo.negate"(%arg1, %1) : (memref<2xf32>, memref<2xf32>) -> ()
    scf.yield %2 : memref<2xf32>
  }
  cf.br ^bb2
^bb2:
  cf.cond_br %arg0, ^bb1, ^bb3
^bb3:
  "lmhlo.negate"(%arg1, %2) : (memref<2xf32>, memref<2xf32>) -> ()
  "lmhlo.negate"(%arg1, %3) : (memref<2xf32>, memref<2xf32>) -> ()
  return
}
//      CHECK:  Value: %alloc {{ *}}
// CHECK-NEXT:  Userange: {(11, 23)}
//      CHECK:  Value: %alloc_0 {{ *}}
// CHECK-NEXT:  Userange: {(11, 23)}
//      CHECK:  Value: %alloc_1 {{ *}}
// CHECK-NEXT:  Userange: {(11, 25)}
//      CHECK:  Value: %alloc_2 {{ *}}
// CHECK-NEXT:  Userange: {(27, 27)}
//      CHECK:  Value: %0 {{ *}}
//      CHECK:  Userange: {(19, 19)}

// -----

// CHECK-LABEL: Testing : condBranchWithAlias
func.func @condBranchWithAlias(%arg0: i1, %arg1: memref<2xf32>, %arg2: memref<2xf32>)
{
  %0 = memref.alloc() : memref<2xf32>
  cf.cond_br %arg0, ^bb1, ^bb2
^bb1:
  "lmhlo.negate"(%arg1, %0) : (memref<2xf32>, memref<2xf32>) -> ()
  cf.br ^bb3(%0 : memref<2xf32>)
^bb2:
  %1 = memref.alloc() : memref<2xf32>
  "lmhlo.negate"(%arg1, %1) : (memref<2xf32>, memref<2xf32>) -> ()
  cf.br ^bb3(%1 : memref<2xf32>)
^bb3(%2 : memref<2xf32>):
  %3 = memref.alloc() : memref<2xf32>
  "lmhlo.copy"(%2, %arg2) : (memref<2xf32>, memref<2xf32>) -> ()
  "lmhlo.copy"(%3, %arg2) : (memref<2xf32>, memref<2xf32>) -> ()
  %4 = memref.alloc() : memref<2xf32>
  "lmhlo.copy"(%4, %arg2) : (memref<2xf32>, memref<2xf32>) -> ()
  cf.br ^bb4(%0 : memref<2xf32>)
^bb4(%5 : memref<2xf32>):
  "lmhlo.copy"(%5, %arg2) : (memref<2xf32>, memref<2xf32>) -> ()
  return
}
//      CHECK:  Value: %alloc {{ *}}
// CHECK-NEXT:  Userange: {(5, 7), (15, 27)}
//      CHECK:  Value: %alloc_0 {{ *}}
// CHECK-NEXT:  Userange: {(11, 17)}
//      CHECK:  Value: %alloc_1 {{ *}}
// CHECK-NEXT:  Userange: {(19, 19)}
//      CHECK:  Value: %alloc_2 {{ *}}
// CHECK-NEXT:  Userange: {(23, 23)}
//      CHECK:  Value: <block argument> of type 'memref<2xf32>' at index: 0
// CHECK-SAME:  Userange: {(15, 17)}
//      CHECK:  Value: <block argument> of type 'memref<2xf32>' at index: 0
// CHECK-SAME:  Userange: {(27, 27)}
