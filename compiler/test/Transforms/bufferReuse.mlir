// RUN: byteir-opt %s -buffer-reuse | FileCheck %s

func.func @lmhlo_many_add(%arg0 : memref<4xf32>, %arg1 : memref<4xf32>, %arg2 : memref<4xf32>) {
  %0 = memref.alloc() : memref<4xf32>
  %1 = memref.alloc() : memref<4xf32>
  %2 = memref.alloc() : memref<4xf32>
  %3 = memref.alloc() : memref<4xf32>
  "lmhlo.add"(%arg0, %arg1, %0) : (memref<4xf32>, memref<4xf32>,  memref<4xf32>) -> ()
  "lmhlo.add"(%arg1, %0, %1) : (memref<4xf32>, memref<4xf32>,  memref<4xf32>) -> ()
  "lmhlo.add"(%arg1, %1, %2) : (memref<4xf32>, memref<4xf32>,  memref<4xf32>) -> ()
  "lmhlo.add"(%arg1, %2, %3) : (memref<4xf32>, memref<4xf32>,  memref<4xf32>) -> ()
  "lmhlo.add"(%arg1, %3, %arg2) : (memref<4xf32>, memref<4xf32>,  memref<4xf32>) -> ()
  return 
}
// CHECK-LABEL: func.func @lmhlo_many_add
// CHECK:   %[[VAR_0:.*]] = memref.alloc() : memref<4xf32>
// CHECK:   %[[VAR_1:.*]] = memref.alloc() : memref<4xf32>
// CHECK:   "lmhlo.add"(%{{.*}}, %{{.*}}, %[[VAR_0]]) : (memref<4xf32>, memref<4xf32>, memref<4xf32>) -> ()
// CHECK:   "lmhlo.add"(%{{.*}}, %[[VAR_0]], %[[VAR_1]]) : (memref<4xf32>, memref<4xf32>, memref<4xf32>) -> ()
// CHECK:   "lmhlo.add"(%{{.*}}, %[[VAR_1]], %[[VAR_0]]) : (memref<4xf32>, memref<4xf32>, memref<4xf32>) -> ()
// CHECK:   "lmhlo.add"(%{{.*}}, %[[VAR_0]], %[[VAR_1]]) : (memref<4xf32>, memref<4xf32>, memref<4xf32>) -> ()
// CHECK:   "lmhlo.add"(%{{.*}}, %[[VAR_1]], %{{.*}}) : (memref<4xf32>, memref<4xf32>, memref<4xf32>) -> ()
