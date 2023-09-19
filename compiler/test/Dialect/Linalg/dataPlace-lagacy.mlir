// RUN: byteir-opt %s -linalg-data-place="mem-spaces=1,2,3" -cse | FileCheck %s
// XFAIL: *

// CHECK-LABEL: func.func @matmul_static
func.func @matmul_static(%arg0: memref<128x64xf32>, %arg1: memref<64x64xf32>, %arg2: memref<128x64xf32>) {
  linalg.matmul {__byteir_data_place__} ins(%arg0, %arg1 : memref<128x64xf32>, memref<64x64xf32>) outs(%arg2 : memref<128x64xf32>)
// CHECK-DAG: %[[V0:.*]] = memref.alloc() : memref<128x64xf32, 1>
// CHECK-DAG: %[[V1:.*]] = memref.alloc() : memref<64x64xf32, 2>
// CHECK-DAG: %[[V2:.*]] = memref.alloc() : memref<128x64xf32, 3>
// CHECK-DAG: linalg.copy ins(%arg0 : memref<128x64xf32>) outs(%[[V0]] : memref<128x64xf32, 1>)
// CHECK-DAG: linalg.copy ins(%arg1 : memref<64x64xf32>) outs(%[[V1]] : memref<64x64xf32, 2>)
// CHECK-DAG: linalg.matmul ins(%[[V0]], %[[V1]] : memref<128x64xf32, 1>, memref<64x64xf32, 2>) outs(%[[V2]] : memref<128x64xf32, 3>)
// CHECK-DAG: linalg.copy ins(%[[V2]] : memref<128x64xf32, 3>) outs(%arg2 : memref<128x64xf32>)
  return
}

// CHECK-LABEL: func.func @matmul_static_completed_tag
func.func @matmul_static_completed_tag(%arg0: memref<128x64xf32>, %arg1: memref<64x64xf32>, %arg2: memref<128x64xf32>) {
  linalg.matmul {__byteir_data_place__ = [4, 5, 6]} ins(%arg0, %arg1 : memref<128x64xf32>, memref<64x64xf32>) outs(%arg2 : memref<128x64xf32>)
// CHECK-DAG: %[[V0:.*]] = memref.alloc() : memref<128x64xf32, 4>
// CHECK-DAG: %[[V1:.*]] = memref.alloc() : memref<64x64xf32, 5>
// CHECK-DAG: %[[V2:.*]] = memref.alloc() : memref<128x64xf32, 6>
// CHECK-DAG: linalg.copy ins(%arg0 : memref<128x64xf32>) outs(%[[V0]] : memref<128x64xf32, 4>)
// CHECK-DAG: linalg.copy ins(%arg1 : memref<64x64xf32>) outs(%[[V1]] : memref<64x64xf32, 5>)
// CHECK: linalg.matmul ins(%[[V0]], %[[V1]] : memref<128x64xf32, 4>, memref<64x64xf32, 5>) outs(%[[V2]] : memref<128x64xf32, 6>)
// CHECK-DAG: linalg.copy ins(%[[V2]] : memref<128x64xf32, 6>) outs(%arg2 : memref<128x64xf32>)
  return
}

// CHECK-LABEL: func.func @matmul_static_bad_tag
func.func @matmul_static_bad_tag(%arg0: memref<128x64xf32>, %arg1: memref<64x64xf32>, %arg2: memref<128x64xf32>) {
  linalg.matmul {__byteir_data_place__ = "another_attr"} ins(%arg0, %arg1 : memref<128x64xf32>, memref<64x64xf32>) outs(%arg2 : memref<128x64xf32>)
// CHECK: linalg.matmul {__byteir_data_place__ = "another_attr"} 
  return
}


// CHECK-LABEL: func.func @matmul_static_partial_tag
func.func @matmul_static_partial_tag(%arg0: memref<128x64xf32>, %arg1: memref<64x64xf32>, %arg2: memref<128x64xf32>) {
  linalg.matmul {__byteir_data_place__ = [4, 5, "another_attr"]} ins(%arg0, %arg1 : memref<128x64xf32>, memref<64x64xf32>) outs(%arg2 : memref<128x64xf32>)
// CHECK-DAG: %[[V0:.*]] = memref.alloc() : memref<128x64xf32, 4>
// CHECK-DAG: %[[V1:.*]] = memref.alloc() : memref<64x64xf32, 5>
// CHECK-DAG: linalg.copy ins(%arg0 : memref<128x64xf32>) outs(%[[V0]] : memref<128x64xf32, 4>)
// CHECK-DAG: linalg.copy ins(%arg1 : memref<64x64xf32>) outs(%[[V1]] : memref<64x64xf32, 5>)
// CHECK: linalg.matmul ins(%[[V0]], %[[V1]] : memref<128x64xf32, 4>, memref<64x64xf32, 5>) outs(%arg2 : memref<128x64xf32>)
  return
}

// CHECK-LABEL: func.func @matmul_dynamic
func.func @matmul_dynamic(%arg0: memref<128x?xf32>, %arg1: memref<?x64xf32>, %arg2: memref<128x64xf32>) {
  linalg.matmul {__byteir_data_place__} ins(%arg0, %arg1 : memref<128x?xf32>, memref<?x64xf32>) outs(%arg2 : memref<128x64xf32>)
// CHECK-DAG: %[[V0:.*]] = memref.alloc(%{{.*}}) : memref<128x?xf32, 1>
// CHECK-DAG: %[[V1:.*]] = memref.alloc(%{{.*}}) : memref<?x64xf32, 2>
// CHECK-DAG: %[[V2:.*]] = memref.alloc() : memref<128x64xf32, 3>
// CHECK-DAG: linalg.copy ins(%arg0 : memref<128x?xf32>) outs(%[[V0]] : memref<128x?xf32, 1>)
// CHECK-DAG: linalg.copy ins(%arg1 : memref<?x64xf32>) outs(%[[V1]] : memref<?x64xf32, 2>)
// CHECK: linalg.matmul ins(%[[V0]], %[[V1]] : memref<128x?xf32, 1>, memref<?x64xf32, 2>) outs(%[[V2]] : memref<128x64xf32, 3>)
// CHECK-DAG: linalg.copy ins(%[[V2]] : memref<128x64xf32, 3>) outs(%arg2 : memref<128x64xf32>)
  return
}

#map = affine_map<(d0, d1)[s0] -> (d0 * 64 + s0 + d1)>
// CHECK-LABEL: func.func @matmul_tiled
func.func @matmul_tiled(%arg0: memref<128x64xf32>, %arg1: memref<64x64xf32>, %arg2: memref<128x64xf32>) {
  %c128 = arith.constant 128 : index
  %c8 = arith.constant 8 : index
  %c0 = arith.constant 0 : index
  scf.for %arg3 = %c0 to %c128 step %c8 {
    %0 = memref.subview %arg0[%arg3, 0] [%c8, 64] [1, 1] : memref<128x64xf32> to memref<?x64xf32, #map>
    %1 = memref.subview %arg2[%arg3, 0] [%c8, 64] [1, 1] : memref<128x64xf32> to memref<?x64xf32, #map>
    linalg.matmul {__byteir_data_place__} ins(%0, %arg1 : memref<?x64xf32, #map>, memref<64x64xf32>) outs(%1 : memref<?x64xf32, #map>)
    // CHECK-DAG: %[[V0:.*]] = memref.alloc(%{{.*}}) : memref<?x64xf32, 1>
    // CHECK-DAG: %[[V1:.*]] = memref.alloc() : memref<64x64xf32, 2>
    // CHECK-DAG: %[[V2:.*]] = memref.alloc(%{{.*}}) : memref<?x64xf32, 3>
    // CHECK-DAG: linalg.copy ins(%{{.*}} : memref<?x64xf32, #map>) outs(%[[V0]] : memref<?x64xf32, 1>)
    // CHECK-DAG: linalg.copy ins(%arg1 : memref<64x64xf32>) outs(%[[V1]] : memref<64x64xf32, 2>)
    // CHECK: linalg.matmul ins(%[[V0]], %[[V1]] : memref<?x64xf32, 1>, memref<64x64xf32, 2>) outs(%[[V2]] : memref<?x64xf32, 3>)
    // CHECK-DAG: linalg.copy ins(%[[V2]] : memref<?x64xf32, 3>) outs(%{{.*}}: memref<?x64xf32, #map>)
  }
  return
}
