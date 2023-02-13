// RUN: byteir-stat -alloc-cnt %s | FileCheck %s

#map0 = affine_map<(d0, d1) -> (d0 * 8 + d1)>

func.func @test_basic(%arg0 : memref<256xf32>, %arg1 : memref<256xf32>) -> memref<256xf32> {
  %0 = memref.alloc() : memref<256xf32>
  %1 = memref.alloc() : memref<256xf32>
  %2 = memref.alloc() : memref<256xf32>
  %3 = memref.alloc() : memref<256xf32>
  %4 = memref.alloc() : memref<256xf32>
  "lmhlo.add"(%arg0, %arg1, %0) : (memref<256xf32>, memref<256xf32>,  memref<256xf32>) -> ()
  "lmhlo.add"(%arg1, %0, %1) : (memref<256xf32>, memref<256xf32>,  memref<256xf32>) -> ()
  "lmhlo.add"(%arg1, %1, %2) : (memref<256xf32>, memref<256xf32>,  memref<256xf32>) -> ()
  "lmhlo.add"(%arg1, %2, %3) : (memref<256xf32>, memref<256xf32>,  memref<256xf32>) -> ()
  "lmhlo.add"(%arg1, %3, %4) : (memref<256xf32>, memref<256xf32>,  memref<256xf32>) -> ()
  return %4 : memref<256xf32>
}

// CHECK-LABEL: test_basic
//   CHECK: nr_static_allocation = 5
//   CHECK: nr_dynamic_allocation = 0
//   CHECK: total_static_allocated_memory = 5120
//   CHECK: peak_static_memory = 2048

func.func @test_strided(%arg0 : memref<16x3xf32, #map0>, %arg1 : memref<16x3xf32, #map0>) -> memref<16x3xf32, #map0> {
  %0 = memref.alloc() : memref<16x3xf32, #map0>
  %1 = memref.alloc() : memref<16x3xf32, #map0>
  %2 = memref.alloc() : memref<16x3xf32, #map0>
  %3 = memref.alloc() : memref<16x3xf32, #map0>
  %4 = memref.alloc() : memref<16x3xf32, #map0>
  "lmhlo.add"(%arg0, %arg1, %0) : (memref<16x3xf32, #map0>, memref<16x3xf32, #map0>,  memref<16x3xf32, #map0>) -> ()
  "lmhlo.add"(%arg1, %0, %1) : (memref<16x3xf32, #map0>, memref<16x3xf32, #map0>,  memref<16x3xf32, #map0>) -> ()
  "lmhlo.add"(%arg1, %1, %2) : (memref<16x3xf32, #map0>, memref<16x3xf32, #map0>,  memref<16x3xf32, #map0>) -> ()
  "lmhlo.add"(%arg1, %2, %3) : (memref<16x3xf32, #map0>, memref<16x3xf32, #map0>,  memref<16x3xf32, #map0>) -> ()
  "lmhlo.add"(%arg1, %3, %4) : (memref<16x3xf32, #map0>, memref<16x3xf32, #map0>,  memref<16x3xf32, #map0>) -> ()
  return %4 : memref<16x3xf32, #map0>
}
// CHECK-LABEL: test_strided
//   CHECK: nr_static_allocation = 5
//   CHECK: nr_dynamic_allocation = 0
//   CHECK: total_static_allocated_memory = 2560
//   CHECK: peak_static_memory = 1024
