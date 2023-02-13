// RUN: byteir-opt %s -convert-func-to-gpu -cse -canonicalize | FileCheck %s 

#map = affine_map<(d0, d1)[s0] -> (d0 * 64 + s0 + d1)>

func.func private @matmul_tiled_1(%arg0: memref<128x64xf32>, %arg1: memref<64x64xf32>, %arg2: memref<128x64xf32>) attributes {__byteir_to_gpu__} {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c128 = arith.constant 128 : index
  %0 = memref.alloc() : memref<8x64xf32, 1>
  %1 = memref.alloc() : memref<64x64xf32, 2>
  %2 = memref.alloc() : memref<8x64xf32, 3>
  scf.for %arg3 = %c0 to %c128 step %c8 {
    %3 = memref.subview %arg0[%arg3, 0] [8, 64] [1, 1] : memref<128x64xf32> to memref<8x64xf32, #map>
    %4 = memref.subview %arg2[%arg3, 0] [8, 64] [1, 1] : memref<128x64xf32> to memref<8x64xf32, #map>
    linalg.copy ins(%3 : memref<8x64xf32, #map>) outs(%0 : memref<8x64xf32, 1>)
    linalg.copy ins(%arg1 : memref<64x64xf32>) outs(%1 : memref<64x64xf32, 2>)
    linalg.matmul ins(%0, %1 : memref<8x64xf32, 1>, memref<64x64xf32, 2>) outs(%2 : memref<8x64xf32, 3>)
    linalg.copy ins(%2 : memref<8x64xf32, 3>) outs(%4 : memref<8x64xf32, #map>)
  } {__byteir_loop_to_simt__ = "block_id.x"}
  return
}

func.func @matmul_tiled_2(%arg0: memref<128x64xf32>, %arg1: memref<64x64xf32>, %arg2: memref<128x64xf32>) attributes {__byteir_to_gpu__} {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c128 = arith.constant 128 : index
  %0 = memref.alloc() : memref<8x64xf32, 1>
  %1 = memref.alloc() : memref<64x64xf32, 2>
  %2 = memref.alloc() : memref<8x64xf32, 3>
  scf.for %arg3 = %c0 to %c128 step %c8 {
    %3 = memref.subview %arg0[%arg3, 0] [8, 64] [1, 1] : memref<128x64xf32> to memref<8x64xf32, #map>
    %4 = memref.subview %arg2[%arg3, 0] [8, 64] [1, 1] : memref<128x64xf32> to memref<8x64xf32, #map>
    linalg.copy ins(%3 : memref<8x64xf32, #map>) outs(%0 : memref<8x64xf32, 1>)
    linalg.copy ins(%arg1 : memref<64x64xf32>) outs(%1 : memref<64x64xf32, 2>)
    linalg.matmul ins(%0, %1 : memref<8x64xf32, 1>, memref<64x64xf32, 2>) outs(%2 : memref<8x64xf32, 3>)
    linalg.copy ins(%2 : memref<8x64xf32, 3>) outs(%4 : memref<8x64xf32, #map>)
  } {__byteir_coarsen_simt__, __byteir_loop_to_simt__ = "block_id.y"}
  return
}

// CHECK-LABEL:  gpu.module @unified
// CHECK-DAG{Matmul1}:  gpu.func @matmul_tiled_1
// CHECK{Matmul1}:    gpu.block_id  x
// CHECK{Matmul1}:    scf.if

// CHECK-DAG{Matmul2}:  gpu.func @matmul_tiled_2
// CHECK{Matmul2}:    gpu.block_id  y
// CHECK{Matmul2}:    scf.for

// CHECK-LABEL:  func.func private @matmul_tiled_1
// CHECK:    gpu.launch_func  @unified::@matmul_tiled_1

// CHECK-LABEL:  func.func @matmul_tiled_2
// CHECK:    gpu.launch_func  @unified::@matmul_tiled_2

func.func @test_call_unchanged(%arg0: memref<128x64xf32>, %arg1: memref<64x64xf32>, %arg2: memref<128x64xf32>) {
  call @matmul_tiled_1(%arg0, %arg1, %arg2) : (memref<128x64xf32>, memref<64x64xf32>, memref<128x64xf32>) -> ()
  return
}
// CHECK-LABEL:  func.func @test_call_unchanged
// CHECK:    call @matmul_tiled_1


