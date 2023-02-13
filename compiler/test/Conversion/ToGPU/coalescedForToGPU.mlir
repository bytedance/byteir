
// RUN: byteir-opt -affine-loop-coalescing -affine-simplify-structures -coalesced-for-to-gpu %s | FileCheck %s

// CHECK-LABEL: loop1
func.func @loop1(%arg0: memref<16xf32>, %arg1: memref<16xf32>, %arg2: memref<f32>, %arg3: memref<16xf32>) -> memref<16xui32> attributes {__byteir_elementwise_fusion__} {
  %0 = memref.alloc() : memref<16xi32>
  affine.for %arg4 = 0 to 16 {
    %4 = memref.load %arg0[%arg4] : memref<16xf32>
    %5 = memref.load %arg1[%arg4] : memref<16xf32>
    %6 = memref.load %arg2[] : memref<f32>
    %7 = memref.load %arg3[%arg4] : memref<16xf32>
    %8 = arith.mulf %6, %7 : f32
    %9 = arith.maxf %4, %5 : f32
    %10 = arith.minf %9, %8 : f32
    %11 = arith.fptosi %10 : f32 to i64
    %12 = arith.trunci %11 : i64 to i32
    memref.store %12, %0[%arg4] : memref<16xi32>
  }
  %1 = bufferization.to_tensor %0 : memref<16xi32>
  %2 = builtin.unrealized_conversion_cast %1 : tensor<16xi32> to tensor<16xui32>
  %3 = bufferization.to_memref %2 : memref<16xui32>
  return %3 : memref<16xui32>
}
// CHECK: gpu.launch
// CHECK-NOT: affine.for

// CHECK-LABEL: loop4
func.func @loop4(%arg0: memref<16x31x128xf32>, %arg1: memref<16x31x127x128xf32>) -> memref<16x31x127x128xi1> attributes {__byteir_elementwise_fusion__} {
  %0 = memref.alloc() : memref<16x31x127x128xi1>
  affine.for %arg2 = 0 to 16 {
    affine.for %arg3 = 0 to 31 {
      affine.for %arg4 = 0 to 127 {
        affine.for %arg5 = 0 to 128 {
          %1 = memref.load %arg1[%arg2, %arg3, %arg4, %arg5] : memref<16x31x127x128xf32>
          %2 = memref.load %arg0[%arg2, %arg3, %arg5] : memref<16x31x128xf32>
          %3 = arith.cmpf oeq, %1, %2 : f32
          memref.store %3, %0[%arg2, %arg3, %arg4, %arg5] : memref<16x31x127x128xi1>
        }
      }
    }
  }
  return %0 : memref<16x31x127x128xi1>
}
// CHECK: gpu.launch
// CHECK-NOT: affine.for