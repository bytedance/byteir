// RUN: byteir-translate -emit-cuda -kernel-only  %s | FileCheck %s

module attributes {gpu.container_module}  {
  func.func @fusion_broadcast(%arg0: memref<6x12x96xf32>, %arg1: memref<6x12x96x96xf32>) -> memref<6x12x96x96xf32> {
    %0 = memref.alloc() : memref<6x12x96x96xf32>
    %c0 = arith.constant 0 : index
    %c6 = arith.constant 6 : index
    %1 = arith.subi %c6, %c0 : index
    %c12 = arith.constant 12 : index
    %2 = arith.subi %c12, %c0 : index
    %c1 = arith.constant 1 : index
    gpu.launch_func  @fusion_broadcast_kernel::@fusion_broadcast_kernel blocks in (%1, %c1, %c1) threads in (%2, %c1, %c1) args(%arg1 : memref<6x12x96x96xf32>, %arg0 : memref<6x12x96xf32>, %0 : memref<6x12x96x96xf32>)
    return %0 : memref<6x12x96x96xf32>
  }
  gpu.module @fusion_broadcast_kernel {
    gpu.func @fusion_broadcast_kernel(%arg0: memref<6x12x96x96xf32>, %arg1: memref<6x12x96xf32>, %arg2: memref<6x12x96x96xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %c0 = arith.constant 0 : index
      %2 = arith.addi %c0, %0 : index
      %3 = arith.addi %c0, %1 : index
      %c96 = arith.constant 96 : index
      %c1 = arith.constant 1 : index
      scf.for %arg3 = %c0 to %c96 step %c1 {
        scf.for %arg4 = %c0 to %c96 step %c1 {
          %4 = memref.load %arg0[%2, %3, %arg3, %arg4] : memref<6x12x96x96xf32>
          %5 = memref.load %arg1[%2, %3, %arg3] : memref<6x12x96xf32>
          %6 = arith.subf %4, %5 : f32
          %7 = emitc.call_opaque "expf" (%6) : (f32) -> f32
          memref.store %7, %arg2[%2, %3, %arg3, %arg4] : memref<6x12x96x96xf32>
        }
      }
      gpu.return
    }
  }

  // CHECK-LABEL: __global__ void fusion_broadcast_kernel
  // CHECK-SAME: (float* [[V1:[^ ]*]], float* [[V2:[^ ]*]], float* [[V3:[^ ]*]])
  // CHECK-NEXT: size_t [[V4:[^ ]*]] = blockIdx.x;
  // CHECK-NEXT: size_t [[V5:[^ ]*]] = threadIdx.x;
}

