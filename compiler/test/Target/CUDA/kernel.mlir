// RUN: byteir-translate -emit-cuda %s | FileCheck %s -check-prefix=CUDA-DEFAULT
// RUN: byteir-translate -emit-cuda -force-extern-c %s | FileCheck %s -check-prefix=CUDA-EXTERN-C

module attributes {gpu.container_module}  {
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

  // CUDA-DEFAULT-LABEL: __global__ void fusion_broadcast_kernel
  // CUDA-DEFAULT-SAME: (float* [[V1:[^ ]*]], float* [[V2:[^ ]*]], float* [[V3:[^ ]*]])
  // CUDA-DEFAULT-NEXT: size_t [[V4:[^ ]*]] = blockIdx.x;
  // CUDA-DEFAULT-NEXT: size_t [[V5:[^ ]*]] = threadIdx.x;

  // CUDA-EXTERN-C-LABEL: extern "C" __global__ void fusion_broadcast_kernel
  // CUDA-EXTERN-C-SAME: (float* [[V1:[^ ]*]], float* [[V2:[^ ]*]], float* [[V3:[^ ]*]])
  // CUDA-EXTERN-C-NEXT: size_t [[V4:[^ ]*]] = blockIdx.x;
  // CUDA-EXTERN-C-NEXT: size_t [[V5:[^ ]*]] = threadIdx.x;
}

