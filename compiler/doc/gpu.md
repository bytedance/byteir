# ByteIR GPU Support

ByteIR Compiler currently has limited support for NVIDIA GPU.

## Passes

The major route from mhlo to GPU backends is through mhlo dialect, linalg-tensor dialect, linalg-memref dialect, affine/scf dialect, gpu dialect. 
The first half, from mhlo to affine/scf dialect, is similar across backends. 
Therefore, we only discuss the second half, from affine/scf dialect to gpu dialect, later.
ByteIR Compiler has developed multiple passes to support GPU backends. 

### InsertTrivialSCFLoop Pass

This pass simply inserts a trivial scf ForOp for scalar computation.
It is typically used to simplify a pass pipeline without checking whether scalar computation later. 
Note the effect of this pass will be removed after the scf canonicalizer.  

```
// input.mlir
func.func @scalar_func(%arg0: memref<f32>) -> memref<f32> {
  %cst = arith.constant 1.000000e+00 : f32
  %alloc = memref.alloc() : memref<f32>
  %cst_0 = arith.constant 0.000000e+00 : f32
  %0 = memref.load %arg0[] : memref<f32>
  %1 = arith.cmpf une, %0, %cst_0 : f32
  %2 = arith.select %1, %0, %cst : f32
  memref.store %2, %alloc[] : memref<f32>
  return %alloc : memref<f32>
}

// result after -insert-trivial-scf-loop
func.func @scalar_func(%arg0: memref<f32>) -> memref<f32> {
  %cst = arith.constant 1.000000e+00 : f32
  %alloc = memref.alloc() : memref<f32>
  %cst_0 = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %arg1 = %c0 to %c1 step %c1 {
    %0 = memref.load %arg0[] : memref<f32>
    %1 = arith.cmpf une, %0, %cst_0 : f32
    %2 = arith.select %1, %0, %cst : f32
    memref.store %2, %alloc[] : memref<f32>
  }
  return %alloc : memref<f32>
}

```

### ConvertFuncToGPU Pass
This pass convert a FuncOp in a loop form into a GPUFuncOp in a SIMT form. 
Some loops with annotation in the source FuncOp are transformed into a SIMT statement. 

```
// input.mlir
#map = affine_map<(d0, d1)[s0] -> (d0 * 64 + s0 + d1)>
func.func private @matmul_tiled(%arg0: memref<128x64xf32>, %arg1: memref<64x64xf32>, %arg2: memref<128x64xf32>) attributes {__byteir_to_gpu__} {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c128 = arith.constant 128 : index
  %alloc = memref.alloc() : memref<8x64xf32, 1>
  %alloc_0 = memref.alloc() : memref<64x64xf32, 2>
  %alloc_1 = memref.alloc() : memref<8x64xf32, 3>
  scf.for %arg3 = %c0 to %c128 step %c8 {
    %subview = memref.subview %arg0[%arg3, 0] [8, 64] [1, 1] : memref<128x64xf32> to memref<8x64xf32, #map>
    %subview_2 = memref.subview %arg2[%arg3, 0] [8, 64] [1, 1] : memref<128x64xf32> to memref<8x64xf32, #map>
    linalg.copy ins(%subview : memref<8x64xf32, #map>) outs(%alloc : memref<8x64xf32, 1>)
    linalg.copy ins(%arg1 : memref<64x64xf32>) outs(%alloc_0 : memref<64x64xf32, 2>)
    linalg.matmul ins(%alloc, %alloc_0 : memref<8x64xf32, 1>, memref<64x64xf32, 2>) outs(%alloc_1 : memref<8x64xf32, 3>)
    linalg.copy ins(%alloc_1 : memref<8x64xf32, 3>) outs(%subview_2 : memref<8x64xf32, #map>)
  } {__byteir_loop_to_simt__ = "block_id.x"}
  return
}

// result after convert-func-to-gpu
gpu.module @unified {
gpu.func @matmul_tiled(%arg0: memref<128x64xf32>, %arg1: memref<64x64xf32>, %arg2: memref<128x64xf32>) workgroup(%arg3 : memref<8x64xf32, 3>, %arg4 : memref<64x64xf32, 3>, %arg5 : memref<8x64xf32, 3>) kernel {
  %c128 = arith.constant 128 : index
  %c8 = arith.constant 8 : index
  %0 = gpu.block_id  x
  %alloc = memref.alloc() : memref<8x64xf32, 3>
  %alloc_0 = memref.alloc() : memref<64x64xf32, 2>
  %1 = arith.muli %0, %c8 : index
  %2 = arith.cmpi slt, %1, %c128 : index
  scf.if %2 {
    %subview = memref.subview %arg0[%1, 0] [8, 64] [1, 1] : memref<128x64xf32> to memref<8x64xf32, #map>
    %subview_1 = memref.subview %arg2[%1, 0] [8, 64] [1, 1] : memref<128x64xf32> to memref<8x64xf32, #map>
    linalg.copy ins(%subview : memref<8x64xf32, #map>) outs(%arg5 : memref<8x64xf32, 3>)
    linalg.copy ins(%arg1 : memref<64x64xf32>) outs(%alloc_0 : memref<64x64xf32, 2>)
    linalg.matmul ins(%arg5, %alloc_0 : memref<8x64xf32, 3>, memref<64x64xf32, 2>) outs(%alloc : memref<8x64xf32, 3>)
    linalg.copy ins(%alloc : memref<8x64xf32, 3>) outs(%subview_1 : memref<8x64xf32, #map>)
  }
  gpu.return
}
}
func.func private @matmul_tiled(%arg0: memref<128x64xf32>, %arg1: memref<64x64xf32>, %arg2: memref<128x64xf32>) {
  %c16 = arith.constant 16 : index
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  gpu.launch_func  @unified::@matmul_tiled blocks in (%c16, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<128x64xf32>, %arg1 : memref<64x64xf32>, %arg2 : memref<128x64xf32>)
  return
}
```

## Backend Target

ByteIR Compiler supports NVIDIA GPU backends through either LLVM PTX codegen or a CUDA C source code emitter. 

Other targets, such as NVIDIA official [NVVM compiler](https://docs.nvidia.com/cuda/nvvm-ir-spec/) or LLVM cubin codegen, will be the future work. 

### LLVM PTX codegen

LLVM PTX codegen lowers GPU dialect into LLVM/NVVM dialect, and then translates LLVM/NVVM dialect to PTX using LLVM PTX backend. 
The first step relies on a common pipeline `NVVMCodegenPipeline`, while the second step uses `byteir-translate` with a `gen-ptx` option.

### CUDA emitter

CUDA emitter directly translates GPU dialect to CUDA C source code using `byteir-translate` with an `emit-cuda` option.

