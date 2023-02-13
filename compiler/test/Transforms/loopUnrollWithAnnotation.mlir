// RUN: byteir-opt %s -unroll="unroll-all=true unroll-full=true annotate-idx=true" -cse -split-input-file | FileCheck %s -check-prefix=UNROLLALL
// RUN: byteir-opt %s -unroll="depth=1 unroll-full=true annotate-idx=true" -cse -split-input-file | FileCheck %s -check-prefix=UNROLLONE
// RUN: byteir-opt %s -unroll="depth=2 unroll-full=true annotate-idx=true" -cse -split-input-file | FileCheck %s -check-prefix=UNROLLTWO
// RUN: byteir-opt %s -unroll="depth=1 unroll-full=true annotate-idx=true" -unroll="depth=0 unroll-full=true annotate-idx=true" -cse -split-input-file | FileCheck %s -check-prefix=TWOSTEPUNROLL

func.func @nested_binary_2x2(%arg0: tensor<1024x512xf32>, %arg1: tensor<1024x512xf32>, %arg2: tensor<1024x512xf32>) -> tensor<1024x512xf32> {
  %c1024 = arith.constant 1024 : index
  %c0 = arith.constant 0 : index
  %c512 = arith.constant 512 : index
  %c256 = arith.constant 256 : index
  %0 = tensor.empty() : tensor<1024x512xf32>
  %1 = scf.for %arg3 = %c0 to %c1024 step %c512 iter_args(%arg4 = %0) -> (tensor<1024x512xf32>) {
    %2 = scf.for %arg5 = %c0 to %c512 step %c256 iter_args(%arg6 = %0) -> (tensor<1024x512xf32>) {
      %extracted_slice = tensor.extract_slice %arg2[%arg3, %arg5] [512, 256] [1, 1] : tensor<1024x512xf32> to tensor<512x256xf32>
      %extracted_slice_0 = tensor.extract_slice %arg0[%arg3, %arg5] [512, 256] [1, 1] : tensor<1024x512xf32> to tensor<512x256xf32>
      %extracted_slice_1 = tensor.extract_slice %arg1[%arg3, %arg5] [512, 256] [1, 1] : tensor<1024x512xf32> to tensor<512x256xf32>
      %3 = tensor.empty() : tensor<512x256xf32>
      %4 = linalg.elemwise_binary ins(%extracted_slice_0, %extracted_slice_1 : tensor<512x256xf32>, tensor<512x256xf32>) outs(%3 : tensor<512x256xf32>) -> tensor<512x256xf32>
      %5 = tensor.empty() : tensor<512x256xf32>
      %6 = linalg.elemwise_binary {__root__} ins(%extracted_slice, %4 : tensor<512x256xf32>, tensor<512x256xf32>) outs(%5 : tensor<512x256xf32>) -> tensor<512x256xf32>
      %inserted_slice = tensor.insert_slice %6 into %0[%arg3, %arg5] [512, 256] [1, 1] : tensor<512x256xf32> into tensor<1024x512xf32>
      scf.yield %inserted_slice : tensor<1024x512xf32>
    } {__byteir_parallel__}
    scf.yield %2 : tensor<1024x512xf32>
  } {__byteir_parallel__}
  return %1 : tensor<1024x512xf32>
}

// UNROLLALL-LABEL: func.func @nested_binary_2x2
// UNROLLALL: linalg.elemwise_binary {__byteir_loop_step__ = 0 : i32, __byteir_loop_total_step__ = 4 : i32}
// UNROLLALL: linalg.elemwise_binary {__byteir_loop_step__ = 0 : i32, __byteir_loop_total_step__ = 4 : i32, __root__}
// UNROLLALL: linalg.elemwise_binary {__byteir_loop_step__ = 1 : i32, __byteir_loop_total_step__ = 4 : i32}
// UNROLLALL: linalg.elemwise_binary {__byteir_loop_step__ = 1 : i32, __byteir_loop_total_step__ = 4 : i32, __root__}
// UNROLLALL: linalg.elemwise_binary {__byteir_loop_step__ = 2 : i32, __byteir_loop_total_step__ = 4 : i32}
// UNROLLALL: linalg.elemwise_binary {__byteir_loop_step__ = 2 : i32, __byteir_loop_total_step__ = 4 : i32, __root__}
// UNROLLALL: linalg.elemwise_binary {__byteir_loop_step__ = 3 : i32, __byteir_loop_total_step__ = 4 : i32}
// UNROLLALL: linalg.elemwise_binary {__byteir_loop_step__ = 3 : i32, __byteir_loop_total_step__ = 4 : i32, __root__}


// UNROLLONE-LABEL: func.func @nested_binary_2x2
// UNROLLONE: scf.for
// UNROLLONE: linalg.elemwise_binary {__byteir_loop_step__ = 0 : i32, __byteir_loop_total_step__ = 2 : i32}
// UNROLLONE: linalg.elemwise_binary {__byteir_loop_step__ = 0 : i32, __byteir_loop_total_step__ = 2 : i32, __root__}
// UNROLLONE: linalg.elemwise_binary {__byteir_loop_step__ = 1 : i32, __byteir_loop_total_step__ = 2 : i32}
// UNROLLONE: linalg.elemwise_binary {__byteir_loop_step__ = 1 : i32, __byteir_loop_total_step__ = 2 : i32, __root__}

// TWOSTEPUNROLL-LABEL: func.func @nested_binary_2x2
// TWOSTEPUNROLL: linalg.elemwise_binary {__byteir_loop_step__ = 0 : i32, __byteir_loop_total_step__ = 4 : i32}
// TWOSTEPUNROLL: linalg.elemwise_binary {__byteir_loop_step__ = 0 : i32, __byteir_loop_total_step__ = 4 : i32, __root__}
// TWOSTEPUNROLL: linalg.elemwise_binary {__byteir_loop_step__ = 1 : i32, __byteir_loop_total_step__ = 4 : i32}
// TWOSTEPUNROLL: linalg.elemwise_binary {__byteir_loop_step__ = 1 : i32, __byteir_loop_total_step__ = 4 : i32, __root__}
// TWOSTEPUNROLL: linalg.elemwise_binary {__byteir_loop_step__ = 2 : i32, __byteir_loop_total_step__ = 4 : i32}
// TWOSTEPUNROLL: linalg.elemwise_binary {__byteir_loop_step__ = 2 : i32, __byteir_loop_total_step__ = 4 : i32, __root__}
// TWOSTEPUNROLL: linalg.elemwise_binary {__byteir_loop_step__ = 3 : i32, __byteir_loop_total_step__ = 4 : i32}
// TWOSTEPUNROLL: linalg.elemwise_binary {__byteir_loop_step__ = 3 : i32, __byteir_loop_total_step__ = 4 : i32, __root__}

// -----

func.func @nested_binary_2x4(%arg0: tensor<1024x512xf32>, %arg1: tensor<1024x512xf32>, %arg2: tensor<1024x512xf32>) -> tensor<1024x512xf32> {
  %c1024 = arith.constant 1024 : index
  %c0 = arith.constant 0 : index
  %c512 = arith.constant 512 : index
  %c128 = arith.constant 128 : index
  %0 = tensor.empty() : tensor<1024x512xf32>
  %1 = scf.for %arg3 = %c0 to %c1024 step %c512 iter_args(%arg4 = %0) -> (tensor<1024x512xf32>) {
    %2 = scf.for %arg5 = %c0 to %c512 step %c128 iter_args(%arg6 = %0) -> (tensor<1024x512xf32>) {
      %extracted_slice = tensor.extract_slice %arg2[%arg3, %arg5] [512, 128] [1, 1] : tensor<1024x512xf32> to tensor<512x128xf32>
      %extracted_slice_0 = tensor.extract_slice %arg0[%arg3, %arg5] [512, 128] [1, 1] : tensor<1024x512xf32> to tensor<512x128xf32>
      %extracted_slice_1 = tensor.extract_slice %arg1[%arg3, %arg5] [512, 128] [1, 1] : tensor<1024x512xf32> to tensor<512x128xf32>
      %3 = tensor.empty() : tensor<512x128xf32>
      %4 = linalg.elemwise_binary ins(%extracted_slice_0, %extracted_slice_1 : tensor<512x128xf32>, tensor<512x128xf32>) outs(%3 : tensor<512x128xf32>) -> tensor<512x128xf32>
      %5 = tensor.empty() : tensor<512x128xf32>
      %6 = linalg.elemwise_binary {__root__} ins(%extracted_slice, %4 : tensor<512x128xf32>, tensor<512x128xf32>) outs(%5 : tensor<512x128xf32>) -> tensor<512x128xf32>
      %inserted_slice = tensor.insert_slice %6 into %0[%arg3, %arg5] [512, 128] [1, 1] : tensor<512x128xf32> into tensor<1024x512xf32>
      scf.yield %inserted_slice : tensor<1024x512xf32>
    } {__byteir_parallel__}
    scf.yield %2 : tensor<1024x512xf32>
  } {__byteir_parallel__}
  return %1 : tensor<1024x512xf32>
}

// UNROLLALL-LABEL: func.func @nested_binary_2x4
// UNROLLALL: linalg.elemwise_binary {__byteir_loop_step__ = 0 : i32, __byteir_loop_total_step__ = 8 : i32}
// UNROLLALL: linalg.elemwise_binary {__byteir_loop_step__ = 0 : i32, __byteir_loop_total_step__ = 8 : i32, __root__}
// UNROLLALL: linalg.elemwise_binary {__byteir_loop_step__ = 1 : i32, __byteir_loop_total_step__ = 8 : i32}
// UNROLLALL: linalg.elemwise_binary {__byteir_loop_step__ = 1 : i32, __byteir_loop_total_step__ = 8 : i32, __root__}
// UNROLLALL: linalg.elemwise_binary {__byteir_loop_step__ = 2 : i32, __byteir_loop_total_step__ = 8 : i32}
// UNROLLALL: linalg.elemwise_binary {__byteir_loop_step__ = 2 : i32, __byteir_loop_total_step__ = 8 : i32, __root__}
// UNROLLALL: linalg.elemwise_binary {__byteir_loop_step__ = 3 : i32, __byteir_loop_total_step__ = 8 : i32}
// UNROLLALL: linalg.elemwise_binary {__byteir_loop_step__ = 3 : i32, __byteir_loop_total_step__ = 8 : i32, __root__}
// UNROLLALL: linalg.elemwise_binary {__byteir_loop_step__ = 4 : i32, __byteir_loop_total_step__ = 8 : i32}
// UNROLLALL: linalg.elemwise_binary {__byteir_loop_step__ = 4 : i32, __byteir_loop_total_step__ = 8 : i32, __root__}
// UNROLLALL: linalg.elemwise_binary {__byteir_loop_step__ = 5 : i32, __byteir_loop_total_step__ = 8 : i32}
// UNROLLALL: linalg.elemwise_binary {__byteir_loop_step__ = 5 : i32, __byteir_loop_total_step__ = 8 : i32, __root__}
// UNROLLALL: linalg.elemwise_binary {__byteir_loop_step__ = 6 : i32, __byteir_loop_total_step__ = 8 : i32}
// UNROLLALL: linalg.elemwise_binary {__byteir_loop_step__ = 6 : i32, __byteir_loop_total_step__ = 8 : i32, __root__}
// UNROLLALL: linalg.elemwise_binary {__byteir_loop_step__ = 7 : i32, __byteir_loop_total_step__ = 8 : i32}
// UNROLLALL: linalg.elemwise_binary {__byteir_loop_step__ = 7 : i32, __byteir_loop_total_step__ = 8 : i32, __root__}

// UNROLLONE-LABEL: func.func @nested_binary_2x4
// UNROLLONE: scf.for
// UNROLLONE: linalg.elemwise_binary {__byteir_loop_step__ = 0 : i32, __byteir_loop_total_step__ = 4 : i32}
// UNROLLONE: linalg.elemwise_binary {__byteir_loop_step__ = 0 : i32, __byteir_loop_total_step__ = 4 : i32, __root__}
// UNROLLONE: linalg.elemwise_binary {__byteir_loop_step__ = 1 : i32, __byteir_loop_total_step__ = 4 : i32}
// UNROLLONE: linalg.elemwise_binary {__byteir_loop_step__ = 1 : i32, __byteir_loop_total_step__ = 4 : i32, __root__}
// UNROLLONE: linalg.elemwise_binary {__byteir_loop_step__ = 2 : i32, __byteir_loop_total_step__ = 4 : i32}
// UNROLLONE: linalg.elemwise_binary {__byteir_loop_step__ = 2 : i32, __byteir_loop_total_step__ = 4 : i32, __root__}
// UNROLLONE: linalg.elemwise_binary {__byteir_loop_step__ = 3 : i32, __byteir_loop_total_step__ = 4 : i32}
// UNROLLONE: linalg.elemwise_binary {__byteir_loop_step__ = 3 : i32, __byteir_loop_total_step__ = 4 : i32, __root__}

// -----

func.func @nested_binary_2x2x4(%arg0: tensor<32x1024x512xf32>, %arg1: tensor<32x1024x512xf32>, %arg2: tensor<32x1024x512xf32>) -> tensor<32x1024x512xf32> {
  %c1024 = arith.constant 1024 : index
  %c32 = arith.constant 32 : index
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %c512 = arith.constant 512 : index
  %c128 = arith.constant 128 : index
  %0 = tensor.empty() : tensor<32x1024x512xf32>
  %1 = scf.for %arg3 = %c0 to %c32 step %c16 iter_args(%arg4 = %0) -> (tensor<32x1024x512xf32>) {
    %2 = scf.for %arg5 = %c0 to %c1024 step %c512 iter_args(%arg6 = %0) -> (tensor<32x1024x512xf32>) {
      %3 = scf.for %arg7 = %c0 to %c512 step %c128 iter_args(%arg8 = %0) -> (tensor<32x1024x512xf32>) {
        %extracted_slice = tensor.extract_slice %arg0[%arg3, %arg5, %arg7] [16, 512, 128] [1, 1, 1] : tensor<32x1024x512xf32> to tensor<16x512x128xf32>
        %extracted_slice_0 = tensor.extract_slice %arg1[%arg3, %arg5, %arg7] [16, 512, 128] [1, 1, 1] : tensor<32x1024x512xf32> to tensor<16x512x128xf32>
        %4 = tensor.empty() : tensor<16x512x128xf32>
        %5 = linalg.elemwise_binary {__root__} ins(%extracted_slice, %extracted_slice_0 : tensor<16x512x128xf32>, tensor<16x512x128xf32>) outs(%4 : tensor<16x512x128xf32>) -> tensor<16x512x128xf32>
        %inserted_slice = tensor.insert_slice %5 into %0[%arg3, %arg5, %arg7] [16, 512, 128] [1, 1, 1] : tensor<16x512x128xf32> into tensor<32x1024x512xf32>
        scf.yield %inserted_slice : tensor<32x1024x512xf32>
      } {__byteir_parallel__}
      scf.yield %3 : tensor<32x1024x512xf32>
    } {__byteir_parallel__}
    scf.yield %2 : tensor<32x1024x512xf32>
  } {__byteir_parallel__}
  return %1 : tensor<32x1024x512xf32>
}
// UNROLLALL-LABEL: func.func @nested_binary_2x2x4
// UNROLLALL: linalg.elemwise_binary {__byteir_loop_step__ = 0 : i32, __byteir_loop_total_step__ = 16 : i32, __root__}
// UNROLLALL: linalg.elemwise_binary {__byteir_loop_step__ = 1 : i32, __byteir_loop_total_step__ = 16 : i32, __root__}
// UNROLLALL: linalg.elemwise_binary {__byteir_loop_step__ = 2 : i32, __byteir_loop_total_step__ = 16 : i32, __root__}
// UNROLLALL: linalg.elemwise_binary {__byteir_loop_step__ = 3 : i32, __byteir_loop_total_step__ = 16 : i32, __root__}
// UNROLLALL: linalg.elemwise_binary {__byteir_loop_step__ = 4 : i32, __byteir_loop_total_step__ = 16 : i32, __root__}
// UNROLLALL: linalg.elemwise_binary {__byteir_loop_step__ = 5 : i32, __byteir_loop_total_step__ = 16 : i32, __root__}
// UNROLLALL: linalg.elemwise_binary {__byteir_loop_step__ = 6 : i32, __byteir_loop_total_step__ = 16 : i32, __root__}
// UNROLLALL: linalg.elemwise_binary {__byteir_loop_step__ = 7 : i32, __byteir_loop_total_step__ = 16 : i32, __root__}
// UNROLLALL: linalg.elemwise_binary {__byteir_loop_step__ = 8 : i32, __byteir_loop_total_step__ = 16 : i32, __root__}
// UNROLLALL: linalg.elemwise_binary {__byteir_loop_step__ = 9 : i32, __byteir_loop_total_step__ = 16 : i32, __root__}
// UNROLLALL: linalg.elemwise_binary {__byteir_loop_step__ = 10 : i32, __byteir_loop_total_step__ = 16 : i32, __root__}
// UNROLLALL: linalg.elemwise_binary {__byteir_loop_step__ = 11 : i32, __byteir_loop_total_step__ = 16 : i32, __root__}
// UNROLLALL: linalg.elemwise_binary {__byteir_loop_step__ = 12 : i32, __byteir_loop_total_step__ = 16 : i32, __root__}
// UNROLLALL: linalg.elemwise_binary {__byteir_loop_step__ = 13 : i32, __byteir_loop_total_step__ = 16 : i32, __root__}
// UNROLLALL: linalg.elemwise_binary {__byteir_loop_step__ = 14 : i32, __byteir_loop_total_step__ = 16 : i32, __root__}
// UNROLLALL: linalg.elemwise_binary {__byteir_loop_step__ = 15 : i32, __byteir_loop_total_step__ = 16 : i32, __root__}

// UNROLLTWO-LABEL: func.func @nested_binary_2x2x4
// UNROLLTWO: scf.for
// UNROLLTWO: scf.for
// UNROLLTWO: linalg.elemwise_binary {__byteir_loop_step__ = 0 : i32, __byteir_loop_total_step__ = 4 : i32, __root__}
// UNROLLTWO: linalg.elemwise_binary {__byteir_loop_step__ = 1 : i32, __byteir_loop_total_step__ = 4 : i32, __root__}
// UNROLLTWO: linalg.elemwise_binary {__byteir_loop_step__ = 2 : i32, __byteir_loop_total_step__ = 4 : i32, __root__}
// UNROLLTWO: linalg.elemwise_binary {__byteir_loop_step__ = 3 : i32, __byteir_loop_total_step__ = 4 : i32, __root__}