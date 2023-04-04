// RUN: byteir-opt -transform-dialect-interpreter --split-input-file %s | FileCheck %s

func.func @batch_matmul(%arg0: tensor<4x1024x512xf32>, %arg1: tensor<4x512x512xf32>, %arg2: tensor<4x1024x512xf32>) -> tensor<4x1024x512xf32> {
  %0 = linalg_ext.batch_matmul ins(%arg0, %arg1 : tensor<4x1024x512xf32>, tensor<4x512x512xf32>) outs(%arg2 : tensor<4x1024x512xf32>) layout = "nn" {__root__}
  return %0 : tensor<4x1024x512xf32>
}
// CHECK-LABEL: func.func @batch_matmul
// CHECK: scf.foreach_thread
// CHECK: linalg_ext.batch_matmul
// CHECK: scf.foreach_thread.perform_concurrently
// CHECK: tensor.parallel_insert_slice

transform.sequence failures(propagate) {
  ^bb0(%arg0: !pdl.operation):
    %0 = transform.structured.match attributes {__root__} in %arg0
    %loop, %init, %tiled, %merge = transform.structured.tile_reduction_using_foreach_thread %0
                                      by num_threads = [0, 0, 0, 8], tile_sizes = []
    cleanup
}