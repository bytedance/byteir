// RUN: byteir-opt %s -transform-dialect-interpreter -canonicalize-ext --split-input-file | FileCheck %s

func.func @reduction_tile(%arg0: tensor<128x15xf32>, %out: tensor<128xf32>) -> tensor<128xf32> {
  %red = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                          affine_map<(d0, d1) -> (d0)>],
   iterator_types = ["parallel", "reduction"]}
   ins(%arg0 : tensor<128x15xf32>)
   outs(%out : tensor<128xf32>) {
    ^bb0(%arg7: f32, %arg9: f32):
      %1 = arith.mulf %arg7, %arg7 : f32
      %2 = arith.addf %1, %arg9 : f32
      linalg.yield %2 : f32
    } -> tensor<128xf32>
  return %red : tensor<128xf32>
}
// CHECK-LABEL: func.func @reduction_tile
// CHECK: tensor.empty() : tensor<128xf32>
// CHECK: scf.foreach_thread
// CHECK: tensor.extract_slice
// CHECK: linalg.generic
// CHECK: ccl.all_reduce
// CHECK: linalg.generic
// CHECK: scf.foreach_thread.perform_concurrently

transform.sequence failures(propagate) {
^bb0(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["linalg.generic"]} in %arg1
  %loop, %init, %tiled, %merge = transform.structured.tile_reduction_using_foreach_thread %0
    by num_threads = [0, 5], tile_sizes = []
  %new_loop, %new_init = transform.structured.shared_output_to_distributed_style %loop, %init, %merge
}