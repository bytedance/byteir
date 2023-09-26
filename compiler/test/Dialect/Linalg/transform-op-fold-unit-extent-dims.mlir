// RUN: byteir-opt --transform-dialect-interpreter --split-input-file %s | FileCheck %s


#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (0, d1, d2)>
func.func @tensor_collapse(%arg0 : tensor<12x1024x1024xf32>, %arg1 : tensor<1x1024x1024xf32>) -> tensor<12x1024x1024xf32> {
  %empty = tensor.empty() : tensor<12x1024x1024xf32>
  %0 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : tensor<12x1024x1024xf32>, tensor<1x1024x1024xf32>) outs(%empty : tensor<12x1024x1024xf32>) {
  ^bb0(%in0: f32, %in1: f32, %out: f32):
    %1 = arith.addf %in0, %in1 : f32
    linalg.yield %1 : f32
  } -> tensor<12x1024x1024xf32>
  return %0 : tensor<12x1024x1024xf32>
}
// CHECK: linalg.generic
//   CHECK-SAME: ins(%[[COLLAPSE0:.+]], %[[COLLAPSE1:.+]] : tensor<12x1024x1024xf32>, tensor<1024x1024xf32>)
//   CHECK-SAME: outs(%[[COLLAPSE2:.+]] : tensor<12x1024x1024xf32>)

transform.sequence failures(propagate) {
^bb0(%arg0: !pdl.operation):
  %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!pdl.operation) -> !pdl.operation
  transform.apply_patterns to %0 {
    transform.apply_patterns.linalg.fold_unit_extent_dims_via_reshapes
  } : !pdl.operation
}
