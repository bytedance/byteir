// RUN: mlir-opt %s -transform-dialect-interpreter -canonicalize -cse | FileCheck %s
// XFAIL: *

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>

func.func @convert_reduce_row(%arg0: tensor<512x1024xf32>, %arg1: tensor<512xf16>) -> tensor<512xf16> {
  %0 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg0 : tensor<512x1024xf32>) outs(%arg1 : tensor<512xf16>) attrs =  {__root__} {
  ^bb0(%in: f32, %out: f16):
    %3 = arith.truncf %in : f32 to f16
    %4 = arith.addf %3, %out : f16
    linalg.yield %4 : f16
  } -> tensor<512xf16>
  return %0 : tensor<512xf16>
}

transform.sequence failures(propagate) {
^bb0(%arg1: !pdl.operation):
  %0 = transform.structured.match attributes{"__root__"} in %arg1 : (!pdl.operation) -> !pdl.operation
  %loop, %fill, %split, %merge = transform.structured.tile_reduction_using_scf %0
    by tile_sizes = [0, 16]
  transform.annotate_ext %split { __split__ }
  transform.annotate_ext %merge { __merge__ }
  cleanup
  //%1 = transform.structured.match attributes{"__split__"} in %arg1 : (!pdl.operation) -> !pdl.operation
  //transform.dump(%1 : !pdl.operation) "__split__"
  //%2, %l = transform.structured.tile_ext %1 [16]
}
