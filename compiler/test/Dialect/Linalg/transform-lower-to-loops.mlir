// RUN: byteir-opt %s -transform-dialect-interpreter -canonicalize-ext -cse | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>

// CHECK-LABEL: func.func @reduction_row
func.func @reduction_row(%arg0: memref<1024x512xf32>, %arg1: memref<1024xf32>) -> memref<1024xf32> {
  %c512 = arith.constant 512 : index
  %cst = arith.constant 0xFF800000 : f32
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %alloc = memref.alloc() : memref<1024x8xf32>
  linalg.fill ins(%cst : f32) outs(%alloc : memref<1024x8xf32>)
  scf.for %arg2 = %c0 to %c512 step %c8 {
    %subview = memref.subview %arg0[0, %arg2] [1024, 8] [1, 1] : memref<1024x512xf32> to memref<1024x8xf32, strided<[512, 1], offset: ?>>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%subview : memref<1024x8xf32, strided<[512, 1], offset: ?>>) outs(%alloc : memref<1024x8xf32>) attrs =  {__split__} {
    ^bb0(%in: f32, %out: f32):
      %0 = arith.maxnumf %out, %in : f32
      linalg.yield %0 : f32
    }
    // CHECK: scf.for
    // CHECK:   scf.for
    // CHECK:     arith.maxnumf
  }
  linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%alloc : memref<1024x8xf32>) outs(%arg1 : memref<1024xf32>) attrs =  {__merge__} {
  ^bb0(%in: f32, %out: f32):
    %0 = arith.maxnumf %in, %out : f32
    linalg.yield %0 : f32
  }
  // CHECK: scf.for
  // CHECK:   scf.for
  // CHECK:     arith.maxnumf
  return %arg1 : memref<1024xf32>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match attributes{"__split__"} in %arg1 : (!pdl.operation) -> !pdl.operation
  %loops_0:2  = transform.structured.lower_to_loops %0 [0, -1] 
  cleanup
  %1 = transform.structured.match attributes{"__merge__"} in %arg1 : (!pdl.operation) -> !pdl.operation
  %loops_1:2  = transform.structured.lower_to_loops %1 [0, -1] 
  cleanup
}
