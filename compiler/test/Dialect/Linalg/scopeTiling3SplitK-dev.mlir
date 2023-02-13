// RUN: byteir-opt %s -linalg-scope-tile="axis=0 tile-size=2" -cse | FileCheck %s -check-prefix=AXIS0
// RUN: byteir-opt %s -linalg-scope-tile="axis=1 tile-size=2" -cse | FileCheck %s -check-prefix=AXIS1
// RUN: byteir-opt %s -linalg-scope-tile="axis=2 tile-size=2" -cse | FileCheck %s -check-prefix=AXIS2
// RUN: byteir-opt %s -linalg-scope-tile="axis=1 tile-size=2 par-reduce" -cse | FileCheck %s -check-prefix=PARAXIS1
// RUN: byteir-opt %s -linalg-scope-tile="axis=2 tile-size=2 par-reduce" -cse | FileCheck %s -check-prefix=PARAXIS2
// XFAIL: *

#map0 = affine_map<(d0, d1) -> (d0, d1)>

func.func @matmul_2(%arg0: memref<128x64xf32>, %arg1: memref<64x64xf32>, %arg2: memref<64x64xf32>,  %arg3: memref<128x64xf32>) {
  %0 = memref.alloc() : memref<128x64xf32>
  %1 = memref.alloc() : memref<128x64xf32>
  linalg.matmul {__byteir_scope_tile_anchor__} ins(%arg0, %arg1 : memref<128x64xf32>, memref<64x64xf32>) outs(%0 : memref<128x64xf32>)
  linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%0 : memref<128x64xf32>) outs(%1 : memref<128x64xf32>){
  ^bb0(%arg4: f32, %arg5: f32):  // no predecessors
    %3 = math.absf %arg4: f32
    linalg.yield %3 : f32
  }
  linalg.matmul ins(%1, %arg2 : memref<128x64xf32>, memref<64x64xf32>) outs(%arg3 : memref<128x64xf32>)
  return
}

// AXIS0-LABEL: func.func @matmul_2(
// AXIS0: scf.parallel
// AXIS0:   linalg.matmul
// AXIS0:   linalg.generic
// AXIS0:   linalg.matmul
// AXIS0:   scf.yield

// AXIS1-LABEL: func.func @matmul_2(
// AXIS1: scf.for
// AXIS1:   linalg.matmul
// AXIS1:   linalg.generic
// AXIS1:   linalg.matmul
// AXIS1:   }
// AXIS1: return

// AXIS2-LABEL: func.func @matmul_2(
// AXIS2: scf.for
// AXIS2:   linalg.matmul
// AXIS2: }
// AXIS2: linalg.generic
// AXIS2: linalg.matmul
// AXIS2: return

// PARAXIS1-LABEL: func.func @matmul_2(
// PARAXIS1: scf.parallel
// PARAXIS1:   linalg.matmul
// PARAXIS1:   linalg.generic
// PARAXIS1:   linalg.matmul
// PARAXIS1:   linalg.generic
// PARAXIS1-SAME: __byteir_atomic_kind__ = 0 : i32
// PARAXIS1:   scf.yield

// PARAXIS2-LABEL: func.func @matmul_2(
// PARAXIS2: scf.parallel
// PARAXIS2:   linalg.matmul
// PARAXIS2:   linalg.generic
// PARAXIS2:   scf.yield
// PARAXIS2: linalg.generic
// PARAXIS2: linalg.matmul
// PARAXIS2: return
