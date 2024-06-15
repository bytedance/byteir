// RUN: byteir-opt %s --transform-interpreter --canonicalize --cse -split-input-file | FileCheck %s

module {
  func.func @blocks_2d(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: f32, %arg3: !gpu.async.token) -> memref<?x?xf32> {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %dim = memref.dim %arg0, %c0 : memref<?x?xf32>
    %dim_0 = memref.dim %arg0, %c1 : memref<?x?xf32>
    %0 = gpu.launch async [%arg3] blocks(%arg4, %arg5, %arg6) in (%arg10 = %c1, %arg11 = %c1, %arg12 = %c1) threads(%arg7, %arg8, %arg9) in (%arg13 = %c1, %arg14 = %c1, %arg15 = %c1) {
      scf.forall (%arg16, %arg17) in (%dim, %dim_0) {
        %1 = memref.load %arg0[%arg16, %arg17] : memref<?x?xf32>
        %2 = memref.load %arg1[%arg16, %arg17] : memref<?x?xf32>
        %3 = math.fma %arg2, %1, %2 : f32
        memref.store %3, %arg1[%arg16, %arg17] : memref<?x?xf32>
      } {mapping = [#gpu.block<x>, #gpu.block<y>]}
      gpu.terminator
    }
    return %arg1 : memref<?x?xf32>
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["gpu.launch"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      %1 = transform.gpu.map_forall_to_blocks_ext %0 grid_dims = [kDynamic, kDynamic, 1] : (!transform.any_op) -> !transform.any_op
      transform.yield 
    }
  }
}

// CHECK-LABEL: func.func @blocks_2d
// CHECK-NEXT: %[[C1:.*]] = arith.constant 1 : index
// CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT: %[[DIM0:.*]] = memref.dim %arg0, %[[C0]] : memref<?x?xf32>
// CHECK-NEXT: %[[DIM1:.*]] = memref.dim %arg0, %[[C1]] : memref<?x?xf32>
// CHECK-NEXT: gpu.launch async [%arg3] blocks(%arg4, %arg5, %arg6) in (%arg10 = %[[DIM0]], %arg11 = %[[DIM1]], %arg12 = %[[C1]])
// CHECK-NEXT: %[[BIDX:.*]] = gpu.block_id  x
// CHECK-NEXT: %[[BIDY:.*]] = gpu.block_id  y
// CHECK-NEXT: %[[P1:.*]] = arith.cmpi ult, %[[BIDX]], %[[DIM0]] : index
// CHECK-NEXT: %[[P2:.*]] = arith.cmpi ult, %[[BIDY]], %[[DIM1]] : index
// CHECK-NEXT: %[[P3:.*]] = arith.andi %[[P1]], %[[P2]] : i1
// CHECK-NEXT: scf.if %[[P3]]
        // CHECK-NEXT: %[[V0:.*]] = memref.load %arg0[%[[BIDX]], %[[BIDY]]] : memref<?x?xf32>
        // CHECK-NEXT: %[[V1:.*]] = memref.load %arg1[%[[BIDX]], %[[BIDY]]] : memref<?x?xf32>
        // CHECK-NEXT: %[[V2:.*]] = math.fma %arg2, %[[V0]], %[[V1]] : f32
        // CHECK-NEXT: memref.store %[[V2]], %arg1[%[[BIDX]], %[[BIDY]]] : memref<?x?xf32>

// -----

module {
  func.func @saxpy4d(%arg0: memref<?x?x?x?xf32>, %arg1: memref<?x?x?x?xf32>, %arg2: f32) -> memref<?x?x?x?xf32> {
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c4 = arith.constant 4 : index
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %dim0 = memref.dim %arg0, %c0 : memref<?x?x?x?xf32>
    %dim1 = memref.dim %arg0, %c1 : memref<?x?x?x?xf32>
    %dim2 = memref.dim %arg0, %c2 : memref<?x?x?x?xf32>
    %dim3 = memref.dim %arg0, %c3 : memref<?x?x?x?xf32>

    scf.forall (%arg3, %arg4) in (%dim0, %dim1) {
      scf.forall (%arg5, %arg6) in (%dim2, %dim3) {
        %0 = memref.load %arg0[%arg3, %arg4, %arg5, %arg6] : memref<?x?x?x?xf32>
        %1 = memref.load %arg1[%arg3, %arg4, %arg5, %arg6] : memref<?x?x?x?xf32>
        %2 = math.fma %arg2, %0, %1 : f32
        memref.store %2, %arg1[%arg3, %arg4, %arg5, %arg6] : memref<?x?x?x?xf32>
      } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
    } {mapping = [#gpu.block<y>, #gpu.block<x>]}
    return %arg1 : memref<?x?x?x?xf32>
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      %1 = transform.gpu.map_forall_to_blocks_ext %0 generate_gpu_launch : (!transform.any_op) -> !transform.any_op
      %2 = transform.gpu.map_nested_forall_to_threads_ext %1 block_dims = [32, 4, 1] : (!transform.any_op) -> !transform.any_op
      transform.yield 
    }
  }
}

// CHECK-LABEL: func.func @saxpy4d
// CHECK-DAG: %[[C3:.*]] = arith.constant 3 : index
// CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[DIM2:.*]] = memref.dim %arg0, %[[C2]] : memref<?x?x?x?xf32>
// CHECK-DAG: %[[DIM0:.*]] = memref.dim %arg0, %[[C0]] : memref<?x?x?x?xf32>
// CHECK-DAG: %[[DIM1:.*]] = memref.dim %arg0, %[[C1]] : memref<?x?x?x?xf32>
// CHECK-DAG: %[[DIM3:.*]] = memref.dim %arg0, %[[C3]] : memref<?x?x?x?xf32>
// CHECK: gpu.launch blocks(%arg3, %arg4, %arg5) in (%arg9 = %[[DIM1]], %arg10 = %[[DIM0]], %arg11 = %c1)
// CHECK-DAG: %[[BIDX:.*]] = gpu.block_id  x
// CHECK-DAG: %[[BIDY:.*]] = gpu.block_id  y
// CHECK-DAG: %[[TIDX:.*]] = gpu.thread_id  x
// CHECK-DAG: %[[TIDY:.*]] = gpu.thread_id  y
// CHECK-DAG: %[[P1:.*]] = arith.cmpi ult, %[[TIDX]], %[[DIM3]] : index
// CHECK-DAG: %[[P2:.*]] = arith.cmpi ult, %[[TIDY]], %[[DIM2]] : index
// CHECK-DAG: %[[P3:.*]] = arith.andi %[[P1]], %[[P2]] : i1
// CHECK: scf.if %[[P3]] {
        // CHECK-NEXT: %[[V0:.*]] = memref.load %arg0[%[[BIDY]], %[[BIDX]], %[[TIDY]], %[[TIDX]]] : memref<?x?x?x?xf32>
        // CHECK-NEXT: %[[V1:.*]] = memref.load %arg1[%[[BIDY]], %[[BIDX]], %[[TIDY]], %[[TIDX]]] : memref<?x?x?x?xf32>
        // CHECK-NEXT: %[[V2:.*]] = math.fma %arg2, %[[V0]], %[[V1]] : f32
        // CHECK-NEXT: memref.store %[[V2]], %arg1[%[[BIDY]], %[[BIDX]], %[[TIDY]], %[[TIDX]]] : memref<?x?x?x?xf32>

// -----

module {
  func.func @block_linear_existing_launch(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<32xf32>, %arg3: f32, %arg4: !gpu.async.token) -> memref<?x?xf32> {
    %c9 = arith.constant 9 : index
    %c7 = arith.constant 7 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %dim0 = memref.dim %arg0, %c0 : memref<?x?xf32>
    %dim1 = memref.dim %arg0, %c1 : memref<?x?xf32>

    %0 = gpu.launch async [%arg4] blocks(%arg5, %arg6, %arg7) in (%arg11 = %c1, %arg12 = %c1, %arg13 = %c1) threads(%arg8, %arg9, %arg10) in (%arg14 = %c1, %arg15 = %c1, %arg16 = %c1) {
      scf.forall (%arg17, %arg18) in (%dim0, %dim1) {
        %1 = memref.load %arg0[%arg17, %arg18] : memref<?x?xf32>
        %2 = memref.load %arg1[%arg17, %arg18] : memref<?x?xf32>
        %3 = math.fma %arg3, %1, %2 : f32
        memref.store %3, %arg1[%arg17, %arg18] : memref<?x?xf32>
      } {mapping = [#gpu.block<linear_dim_1>, #gpu.block<linear_dim_0>]}
      gpu.terminator
    }
    return %arg1 : memref<?x?xf32>
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["gpu.launch"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      %1 = transform.gpu.map_forall_to_blocks_ext %0 grid_dims = [kDynamic, kDynamic, kDynamic] : (!transform.any_op) -> !transform.any_op
      transform.yield 
    }
  }
}

// CHECK: #[[$MAP_TOTAL:.*]] = affine_map<()[s0, s1] -> (s0 * s1)>
// CHECK: #[[$MAP_LINEAR_IDY:.*]] = affine_map<(d0)[s0] -> (d0 floordiv s0)>
// CHECK: #[[$MAP_LINEAR_IDX:.*]] = affine_map<(d0)[s0] -> (d0 mod s0)>

// CHECK-LABLE: func.func @block_linear_existing_launch
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[DIM0:.*]] = memref.dim %arg0, %[[C0]] : memref<?x?xf32>
// CHECK-DAG: %[[DIM1:.*]] = memref.dim %arg0, %[[C1]] : memref<?x?xf32>
// CHECK-DAG: %[[TOTAL_SIZE:.*]] = affine.apply #[[$MAP_TOTAL]]()[%[[DIM1]], %[[DIM0]]]
// CHECK: gpu.launch async [%arg4] blocks(%arg5, %arg6, %arg7) in (%arg11 = %[[TOTAL_SIZE]], %arg12 = %[[C1]], %arg13 = %[[C1]])
// CHECK-DAG: %[[BIDX:.*]] = gpu.block_id  x
// CHECK-DAG: %[[LINEAR_IDY:.*]] = affine.apply #[[$MAP_LINEAR_IDY]](%[[BIDX]])[%[[DIM1]]]
// CHECK-DAG: %[[LINEAR_IDX:.*]] = affine.apply #[[$MAP_LINEAR_IDX]](%[[BIDX]])[%[[DIM1]]]
// CHECK-DAG: %[[V0:.*]] = memref.load %arg0[%[[LINEAR_IDY]], %[[LINEAR_IDX]]] : memref<?x?xf32>
// CHECK-DAG: %[[V1:.*]] = memref.load %arg1[%[[LINEAR_IDY]], %[[LINEAR_IDX]]] : memref<?x?xf32>
// CHECK-DAG: %[[V2:.*]] = math.fma %arg3, %[[V0]], %[[V1]] : f32
// CHECK-DAG: memref.store %[[V2]], %arg1[%[[LINEAR_IDY]], %[[LINEAR_IDX]]] : memref<?x?xf32>

// -----

#map = affine_map<(d0) -> (d0 * 128)>
#map1 = affine_map<(d0) -> (d0 * 32)>
module {
  func.func @simple_fill(%arg0: memref<128xf32>) -> memref<128xf32> {
    %c0 = arith.constant 0 : index
    %cst = arith.constant dense<0.000000e+00> : vector<32xf32>
    scf.forall (%arg1) in (1) {
      %0 = affine.apply #map(%arg1)
      %subview = memref.subview %arg0[%0] [128] [1] : memref<128xf32> to memref<128xf32, strided<[1], offset: ?>>
      scf.forall (%arg2) in (4) {
        %1 = affine.apply #map1(%arg2)
        %subview_0 = memref.subview %subview[%1] [32] [1] : memref<128xf32, strided<[1], offset: ?>> to memref<32xf32, strided<[1], offset: ?>>
        vector.transfer_write %cst, %subview_0[%c0] {in_bounds = [true]} : vector<32xf32>, memref<32xf32, strided<[1], offset: ?>>
        memref.copy %subview_0, %subview_0 : memref<32xf32, strided<[1], offset: ?>> to memref<32xf32, strided<[1], offset: ?>>
      } {mapping = [#gpu.warp<linear_dim_0>]}
      memref.copy %subview, %subview : memref<128xf32, strided<[1], offset: ?>> to memref<128xf32, strided<[1], offset: ?>>
    } {mapping = [#gpu.block<x>]}
    return %arg0 : memref<128xf32>
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      %1 = transform.gpu.map_forall_to_blocks_ext %0 generate_gpu_launch : (!transform.any_op) -> !transform.any_op
      %2 = transform.gpu.map_nested_forall_to_threads_ext %1 block_dims = [4, 8, 4] : (!transform.any_op) -> !transform.any_op
      transform.yield 
    }
  }
}

// CHECK: #[[$MAPW:.*]] = affine_map<(d0, d1, d2) -> (d2 * 32 + ((d0 + d1 * 4) floordiv 32) * 32)>
// CHECK-LABEL: func.func @simple_fill
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C4:.*]] = arith.constant 4 : index
// CHECK-DAG: %[[C8:.*]] = arith.constant 8 : index
// CHECK: gpu.launch blocks(%arg1, %arg2, %arg3) in (%arg7 = %[[C1]], %arg8 = %[[C1]], %arg9 = %[[C1]]) threads(%arg4, %arg5, %arg6) in (%arg10 = %[[C4]], %arg11 = %[[C8]], %arg12 = %[[C4]])
    // CHECK: %[[TIDX:.*]] = gpu.thread_id  x
    // CHECK: %[[TIDY:.*]] = gpu.thread_id  y
    // CHECK: %[[TIDZ:.*]] = gpu.thread_id  z
    // CHECK: %[[THX:.*]] = affine.apply #[[$MAPW]](%[[TIDX]], %[[TIDY]], %[[TIDZ]])
    // CHECK-NOT:     scf.if
    // CHECK: memref.subview %{{.*}}[%[[THX]]]

// -----

module {
  func.func @block_linear_existing_launch(%arg0: memref<2x32xf32>, %arg1: memref<2x32xf32>, %arg2: memref<32xf32>, %arg3: f32, %arg4: !gpu.async.token) -> memref<2x32xf32> {
    %c9 = arith.constant 9 : index
    %c7 = arith.constant 7 : index
    %c1 = arith.constant 1 : index
    %0 = gpu.launch async [%arg4] blocks(%arg5, %arg6, %arg7) in (%arg11 = %c1, %arg12 = %c1, %arg13 = %c1) threads(%arg8, %arg9, %arg10) in (%arg14 = %c1, %arg15 = %c1, %arg16 = %c1) {
      scf.forall (%arg17, %arg18) in (%c7, %c9) {
        %1 = memref.load %arg0[%arg17, %arg18] : memref<2x32xf32>
        %2 = memref.load %arg1[%arg17, %arg18] : memref<2x32xf32>
        %3 = math.fma %arg3, %1, %2 : f32
        memref.store %3, %arg1[%arg17, %arg18] : memref<2x32xf32>
      } {mapping = [#gpu.block<linear_dim_1>, #gpu.block<linear_dim_0>]}
      gpu.terminator
    }
    return %arg1 : memref<2x32xf32>
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["gpu.launch"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      %1 = transform.gpu.map_forall_to_blocks_ext %0 grid_dims = [12, 9, 1] : (!transform.any_op) -> !transform.any_op
      transform.yield 
    }
  }
}

// CHECK-DAG: #[[$MAP_LINEAR_ID:.*]] = affine_map<(d0, d1) -> (d0 + d1 * 12)>
// CHECK-DAG: #[[$MAP_LINEAR_IDY:.*]] = affine_map<(d0, d1) -> ((d0 + d1 * 12) floordiv 9)>
// CHECK-DAG: #[[$MAP_LINEAR_IDX:.*]] = affine_map<(d0, d1) -> ((d0 + d1 * 12) mod 9)>

// CHECK-LABEL: func.func @block_linear_existing_launch
// CHECK-DAG: %[[C63:.*]] = arith.constant 63 : index
// CHECK-DAG: %[[C9:.*]] = arith.constant 9 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C12:.*]] = arith.constant 12 : index
// CHECK: gpu.launch async [%arg4] blocks(%arg5, %arg6, %arg7) in (%arg11 = %[[C12]], %arg12 = %[[C9]], %arg13 = %c1)
// CHECK-DAG: %[[BIDX:.*]] = gpu.block_id  x
// CHECK-DAG: %[[BIDY:.*]] = gpu.block_id  y
// CHECK-DAG: %[[LINEAR_ID:.*]] = affine.apply #[[$MAP_LINEAR_ID]](%[[BIDX]], %[[BIDY]])
// CHECK-DAG: %[[LINEAR_IDY:.*]] = affine.apply #[[$MAP_LINEAR_IDY]](%[[BIDX]], %[[BIDY]])
// CHECK-DAG: %[[LINEAR_IDX:.*]] = affine.apply #[[$MAP_LINEAR_IDX]](%[[BIDX]], %[[BIDY]])
// CHECK-DAG: %[[P:.*]] = arith.cmpi ult, %[[LINEAR_ID]], %[[C63]] : index
// CHECK: scf.if %[[P]]
        // CHECK-DAG: %[[V0:.*]] = memref.load %arg0[%[[LINEAR_IDY]], %[[LINEAR_IDX]]] : memref<2x32xf32>
        // CHECK-DAG: %[[V1:.*]] = memref.load %arg1[%[[LINEAR_IDY]], %[[LINEAR_IDX]]] : memref<2x32xf32>
        // CHECK-DAG: %[[V2:.*]] = math.fma %arg3, %[[V0]], %[[V1]] : f32
        // CHECK-DAG: memref.store %[[V2]], %arg1[%[[LINEAR_IDY]], %[[LINEAR_IDX]]] : memref<2x32xf32>
