// RUN: byteir-opt %s --transform-interpreter --canonicalize --cse -split-input-file | FileCheck %s

module attributes {transform.with_named_sequence} {
  func.func @dynamic_mapping(%arg0: memref<?x?x2xf32>, %arg1: memref<?x?x2xf32>, %arg2: f32, %arg3: !gpu.async.token) -> memref<?x?x2xf32> {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %dim = memref.dim %arg0, %c0 : memref<?x?x2xf32>
    %dim_0 = memref.dim %arg0, %c1 : memref<?x?x2xf32>
    %0 = gpu.launch async [%arg3] blocks(%arg4, %arg5, %arg6) in (%arg10 = %c1, %arg11 = %c1, %arg12 = %c1) threads(%arg7, %arg8, %arg9) in (%arg13 = %c1, %arg14 = %c1, %arg15 = %c1) {
      scf.forall (%arg16, %arg17, %arg18) in (%dim, %dim_0, %c2) {
        %1 = memref.load %arg0[%arg16, %arg17, %arg18] : memref<?x?x2xf32>
        %2 = memref.load %arg1[%arg16, %arg17, %arg18] : memref<?x?x2xf32>
        %3 = math.fma %arg2, %1, %2 : f32
        memref.store %3, %arg1[%arg16, %arg17, %arg18] : memref<?x?x2xf32>
      } {mapping = [#gpu.thread<y>, #gpu.thread<x>, #gpu.thread<z>]}
      gpu.terminator
    }
    return %arg1 : memref<?x?x2xf32>
  }
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["gpu.launch"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %1 = transform.gpu.map_nested_forall_to_threads_ext %0 block_dims = [32, 4, 2] sync_after_distribute = false : (!transform.any_op) -> !transform.any_op
    transform.yield 
  }
}


// CHECK-LABEL: func.func @dynamic_mapping
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[DIM0:.*]] = memref.dim %arg0, %[[C0]] : memref<?x?x2xf32>
// CHECK-DAG: %[[DIM1:.*]] = memref.dim %arg0, %[[C1]] : memref<?x?x2xf32>
// CHECK: %[[LAUNCH:.*]] = gpu.launch async
      // CHECK: %[[TIDX:.*]] = gpu.thread_id  x
      // CHECK: %[[TIDY:.*]] = gpu.thread_id  y
      // CHECK: %[[TIDZ:.*]] = gpu.thread_id  z
      // CHECK: %[[P1:.*]] = arith.cmpi ult, %[[TIDX]], %[[DIM1]] : index
      // CHECK: %[[P2:.*]] = arith.cmpi ult, %[[TIDY]], %[[DIM0]] : index
      // CHECK: %[[P3:.*]] = arith.andi %[[P1]], %[[P2]] : i1
      // CHECK: scf.if %[[P3]]
                //CHECK-NEXT: %[[V0:.*]] = memref.load %arg0[%[[TIDY]], %[[TIDX]], %[[TIDZ]]]
                //CHECK-NEXT: %[[V1:.*]] = memref.load %arg1[%[[TIDY]], %[[TIDX]], %[[TIDZ]]]
                //CHECK-NEXT: %[[V2:.*]] = math.fma %arg2, %[[V0]], %[[V1]] : f32
                //CHECK-NEXT: memref.store %[[V2]], %arg1[%[[TIDY]], %[[TIDX]], %[[TIDZ]]]
// -----

module {
  func.func @thread_linear(%arg0: memref<2x?x64xf32>, %arg1: memref<2x?x64xf32>, %arg2: f32, %arg3: !gpu.async.token) -> memref<2x?x64xf32> {
    %c7 = arith.constant 7 : index
    %c9 = arith.constant 9 : index
    %c1 = arith.constant 1 : index
    %dim = memref.dim %arg0, %c1 : memref<2x?x64xf32>
    %0 = gpu.launch async [%arg3] blocks(%arg4, %arg5, %arg6) in (%arg10 = %c1, %arg11 = %c1, %arg12 = %c1) threads(%arg7, %arg8, %arg9) in (%arg13 = %c1, %arg14 = %c1, %arg15 = %c1) {
      scf.forall (%arg16, %arg17, %arg18) in (%c7, %dim, %c9) {
        %1 = memref.load %arg0[%arg16, %arg17, %arg18] : memref<2x?x64xf32>
        %2 = memref.load %arg1[%arg16, %arg17, %arg18] : memref<2x?x64xf32>
        %3 = math.fma %arg2, %1, %2 : f32
        memref.store %3, %arg1[%arg16, %arg17, %arg18] : memref<2x?x64xf32>
      } {mapping = [#gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>, #gpu.thread<linear_dim_2>]}
      gpu.terminator
    }
    return %arg1 : memref<2x?x64xf32>
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["gpu.launch"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      %1 = transform.gpu.map_nested_forall_to_threads_ext %0 block_dims = [32, 8, 4] : (!transform.any_op) -> !transform.any_op
      transform.yield 
    }
  }
}

// CHECK: #map = affine_map<(d0, d1, d2) -> (d0 + d1 * 32 + d2 * 256)>
// CHECK: #map1 = affine_map<(d0, d1, d2)[s0] -> ((d0 + d1 * 32 + d2 * 256) floordiv (s0 * 7))>
// CHECK: #map2 = affine_map<(d0, d1, d2)[s0] -> (((d0 + d1 * 32 + d2 * 256) mod (s0 * 7)) floordiv s0)>
// CHECK: #map3 = affine_map<(d0, d1, d2)[s0] -> (((d0 + d1 * 32 + d2 * 256) mod (s0 * 7)) mod s0)>
// CHECK: #map4 = affine_map<()[s0] -> (s0 * 63)>
// CHECK-LABEL: func.func @thread_linear
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[DIM:.*]] = memref.dim %arg0, %[[C1]] : memref<2x?x64xf32>
// CHECK: %[[LAUNCH:.*]] = gpu.launch async
// CHECK-DAG: %[[TIDX:.*]] = gpu.thread_id  x
// CHECK-DAG: %[[TIDY:.*]] = gpu.thread_id  y
// CHECK-DAG: %[[TIDZ:.*]] = gpu.thread_id  z
// CHECK-DAG: %[[LINEAR_ID:.*]] = affine.apply #map(%[[TIDX]], %[[TIDY]], %[[TIDZ]])
// CHECK-DAG: %[[LID_2:.*]] = affine.apply #map1(%[[TIDX]], %[[TIDY]], %[[TIDZ]])[%[[DIM]]]
// CHECK-DAG: %[[LID_1:.*]] = affine.apply #map2(%[[TIDX]], %[[TIDY]], %[[TIDZ]])[%[[DIM]]]
// CHECK-DAG: %[[LID_0:.*]] = affine.apply #map3(%[[TIDX]], %[[TIDY]], %[[TIDZ]])[%[[DIM]]]
// CHECK-DAG: %[[ACTIVE_SIZE:.*]] = affine.apply #map4()[%[[DIM]]]
// CHECK-DAG: %[[P:.*]] = arith.cmpi ult, %[[LINEAR_ID]], %[[ACTIVE_SIZE]] : index
// CHECK-NEXT: scf.if %[[P]]
        // CHECK-NEXT: %[[V0:.*]] = memref.load %arg0[%[[LID_1]], %[[LID_0]], %[[LID_2]]] : memref<2x?x64xf32>
        // CHECK-NEXT: %[[V1:.*]] = memref.load %arg1[%[[LID_1]], %[[LID_0]], %[[LID_2]]] : memref<2x?x64xf32>
        // CHECK-NEXT: %[[V2:.*]] = math.fma %arg2, %[[V0]], %[[V1]] : f32
        // CHECK-NEXT: memref.store %[[V2]], %arg1[%[[LID_1]], %[[LID_0]], %[[LID_2]]] : memref<2x?x64xf32>

// -----

module {
  func.func @warp_linear(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: f32, %arg3: !gpu.async.token) -> memref<?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c1_0 = arith.constant 1 : index
    %dim = memref.dim %arg0, %c0 : memref<?x?xf32>
    %dim_1 = memref.dim %arg0, %c1 : memref<?x?xf32>
    %0 = gpu.launch async [%arg3] blocks(%arg4, %arg5, %arg6) in (%arg10 = %c1_0, %arg11 = %c1_0, %arg12 = %c1_0) threads(%arg7, %arg8, %arg9) in (%arg13 = %c1_0, %arg14 = %c1_0, %arg15 = %c1_0) {
      scf.forall (%arg16, %arg17) in (%dim, %dim_1) {
        %1 = memref.load %arg0[%arg16, %arg17] : memref<?x?xf32>
        %2 = memref.load %arg1[%arg16, %arg17] : memref<?x?xf32>
        %3 = math.fma %arg2, %1, %2 : f32
        memref.store %3, %arg1[%arg16, %arg17] : memref<?x?xf32>
      } {mapping = [#gpu.warp<linear_dim_1>, #gpu.warp<linear_dim_0>]}
      gpu.terminator
    }
    return %arg1 : memref<?x?xf32>
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["gpu.launch"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      %1 = transform.gpu.map_nested_forall_to_threads_ext %0 block_dims = [32, 8, 4] : (!transform.any_op) -> !transform.any_op
      transform.yield 
    }
  }
}


// CHECK: #[[$MAP_LINEAR:.*]] = affine_map<(d0, d1, d2) -> (d0 + d1 * 32 + d2 * 256)>
// CHECK: #[[$MAP_WY:.*]] = affine_map<(d0, d1, d2)[s0] -> ((d1 + d2 * 8 + d0 floordiv 32) floordiv s0)>
// CHECK: #[[$MAP_WX:.*]] = affine_map<(d0, d1, d2)[s0] -> ((d1 + d2 * 8 + d0 floordiv 32) mod s0)>
// CHECK: #[[$MAP_ACTIVE:.*]] = affine_map<()[s0, s1] -> ((s0 * s1) * 32)>

// CHECK-LABEL: func.func @warp_linear
//CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
//CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
//CHECK-DAG: %[[DIM0:.*]] = memref.dim %arg0, %[[C0]] : memref<?x?xf32>
//CHECK-DAG: %[[DIM1:.*]] = memref.dim %arg0, %[[C1]] : memref<?x?xf32>
//CHECK: %[[LAUNCH]] = gpu.launch async
//CHECK-DAG: %[[TIDX:.*]] = gpu.thread_id  x
//CHECK-DAG: %[[TIDY:.*]] = gpu.thread_id  y
//CHECK-DAG: %[[TIDZ:.*]] = gpu.thread_id  z
//CHECK-DAG: %[[LINEAR_ID:.*]] = affine.apply #[[$MAP_LINEAR]](%[[TIDX]], %[[TIDY]], %[[TIDZ]])
//CHECK-DAG: %[[WY_ID:.*]] = affine.apply #[[$MAP_WY]](%[[TIDX]], %[[TIDY]], %[[TIDZ]])[%[[DIM1]]]
//CHECK-DAG: %[[WX_ID:.*]] = affine.apply #[[$MAP_WX]](%[[TIDX]], %[[TIDY]], %[[TIDZ]])[%[[DIM1]]]
//CHECK-DAG: %[[ACTIVE_SIZE:.*]] = affine.apply #[[$MAP_ACTIVE]]()[%[[DIM1]], %[[DIM0]]]
//CHECK-DAG: %[[P1:.*]] = arith.cmpi ult, %[[LINEAR_ID]], %[[ACTIVE_SIZE]] : index
//CHECK: scf.if %[[P1]]
        // CHECK-NEXT: %[[V0:.*]] = memref.load %arg0[%[[WY_ID]], %[[WX_ID]]] : memref<?x?xf32>
        // CHECK-NEXT: %[[V1:.*]] = memref.load %arg1[%[[WY_ID]], %[[WX_ID]]] : memref<?x?xf32>
        // CHECK-NEXT: %[[V2:.*]] = math.fma %arg2, %[[V0]], %[[V1]] : f32
        // CHECK-NEXT: memref.store %[[V2]], %arg1[%[[WY_ID]], %[[WX_ID]]] : memref<?x?xf32>

// -----

module {
  func.func @map_multi_level_linear(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?xf32>, %arg3: f32, %arg4: !gpu.async.token) -> memref<?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c10 = arith.constant 10 : index
    %c9 = arith.constant 9 : index
    %c7 = arith.constant 7 : index
    %c1_0 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %dim0 = memref.dim %arg0, %c0 : memref<?x?xf32>
    %dim1 = memref.dim %arg0, %c1 : memref<?x?xf32>
    %dim2 = memref.dim %arg2, %c0 : memref<?xf32>
    %0 = gpu.launch async [%arg4] blocks(%arg5, %arg6, %arg7) in (%arg11 = %c1, %arg12 = %c1, %arg13 = %c1) threads(%arg8, %arg9, %arg10) in (%arg14 = %c1, %arg15 = %c1, %arg16 = %c1) {

      scf.forall (%arg17, %arg18) in (%dim0, %c9) {
        %1 = memref.load %arg0[%arg17, %arg18] : memref<?x?xf32>
        %2 = memref.load %arg1[%arg17, %arg18] : memref<?x?xf32>
        %3 = math.fma %arg3, %1, %2 : f32
        memref.store %3, %arg1[%arg17, %arg18] : memref<?x?xf32>
      } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}

      scf.forall (%arg17, %arg18, %arg19) in (%dim0, %dim1, %c1_0) {
        %1 = memref.load %arg0[%arg17, %arg18] : memref<?x?xf32>
        %2 = arith.addf %arg3, %1 : f32
        memref.store %2, %arg1[%arg17, %arg18] : memref<?x?xf32>
      } {mapping = [#gpu.warp<linear_dim_0>, #gpu.warp<linear_dim_1>, #gpu.warp<linear_dim_2>]}

      scf.forall (%arg17, %arg18) in (%dim2, %c2) {
        %1 = memref.load %arg2[%arg17] : memref<?xf32>
        %2 = arith.addf %arg3, %1 : f32
        memref.store %2, %arg2[%arg18] : memref<?xf32>
      } {mapping = [#gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>]}

      gpu.terminator
    }
    return %arg1 : memref<?x?xf32>
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["gpu.launch"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      %1 = transform.gpu.map_nested_forall_to_threads_ext %0 block_dims = [18, 11, 1] : (!transform.any_op) -> !transform.any_op
      transform.yield 
    }
  }
}

// CHECK: #[[$MAP_LINEAR:.*]] = affine_map<(d0, d1) -> (d0 + d1 * 18)>
// CHECK: #[[$MAP_LINEAR_WY:.*]] = affine_map<(d0, d1)[s0, s1] -> ((((d0 + d1 * 18) floordiv 32) mod (s0 * s1)) floordiv s1)>
// CHECK: #[[$MAP_LINEAR_WX:.*]] = affine_map<(d0, d1)[s0, s1] -> ((((d0 + d1 * 18) floordiv 32) mod (s0 * s1)) mod s1)>
// CHECK: #[[$MAP_ACTIVE_0:.*]] = affine_map<()[s0, s1] -> ((s0 * s1) * 32)>
// CHECK: #[[$MAP_LINEAR_TY:.*]] = affine_map<(d0, d1) -> (d1 * 9 + d0 floordiv 2)>
// CHECK: #[[$MAP_LINEAR_TX:.*]] = affine_map<(d0) -> (d0 mod 2)>
// CHECK: #[[$MAP_ACTIVE_2:.*]] = affine_map<()[s0] -> (s0 * 2)>

// CHECK-LABEL: func.func @map_multi_level_linear
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C9:.*]] = arith.constant 9 : index
// CHECK-DAG: %[[DIM0:.*]] = memref.dim %arg0, %[[C0]] : memref<?x?xf32>
// CHECK-DAG: %[[DIM1:.*]] = memref.dim %arg0, %[[C1]] : memref<?x?xf32>
// CHECK-DAG: %[[DIM2:.*]] = memref.dim %arg2, %[[C0]] : memref<?xf32>
// CHECK: %[[LAUNCH:.*]] = gpu.launch async
// CHECK-DAG: %[[TIDX:.*]] = gpu.thread_id  x
// CHECK-DAG: %[[TIDY:.*]] = gpu.thread_id  y
// CHECK-DAG: %[[P1:.*]] = arith.cmpi ult, %[[TIDX]], %[[C9]] : index
// CHECK-DAG: %[[P2:.*]] = arith.cmpi ult, %[[TIDY]], %[[DIM0]] : index
// CHECK-DAG: %[[P3:.*]] = arith.andi %[[P1]], %[[P2]] : i1
// CHECK: scf.if %[[P3]]
        // CHECK-NEXT: %[[V0:.*]] = memref.load %arg0[%[[TIDY]], %[[TIDX]]] : memref<?x?xf32>
        // CHECK-NEXT: %[[V1:.*]] = memref.load %arg1[%[[TIDY]], %[[TIDX]]] : memref<?x?xf32>
        // CHECK-NEXT: %[[V2:.*]] = math.fma %arg3, %[[V0]], %[[V1]] : f32
        // CHECK-NEXT: memref.store %[[V2]], %arg1[%[[TIDY]], %[[TIDX]]] : memref<?x?xf32>

// CHECK-DAG: %[[LINEAR_ID:.*]] = affine.apply #[[$MAP_LINEAR]](%[[TIDX]], %[[TIDY]])
// CHECK-DAG: %[[LINEAR_WY_ID:.*]] = affine.apply #[[$MAP_LINEAR_WY]](%[[TIDX]], %[[TIDY]])[%[[DIM1]], %[[DIM0]]] 
// CHECK-DAG: %[[LINEAR_WX_ID:.*]] = affine.apply #[[$MAP_LINEAR_WX]](%[[TIDX]], %[[TIDY]])[%[[DIM1]], %[[DIM0]]] 
// CHECK-DAG: %[[ACTIVE_SIZE_0:.*]] = affine.apply #[[$MAP_ACTIVE_0]]()[%[[DIM0]], %[[DIM1]]] 
// CHECK-DAG: %[[P4:.*]] = arith.cmpi ult, %[[LINEAR_ID]], %[[ACTIVE_SIZE_0]] : index
// CHECK: scf.if %[[P4]]
        // CHECK-NEXT: %[[V3:.*]] = memref.load %arg0[%[[LINEAR_WX_ID]], %[[LINEAR_WY_ID]]] : memref<?x?xf32>
        // CHECK-NEXT: %[[V4:.*]] = arith.addf %arg3, %[[V3]] : f32
        // CHECK-NEXT: memref.store %[[V4]], %arg1[%[[LINEAR_WX_ID]], %[[LINEAR_WY_ID]]] : memref<?x?xf32>

// CHECK-DAG: %[[LINEAR_TY_ID:.*]] = affine.apply #[[$MAP_LINEAR_TY]](%[[TIDX]], %[[TIDY]])
// CHECK-DAG: %[[LINEAR_TX_ID:.*]] = affine.apply #[[$MAP_LINEAR_TX]](%[[TIDX]])
// CHECK-DAG: %[[ACTIVE_SIZE_2:.*]] = affine.apply #[[$MAP_ACTIVE_2]]()[%[[DIM2]]]
// CHECK-DAG: %[[P5:.*]] = arith.cmpi ult, %[[LINEAR_ID]], %[[ACTIVE_SIZE_2]] : index
// CHECK: scf.if %[[P5]]
        // CHECK-NEXT: %[[V5:.*]] = memref.load %arg2[%[[LINEAR_TY_ID]]] : memref<?xf32>
        // CHECK-NEXT: %[[V6:.*]] = arith.addf %arg3, %[[V5]] : f32
        // CHECK-NEXT: memref.store %[[V6]], %arg2[%[[LINEAR_TX_ID]]] : memref<?xf32>

// -----

module {
  func.func @warpgroup_mapping(%arg0: memref<2x?xf32>, %arg1: memref<2x?xf32>, %arg2: f32, %arg3: !gpu.async.token) -> memref<2x?xf32> {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c1_0 = arith.constant 1 : index
    %dim = memref.dim %arg0, %c1 : memref<2x?xf32>
    %0 = gpu.launch async [%arg3] blocks(%arg4, %arg5, %arg6) in (%arg10 = %c1_0, %arg11 = %c1_0, %arg12 = %c1_0) threads(%arg7, %arg8, %arg9) in (%arg13 = %c1_0, %arg14 = %c1_0, %arg15 = %c1_0) {
      scf.forall (%arg16, %arg17) in (%c2, %dim) {
        %1 = memref.load %arg0[%arg16, %arg17] : memref<2x?xf32>
        %2 = memref.load %arg1[%arg16, %arg17] : memref<2x?xf32>
        %3 = math.fma %arg2, %1, %2 : f32
        memref.store %3, %arg1[%arg16, %arg17] : memref<2x?xf32>
      } {mapping = [#gpu.warpgroup<y>, #gpu.warpgroup<x>]}
      gpu.terminator
    }
    return %arg1 : memref<2x?xf32>
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["gpu.launch"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      %1 = transform.gpu.map_nested_forall_to_threads_ext %0 block_dims = [512, 2, 1] : (!transform.any_op) -> !transform.any_op
      transform.yield 
    }
  }
}

// CHECK: #[[$MAP_SCALE_WG:.*]] = affine_map<(d0) -> (d0 floordiv 128)>

// CHECK-LABEL: func.func @warpgroup_mapping
// CHECK-DAG: %[[C128:.*]] = arith.constant 128 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG: %[[C512:.*]] = arith.constant 512 : index
// CHECK-DAG: %[[DIM:.*]] = memref.dim %arg0, %[[C1]] : memref<2x?xf32>
// CHECK: %[[LAUNCH:.*]] = gpu.launch async
// CHECK-DAG: %[[TIDY:.*]] = gpu.thread_id  y
// CHECK-DAG: %[[WG_ID_X:.*]] = affine.apply #map(%[[TIDX]])
// CHECK-DAG: %[[ACTIVE_SIZE_X:.*]] = arith.muli %[[DIM]], %[[C128]] : index
// CHECK-DAG: %[[P:.*]] = arith.cmpi ult, %[[TIDX]], %[[ACTIVE_SIZE_X]] : index
// CHECK: scf.if %[[P]]
        // CHECK-NEXT: %[[V0:.*]] = memref.load %arg0[%[[TIDY]], %[[WG_ID_X]]] : memref<2x?xf32>
        // CHECK-NEXT: %[[V1:.*]] = memref.load %arg1[%[[TIDY]], %[[WG_ID_X]]] : memref<2x?xf32>
        // CHECK-NEXT: %[[V2:.*]] = math.fma %arg2, %[[V0]], %[[V1]] : f32
        // CHECK-NEXT: memref.store %[[V2]], %arg1[%[[TIDY]], %[[WG_ID_X]]] : memref<2x?xf32>
