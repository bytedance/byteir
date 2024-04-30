// RUN: byteir-opt %s --transform-interpreter -split-input-file | FileCheck %s

func.func @dynamic_mapping(%x: memref<?x?x768xf32>, %y: memref<?x?x768xf32>,  %alpha : f32, %stream : !gpu.async.token) -> memref<?x?x768xf32> {
  %one = arith.constant 1 : index
  %c0 = arith.constant 0 : index

  
  %c768 = arith.constant 768 : index

  %dim0 = memref.dim %x, %c0 : memref<?x?x768xf32>
  %dim1 = memref.dim %x, %one : memref<?x?x768xf32>
  

  %name = gpu.launch async[%stream] blocks(%arg3, %arg4, %arg5) in (%arg9 = %one, %arg10 = %one, %arg11 = %one)
            threads(%arg6, %arg7, %arg8) in (%arg12 = %one, %arg13 = %one, %arg14 = %one) {
    scf.forall (%i, %j, %k) in (%dim0, %dim1, %c768) {
        %4 = memref.load %x[%i, %j, %k] : memref<?x?x768xf32>
        %5 = memref.load %y[%i, %j, %k] : memref<?x?x768xf32>
        %6 = math.fma %alpha, %4, %5 : f32
        memref.store %6, %y[%i, %j, %k] : memref<?x?x768xf32>
     }  { mapping = [#gpu.thread<z>, #gpu.thread<y>, #gpu.thread<x>] }
    gpu.terminator
  }
  return %y : memref<?x?x768xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %funcop = transform.structured.match ops{["gpu.launch"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.gpu.map_nested_forall_to_threads_ext %funcop block_dims = [32, 4, 2] sync_after_distribute = false : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func.func @dynamic_mapping
// CHECK: scf.for %[[IV0:.*]] = 
        //CHECK: scf.for %[[IV1:.*]] = 
                //CHECK: scf.for %[[IV2:.*]] = 
                //CHECK-NEXT: %[[V0:.*]] = memref.load %arg0[%[[IV0]], %[[IV1]], %[[IV2]]]
                //CHECK-NEXT: %[[V1:.*]] = memref.load %arg1[%[[IV0]], %[[IV1]], %[[IV2]]]
                //CHECK-NEXT: %[[V2:.*]] = math.fma %arg2, %[[V0]], %[[V1]] : f32
                //CHECK-NEXT: memref.store %[[V2]], %arg1[%[[IV0]], %[[IV1]], %[[IV2]]]
