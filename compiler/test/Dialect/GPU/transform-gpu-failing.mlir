// RUN: byteir-opt --transform-interpreter --split-input-file  -canonicalize -cse --verify-diagnostics %s

module {
  func.func @map_forall_to_blocks_not_gpu_launch() {
    // expected-note @below {{when applied to this payload op}}
    %0 = tensor.empty() : tensor<4xf32>
    return
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["tensor.empty"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      // expected-error @below {{Given target is not gpu.launch}}
      %1 = transform.gpu.map_forall_to_blocks_ext %0 : (!transform.any_op) -> !transform.any_op
      transform.yield 
    }
  }
}

// -----

module {
  func.func @saxpy2d_singleloop(%arg0: memref<32x32xf32>, %arg1: memref<32x32xf32>, %arg2: !gpu.async.token) -> memref<32x32xf32> {
    %c32 = arith.constant 32 : index
    %c1 = arith.constant 1 : index
    %0 = gpu.launch async [%arg2] blocks(%arg3, %arg4, %arg5) in (%arg9 = %c1, %arg10 = %c1, %arg11 = %c1) threads(%arg6, %arg7, %arg8) in (%arg12 = %c1, %arg13 = %c1, %arg14 = %c1) {
      scf.forall (%arg15, %arg16) in (%c32, %c32) {
        %1 = memref.load %arg0[%arg15, %arg16] : memref<32x32xf32>
        %2 = memref.load %arg1[%arg15, %arg16] : memref<32x32xf32>
        %3 = arith.mulf %1, %2 : f32
        memref.store %3, %arg1[%arg15, %arg16] : memref<32x32xf32>
      } {mapping = [#gpu.thread<x>, #gpu.thread<x>]}
      gpu.terminator
    }
    return %arg1 : memref<32x32xf32>
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["gpu.launch"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      // expected-error @below {{duplicate attribute, cannot map different loops to the same mapping id}}
      %1 = transform.gpu.map_nested_forall_to_threads_ext %0 block_dims = [32, 32, 1] : (!transform.any_op) -> !transform.any_op
      transform.yield 
    }
  }
}

// -----

module {
  func.func @saxpy2d_singleloop(%arg0: memref<32x32xf32>, %arg1: memref<32x32xf32>, %arg2: !gpu.async.token) -> memref<32x32xf32> {
    %c32 = arith.constant 32 : index
    %c1 = arith.constant 1 : index
    %0 = gpu.launch async [%arg2] blocks(%arg3, %arg4, %arg5) in (%arg9 = %c1, %arg10 = %c1, %arg11 = %c1) threads(%arg6, %arg7, %arg8) in (%arg12 = %c1, %arg13 = %c1, %arg14 = %c1) {
      scf.forall (%arg15, %arg16) in (%c32, %c32) {
        %1 = memref.load %arg0[%arg15, %arg16] : memref<32x32xf32>
        %2 = memref.load %arg1[%arg15, %arg16] : memref<32x32xf32>
        %3 = arith.mulf %1, %2 : f32
        memref.store %3, %arg1[%arg15, %arg16] : memref<32x32xf32>
      } {mapping = [#gpu.thread<x>, #gpu.thread<linear_dim_0>]}
      gpu.terminator
    }
    return %arg1 : memref<32x32xf32>
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["gpu.launch"]} in %arg0 : (!transform.any_op) -> !transform.any_op
       // expected-error @below {{cannot mix linear and non-linear mapping modes}}
      %1 = transform.gpu.map_nested_forall_to_threads_ext %0 block_dims = [32, 32, 1] : (!transform.any_op) -> !transform.any_op
      transform.yield 
    }
  }
}

// -----

func.func public @not_a_block_mapping_attribute(%arg0: memref<32x32xf32>, %arg1: memref<32x32xf32>, %arg2: memref<32x32xf32>) {
  scf.forall (%arg3, %arg4) in (1, 1) {
    linalg.matmul ins(%arg0, %arg1 : memref<32x32xf32>, memref<32x32xf32>) outs(%arg2 : memref<32x32xf32>)
  } {mapping = [#gpu.thread<x>, #gpu.thread<y>]}
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.consumed}) {
    %arg0 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{scf.forall op requires a mapping attribute of kind 'block'}}
    %5 = transform.gpu.map_forall_to_blocks_ext %arg1 generate_gpu_launch grid_dims = [50, 16, 1] : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

func.func @not_a_thread_or_warp_mapping_attribute(%x: memref<2 x 32 x f32>, %y: memref<2 x 32 x f32>, %t: memref<32 x f32>, %alpha : f32, %stream : !gpu.async.token) -> memref<2 x 32 x f32> {
  %one = arith.constant 1 : index
  %c900 = arith.constant 900 : index
  %c9 = arith.constant 9 : index
  %c7 = arith.constant 7 : index
  %name = gpu.launch async[%stream] blocks(%arg3, %arg4, %arg5) in (%arg9 = %one, %arg10 = %one, %arg11 = %one)
            threads(%arg6, %arg7, %arg8) in (%arg12 = %one, %arg13 = %one, %arg14 = %one)
  {
    scf.forall (%i, %j) in (%c7, %c900) {
        %4 = memref.load %x[%i, %j] : memref<2 x 32 x f32>
        %5 = memref.load %y[%i, %j] : memref<2 x 32 x f32>
        %6 = math.fma %alpha, %4, %5 : f32
        memref.store %6, %y[%i, %j] : memref<2 x 32 x f32>
     }  { mapping = [#gpu.block<y>, #gpu.block<x>] }
    gpu.terminator
  }

  return %y : memref<2 x 32 x f32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %funcop = transform.structured.match ops{["gpu.launch"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{scf.forall op requires a mapping attribute of kind 'thread' or 'warp'}}
    transform.gpu.map_nested_forall_to_threads_ext %funcop block_dims = [1, 1, 1] : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

module {
  func.func @map_nested_forall_to_threads_excessive_threads(%arg0: memref<2x32xf32>, %arg1: memref<2x32xf32>, %arg2: memref<32xf32>, %arg3: f32, %arg4: !gpu.async.token) -> memref<2x32xf32> {
    %c1 = arith.constant 1 : index
    %c900 = arith.constant 900 : index
    %c9 = arith.constant 9 : index
    %c7 = arith.constant 7 : index
    %0 = gpu.launch async [%arg4] blocks(%arg5, %arg6, %arg7) in (%arg11 = %c1, %arg12 = %c1, %arg13 = %c1) threads(%arg8, %arg9, %arg10) in (%arg14 = %c1, %arg15 = %c1, %arg16 = %c1) {
      scf.forall (%arg17, %arg18) in (%c7, %c900) {
        %2 = memref.load %arg0[%arg17, %arg18] : memref<2x32xf32>
        %3 = memref.load %arg1[%arg17, %arg18] : memref<2x32xf32>
        %4 = math.fma %arg3, %2, %3 : f32
        memref.store %4, %arg1[%arg17, %arg18] : memref<2x32xf32>
      } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
      gpu.terminator
    }
    %1 = gpu.launch async [%arg4] blocks(%arg5, %arg6, %arg7) in (%arg11 = %c1, %arg12 = %c1, %arg13 = %c1) threads(%arg8, %arg9, %arg10) in (%arg14 = %c1, %arg15 = %c1, %arg16 = %c1) {
      scf.forall (%arg17, %arg18) in (%c7, %c9) {
        %2 = memref.load %arg0[%arg17, %arg18] : memref<2x32xf32>
        %3 = memref.load %arg1[%arg17, %arg18] : memref<2x32xf32>
        %4 = math.fma %arg3, %2, %3 : f32
        memref.store %4, %arg1[%arg17, %arg18] : memref<2x32xf32>
      } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
      gpu.terminator
    }
    return %arg1 : memref<2x32xf32>
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["gpu.launch"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      // expected-error @below {{Trying to launch a GPU kernel with grid_dims = (1, 1, 1) block_dims = (1200, 9, 1). It is larger than the limits.}}
      // expected-note @below {{"block_dims" is too large}}
      %1 = transform.gpu.map_nested_forall_to_threads_ext %0 block_dims = [1200, 9, 1] : (!transform.any_op) -> !transform.any_op
      transform.yield 
    }
  }
}

// -----

module {
  func.func @map_nested_forall_to_threads_fewer_threads(%arg0: memref<2x32xf32>, %arg1: memref<2x32xf32>, %arg2: memref<32xf32>, %arg3: f32, %arg4: !gpu.async.token) -> memref<2x32xf32> {
    %c1 = arith.constant 1 : index
    %c900 = arith.constant 900 : index
    %c9 = arith.constant 9 : index
    %c7 = arith.constant 7 : index
    %0 = gpu.launch async [%arg4] blocks(%arg5, %arg6, %arg7) in (%arg11 = %c1, %arg12 = %c1, %arg13 = %c1) threads(%arg8, %arg9, %arg10) in (%arg14 = %c1, %arg15 = %c1, %arg16 = %c1) {
      scf.forall (%arg17, %arg18) in (%c7, %c900) {
        %2 = memref.load %arg0[%arg17, %arg18] : memref<2x32xf32>
        %3 = memref.load %arg1[%arg17, %arg18] : memref<2x32xf32>
        %4 = math.fma %arg3, %2, %3 : f32
        memref.store %4, %arg1[%arg17, %arg18] : memref<2x32xf32>
      } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
      gpu.terminator
    }
    %1 = gpu.launch async [%arg4] blocks(%arg5, %arg6, %arg7) in (%arg11 = %c1, %arg12 = %c1, %arg13 = %c1) threads(%arg8, %arg9, %arg10) in (%arg14 = %c1, %arg15 = %c1, %arg16 = %c1) {
      scf.forall (%arg17, %arg18) in (%c7, %c9) {
        %2 = memref.load %arg0[%arg17, %arg18] : memref<2x32xf32>
        %3 = memref.load %arg1[%arg17, %arg18] : memref<2x32xf32>
        %4 = math.fma %arg3, %2, %3 : f32
        memref.store %4, %arg1[%arg17, %arg18] : memref<2x32xf32>
      } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
      gpu.terminator
    }
    return %arg1 : memref<2x32xf32>
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["gpu.launch"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      // expected-error @below {{the number of required parallel resources (blocks or threads) 6300 overflows the number of available resources 512}}
      %1 = transform.gpu.map_nested_forall_to_threads_ext %0 block_dims = [128, 4, 1] : (!transform.any_op) -> !transform.any_op
      transform.yield 
    }
  }
}

// -----

module {
  func.func @map_forall_to_blocks_not_unique(%arg0: memref<2x32xf32>, %arg1: memref<2x32xf32>, %arg2: memref<32xf32>, %arg3: f32, %arg4: !gpu.async.token) -> memref<2x32xf32> {
    %c1 = arith.constant 1 : index
    %c900 = arith.constant 900 : index
    %c9 = arith.constant 9 : index
    %c7 = arith.constant 7 : index
    // expected-note @below {{when applied to this payload op}}
    %0 = gpu.launch async [%arg4] blocks(%arg5, %arg6, %arg7) in (%arg11 = %c1, %arg12 = %c1, %arg13 = %c1) threads(%arg8, %arg9, %arg10) in (%arg14 = %c1, %arg15 = %c1, %arg16 = %c1) {
      scf.forall (%arg17, %arg18) in (%c7, %c900) {
        %1 = memref.load %arg0[%arg17, %arg18] : memref<2x32xf32>
        %2 = memref.load %arg1[%arg17, %arg18] : memref<2x32xf32>
        %3 = math.fma %arg3, %1, %2 : f32
        memref.store %3, %arg1[%arg17, %arg18] : memref<2x32xf32>
      } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
      scf.forall (%arg17, %arg18) in (%c7, %c9) {
        %1 = memref.load %arg0[%arg17, %arg18] : memref<2x32xf32>
        %2 = memref.load %arg1[%arg17, %arg18] : memref<2x32xf32>
        %3 = math.fma %arg3, %1, %2 : f32
        memref.store %3, %arg1[%arg17, %arg18] : memref<2x32xf32>
      } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
      gpu.terminator
    }
    return %arg1 : memref<2x32xf32>
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["gpu.launch"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      // expected-error @below {{could not find a unique topLevel scf.forall}}
      %1 = transform.gpu.map_forall_to_blocks_ext %0 : (!transform.any_op) -> !transform.any_op
      transform.yield 
    }
  }
}

// -----

module {
  func.func @saxpy2d_singleloop(%arg0: memref<32x32xf32>, %arg1: memref<32x32xf32>, %arg2: !gpu.async.token) -> memref<32x32xf32> {
    %c32 = arith.constant 32 : index
    %c1 = arith.constant 1 : index
    %0 = gpu.launch async [%arg2] blocks(%arg3, %arg4, %arg5) in (%arg9 = %c1, %arg10 = %c1, %arg11 = %c1) threads(%arg6, %arg7, %arg8) in (%arg12 = %c1, %arg13 = %c1, %arg14 = %c1) {
      scf.forall (%arg15, %arg16) in (%c32, %c32) {
        %1 = memref.load %arg0[%arg15, %arg16] : memref<32x32xf32>
        %2 = memref.load %arg1[%arg15, %arg16] : memref<32x32xf32>
        %3 = arith.mulf %1, %2 : f32
        memref.store %3, %arg1[%arg15, %arg16] : memref<32x32xf32>
      } {mapping = [#gpu.thread<x>, #gpu.thread<linear_dim_0>]}
      gpu.terminator
    }
    return %arg1 : memref<32x32xf32>
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["gpu.launch"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      // expected-error @below {{cannot mix linear and non-linear mapping modes}}
      %1 = transform.gpu.map_nested_forall_to_threads %0 block_dims = [32, 32, 1] : (!transform.any_op) -> !transform.any_op
      transform.yield 
    }
  }
}
