// RUN: byteir-opt -convert-to-byre="append-arg-types" -cse %s | FileCheck %s

module {
  func.func @mhlo_add(%arg0: memref<4xf32> {__placeholder__byre.argname = "A"}, %arg1: memref<4xf32> {__placeholder__byre.argname = "B"}) -> (memref<4xf32> {__placeholder__byre.argname = "C"}) attributes { __placeholder__byre.entry_point} {
    %1 = memref.alloc() : memref<4xf32>
    "lmhlo.add"(%arg0, %arg1, %1) : (memref<4xf32>, memref<4xf32>, memref<4xf32>) -> ()
    return %1 : memref<4xf32>
  }
// CHECK-LABEL:  func.func @mhlo_add
// CHECK-NEXT:     byre.compute @AddOpf32f32f32

  func.func @mhlo_add_splat_const(%arg0: memref<4xf32>) -> memref<4xf32> attributes { __placeholder__byre.entry_point} {
    %0 = memref.alloc() : memref<4xf32>
    "lmhlo.constant"(%0) {value = dense<2.000000e+00> : tensor<4xf32>, name = "two"} : (memref<4xf32>) -> ()
    %1 = memref.alloc() : memref<4xf32>
    "lmhlo.add"(%arg0, %0, %1) : (memref<4xf32>, memref<4xf32>, memref<4xf32>) -> ()
    return %1 : memref<4xf32>
  }
// CHECK-LABEL: func.func @mhlo_add_splat_const
// CHECK-NEXT: memref.alloc()
// CHECK-NEXT: byre.compute @FillOp
// CHECK-NEXT: byre.compute @AddOpf32f32f32

  func.func @mhlo_matmul(%arg0: memref<128x64xf32> {__placeholder__byre.argname = "A"}, %arg1: memref<64x32xf32> {__placeholder__byre.argname = "B"}) -> (memref<128x32xf32> {__placeholder__byre.argname = "C"}) attributes { __placeholder__byre.entry_point} {
    %0 = memref.alloc() : memref<128x32xf32>
    "lmhlo.dot"(%arg0, %arg1, %0) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (memref<128x64xf32>, memref<64x32xf32>, memref<128x32xf32>) -> ()
    return %0 : memref<128x32xf32>
  }
// CHECK-LABEL: func.func @mhlo_matmul
// CHECK: byre.compute @MatmulOpf32f32f32

  func.func @mhlo_matmul1(%arg0: memref<128x64xf32> {__placeholder__byre.argname = "A"}, %arg1: memref<64x32xf32> {__placeholder__byre.argname = "B"}) -> (memref<128x32xf32> {__placeholder__byre.argname = "C"}) attributes { __placeholder__byre.entry_point} {
    %0 = memref.alloc() : memref<128x32xf32>
    "lmhlo.dot"(%arg0, %arg1, %0) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [], rhs_batching_dimensions = [], lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (memref<128x64xf32>, memref<64x32xf32>, memref<128x32xf32>) -> ()
    return %0 : memref<128x32xf32>
  }
// CHECK-LABEL: func.func @mhlo_matmul1
// CHECK: byre.compute @MatmulOpf32f32f32

  func.func @mhlo_batch_matmul(%arg0: memref<3x128x64xf32> {__placeholder__byre.argname = "A"}, %arg1: memref<3x64x32xf32> {__placeholder__byre.argname = "B"}) -> (memref<3x128x32xf32> {__placeholder__byre.argname = "C"}) attributes { __placeholder__byre.entry_point} {
    %0 = memref.alloc() : memref<3x128x32xf32>
    "lmhlo.dot"(%arg0, %arg1, %0) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (memref<3x128x64xf32>, memref<3x64x32xf32>, memref<3x128x32xf32>) -> ()
    return %0 : memref<3x128x32xf32>
  }
// CHECK-LABEL: func.func @mhlo_batch_matmul
// CHECK: byre.compute @BatchMatmulOpf32f32f32

  func.func @mhlo_scatter(%arg0: memref<512x128xf32> {__placeholder__byre.argname = "A"}, %arg1: memref<128x1xi64> {__placeholder__byre.argname = "B"}, %arg2: memref<128x128xf32> {__placeholder__byre.argname = "C"}) -> (memref<512x128xf32> {__placeholder__byre.argname = "D"}) attributes { __placeholder__byre.entry_point} {
    %0 = memref.alloc() : memref<512x128xf32>
    "lmhlo.scatter"(%arg0, %arg1, %arg2, %0) ( {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):  // no predecessors
      %1 = mhlo.add %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%1) : (tensor<f32>) -> ()
    }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (memref<512x128xf32>, memref<128x1xi64>, memref<128x128xf32>, memref<512x128xf32>) -> ()
    return %0 : memref<512x128xf32>
  }
  // CHECK-LABEL: mhlo_scatter
  // CHECK-NEXT: byre.compute @IndexPutOpf32i64f32f32

  func.func @mhlo_gather(%arg0: memref<30522x128xf32> {__placeholder__byre.argname = "A"}, %arg1: memref<128xui32> {__placeholder__byre.argname = "B"}) -> (memref<128x128xf32> {__placeholder__byre.argname = "C"}) attributes { __placeholder__byre.entry_point} {
    %2 = memref.alloc() : memref<128x128xf32>
    "lmhlo.gather"(%arg0, %arg1, %2) {dimension_numbers = #mhlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 128]> : tensor<2xi64>} : (memref<30522x128xf32>, memref<128xui32>, memref<128x128xf32>) -> ()
    return %2 : memref<128x128xf32>
  }
  // CHECK-LABEL: mhlo_gather
  // CHECK-NEXT: byre.compute @IndexSelectOpf32ui32f32

  func.func @mhlo_slice_input(%arg0: memref<1x512xi64> {__placeholder__byre.argname = "A"}) -> (memref<1x128xi64> {__placeholder__byre.argname = "B"}) attributes { __placeholder__byre.entry_point} {
    %0 = memref.alloc() : memref<1x128xi64>
    %1 = memref.alloc() : memref<1x128xi64>
    "lmhlo.slice"(%arg0, %1) {limit_indices = dense<[1, 128]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (memref<1x512xi64>, memref<1x128xi64>) -> ()
    "lmhlo.add"(%1, %1, %0) : (memref<1x128xi64>, memref<1x128xi64>, memref<1x128xi64>) -> ()
    return %0 : memref<1x128xi64>
  }
  // CHECK-LABEL: mhlo_slice_input
  // CHECK-NEXT: memref.alloc
  // CHECK-NEXT: byre.compute @AliasOp
  // CHECK-NEXT: byre.compute @AddOpi64i64i64

  func.func @mhlo_reduce(%arg0: memref<1x128x128xf32> {__placeholder__byre.argname = "A"}) -> (memref<128xf32> {__placeholder__byre.argname = "B"}) attributes { __placeholder__byre.entry_point} {
    %0 = memref.alloc() : memref<f32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<f32>} : (memref<f32>) -> ()
    %1 = memref.alloc() : memref<128xf32>
    "lmhlo.reduce"(%arg0, %0, %1) ( {
    ^bb0(%arg1: memref<f32>, %arg2: memref<f32>, %arg3: memref<f32>):  // no predecessors
      "lmhlo.add"(%arg1, %arg2, %arg3) : (memref<f32>, memref<f32>, memref<f32>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (memref<1x128x128xf32>, memref<f32>, memref<128xf32>) -> ()
    return %1 : memref<128xf32>
  }
  // CHECK-LABEL: mhlo_reduce
  // CHECK-NEXT: byre.compute @ReduceSumOpf32f32(%arg0, %arg1)
  //   CHECK-DAG: dimensions = dense<[0, 1]>

  func.func @mhlo_reduce_consecutive_dims(%arg0: memref<2x128x128xf32> {__placeholder__byre.argname = "A"}) -> (memref<128xf32> {__placeholder__byre.argname = "B"}) attributes { __placeholder__byre.entry_point} {
    %0 = memref.alloc() : memref<f32>
    "lmhlo.constant"(%0) {value = dense<0.000000e+00> : tensor<f32>} : (memref<f32>) -> ()
    %1 = memref.alloc() : memref<128xf32>
    "lmhlo.reduce"(%arg0, %0, %1) ( {
    ^bb0(%arg1: memref<f32>, %arg2: memref<f32>, %arg3: memref<f32>):  // no predecessors
      "lmhlo.add"(%arg1, %arg2, %arg3) : (memref<f32>, memref<f32>, memref<f32>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (memref<2x128x128xf32>, memref<f32>, memref<128xf32>) -> ()
    return %1 : memref<128xf32>
  }
  // CHECK-LABEL: mhlo_reduce_consecutive_dims
  // CHECK-NEXT: byre.compute @ReduceSumOpf32f32(%arg0, %arg1)
  //   CHECK-DAG: dimensions = dense<[0, 1]>

  func.func @reduce_window(%arg: memref<1x64x112x112xf32> {__placeholder__byre.argname = "A"}) -> (memref<1x64x56x56xf32> {__placeholder__byre.argname = "B"}) attributes { __placeholder__byre.entry_point} {
    %0 = memref.alloc() : memref<f32>
    "lmhlo.constant"(%0) {value = dense<0xFF800000> : tensor<f32>} : (memref<f32>) -> ()
    %1 = memref.alloc() : memref<1x64x56x56xf32>
    "lmhlo.reduce_window"(%arg, %0, %1) ( {
      ^bb0(%lhs: memref<f32>, %rhs: memref<f32>, %res: memref<f32>):
        "lmhlo.maximum"(%lhs, %rhs, %res)
          : (memref<f32>, memref<f32>, memref<f32>) -> ()
        "lmhlo.terminator"() : () -> ()
      }) {
        padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>,
        window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, 
        window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>
      } : (memref<1x64x112x112xf32>, memref<f32>, memref<1x64x56x56xf32>) -> ()
    return %1 : memref<1x64x56x56xf32>  
  }
  // CHECK-LABEL: reduce_window
  // CHECK-NEXT: byre.compute @PoolMaxOpf32f32(%arg0, %arg1)

  func.func @select_and_scatter(%arg0: memref<32x64x112x112xf16>  {__placeholder__byre.argname = "A"}, %arg1: memref<32x64x56x56xf16> {__placeholder__byre.argname = "B"}) -> (memref<32x64x112x112xf16> {__placeholder__byre.argname = "C"}) attributes { __placeholder__byre.entry_point} {
    %0 = memref.alloc() : memref<32x64x112x112xf16>
    %1 = memref.alloc() : memref<f16>
    "lmhlo.constant"(%1) {value = dense<0.000000e+00> : tensor<f32>} : (memref<f16>) -> ()
    "lmhlo.select_and_scatter"(%arg0, %arg1, %1, %0) ({
    ^bb0(%arg3: tensor<f16>, %arg4: tensor<f16>):  // no predecessors
      %2 = "mhlo.compare"(%arg3, %arg4) {comparison_direction = #mhlo<comparison_direction GE>} : (tensor<f16>, tensor<f16>) -> tensor<i1>
      "mhlo.return"(%2) : (tensor<i1>) -> ()
    }, {
    ^bb0(%arg3: tensor<f16>, %arg4: tensor<f16>):  // no predecessors
      %2 = mhlo.add %arg3, %arg4 : tensor<f16>
      "mhlo.return"(%2) : (tensor<f16>) -> ()
    }) {padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>} : (memref<32x64x112x112xf16>, memref<32x64x56x56xf16>, memref<f16>, memref<32x64x112x112xf16>) -> ()
    return %0 : memref<32x64x112x112xf16>
  }
  // CHECK-LABEL: select_and_scatter
  // CHECK-NEXT: byre.compute @PoolMaxGradOpf16f16f16(%arg0, %arg1, %arg2)

    func.func @convert(%arg0: memref<3x4xf32> {__placeholder__byre.argname = "A"}) -> (memref<3x4xf16> {__placeholder__byre.argname = "B"}) attributes { __placeholder__byre.entry_point} {
    %0 = memref.alloc() : memref<3x4xf16>
    "lmhlo.convert"(%arg0, %0) : (memref<3x4xf32>, memref<3x4xf16>) -> ()
    return %0 : memref<3x4xf16>
  }
  // CHECK-LABEL: convert
  // CHECK-NEXT: byre.compute @Typecvtf32f16

  func.func @test_call(%arg0: memref<4xf32> {__placeholder__byre.argname = "A"}, %arg1: memref<4xf32> {__placeholder__byre.argname = "B"}) -> (memref<4xf32> {__placeholder__byre.argname = "C"}) attributes { __placeholder__byre.entry_point } {
    %0 = call @some_func(%arg0, %arg1) : (memref<4xf32>, memref<4xf32>) -> memref<4xf32>
    return %0 : memref<4xf32>
  }

  func.func private @some_func(%arg0: memref<4xf32>, %arg1: memref<4xf32>) -> memref<4xf32> attributes { byre_compute_name = "customAddOp"}  {
    %0 = memref.alloc() : memref<4xf32>
    "lmhlo.add"(%arg0, %arg1, %0) : (memref<4xf32>, memref<4xf32>, memref<4xf32>) -> ()
    %1 = memref.alloc() : memref<4xf32>
    "lmhlo.add"(%arg0, %0, %1) : (memref<4xf32>, memref<4xf32>, memref<4xf32>) -> ()
    return %1 : memref<4xf32>
  }
  // CHECK-LABEL: test_call
  // CHECK-NEXT: byre.compute @customAddOpf32f32f32

    func.func @test_call_codegen(%arg0: memref<4xf32> {__placeholder__byre.argname = "A"}, %arg1: memref<4xf32> {__placeholder__byre.argname = "B"}) -> (memref<4xf32> {__placeholder__byre.argname = "C"}) attributes { __placeholder__byre.entry_point } {
    %0 = call @some_func_2(%arg0, %arg1) : (memref<4xf32>, memref<4xf32>) -> memref<4xf32>
    return %0 : memref<4xf32>
  }

  func.func private @some_func_2(%arg0: memref<4xf32>, %arg1: memref<4xf32>) -> memref<4xf32> attributes { byre_codegen_op, byre_compute_name = "customAddOp"}  {
    %0 = memref.alloc() : memref<4xf32>
    "lmhlo.add"(%arg0, %arg1, %0) : (memref<4xf32>, memref<4xf32>, memref<4xf32>) -> ()
    %1 = memref.alloc() : memref<4xf32>
    "lmhlo.add"(%arg0, %0, %1) : (memref<4xf32>, memref<4xf32>, memref<4xf32>) -> ()
    return %1 : memref<4xf32>
  }
  // CHECK-LABEL: test_call_codegen
  // CHECK-NEXT: byre.compute @customAddOp
}
