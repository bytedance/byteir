// RUN: byteir-opt %s -hlo-opt="outline-single-elemwise-op" | FileCheck %s

// CHECK-LABEL: func.func @main
module attributes {torch.debug_module_name = "GraphModule"} {
  func.func @main(%arg0: tensor<30522x128xf32>, %arg1: tensor<512x128xf32>, %arg2: tensor<2x128xf32>, %arg3: tensor<128xf32>, %arg4: tensor<128xf32>, %arg5: tensor<128x128xf32>, %arg6: tensor<128xf32>, %arg7: tensor<128x128xf32>, %arg8: tensor<128xf32>, %arg9: tensor<128x128xf32>, %arg10: tensor<128xf32>, %arg11: tensor<128x128xf32>, %arg12: tensor<128xf32>, %arg13: tensor<128xf32>, %arg14: tensor<128xf32>, %arg15: tensor<512x128xf32>, %arg16: tensor<512xf32>, %arg17: tensor<128x512xf32>, %arg18: tensor<128xf32>, %arg19: tensor<128xf32>, %arg20: tensor<128xf32>, %arg21: tensor<128x128xf32>, %arg22: tensor<128xf32>, %arg23: tensor<128x128xf32>, %arg24: tensor<128xf32>, %arg25: tensor<128x128xf32>, %arg26: tensor<128xf32>, %arg27: tensor<128x128xf32>, %arg28: tensor<128xf32>, %arg29: tensor<128xf32>, %arg30: tensor<128xf32>, %arg31: tensor<512x128xf32>, %arg32: tensor<512xf32>, %arg33: tensor<128x512xf32>, %arg34: tensor<128xf32>, %arg35: tensor<128xf32>, %arg36: tensor<128xf32>, %arg37: tensor<30522xf32>, %arg38: tensor<128x128xf32>, %arg39: tensor<128xf32>, %arg40: tensor<128xf32>, %arg41: tensor<128xf32>, %arg42: tensor<30522x128xf32>, %arg43: tensor<30522xf32>, %arg44: tensor<1x512xi64>, %arg45: tensor<1x512xi64>, %arg46: tensor<2x128xi64>) -> (tensor<2x128x30522xf32>, tensor<256x30522xf32>, tensor<4x128x64xf32>, tensor<128xf32>, tensor<2x128x512xf32>, tensor<128xf32>, tensor<4x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<2x128x1xf32>, tensor<2x128xi64>, tensor<2x2x128x128xf32>, tensor<256x128xf32>, tensor<256x128xf32>, tensor<128x512xf32>, tensor<128x512xf32>, tensor<4x128x64xf32>, tensor<2x128xi64>, tensor<128x128xf32>, tensor<2x128x1xf32>, tensor<256x512xf32>, tensor<128x128xf32>, tensor<2x2x128x128xf32>, tensor<2x128x512xf32>, tensor<2x128x1xf32>, tensor<4x64x128xf32>, tensor<2x128x1xf32>, tensor<4x128x64xf32>, tensor<128x128xf32>, tensor<128x128xf32>, tensor<2x128x128xf32>, tensor<256x128xf32>, tensor<256x128xf32>, tensor<256x128xf32>, tensor<2x128x1xf32>, tensor<2x128x1xf32>, tensor<128xf32>, tensor<2x128x128xf32>, tensor<128x30522xf32>, tensor<256x128xf32>, tensor<2x128x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128xf32>, tensor<2x128x128xf32>, tensor<256x128xf32>, tensor<4x128x128xf32>, tensor<4x128x64xf32>, tensor<256x128xf32>, tensor<2x128x1xf32>, tensor<1x128xi64>, tensor<256x512xf32>, tensor<2x128x1xf32>, tensor<2x128x128xf32>, tensor<4x64x128xf32>, tensor<128x128xf32>, tensor<128x128xf32>, tensor<2x128x1xf32>, tensor<2x128x1xf32>, tensor<512x128xf32>, tensor<256x128xf32>, tensor<2x128x128xf32>, tensor<256x128xf32>, tensor<256x128xf32>, tensor<128x128xf32>, tensor<128x128xf32>, tensor<2x128x128xf32>, tensor<512x128xf32>, tensor<2x128x128xf32>, tensor<2x128x1xf32>) {
    %0 = mhlo.constant dense<8.000000e+00> : tensor<2x2x128x128xf32>
    %1 = "mhlo.slice"(%arg45) {limit_indices = dense<[1, 128]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<1x512xi64>) -> tensor<1x128xi64>
    %2 = "mhlo.broadcast_in_dim"(%1) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x128xi64>) -> tensor<2x128xi64>
    %3 = "mhlo.slice"(%arg44) {limit_indices = dense<[1, 128]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<1x512xi64>) -> tensor<1x128xi64>
    %4 = "mhlo.gather"(%arg0, %arg46) {dimension_numbers = #mhlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = dense<[1, 128]> : tensor<2xi64>} : (tensor<30522x128xf32>, tensor<2x128xi64>) -> tensor<2x128x128xf32>
    %5 = "mhlo.gather"(%arg2, %2) {dimension_numbers = #mhlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = dense<[1, 128]> : tensor<2xi64>} : (tensor<2x128xf32>, tensor<2x128xi64>) -> tensor<2x128x128xf32>
    %6 = mhlo.add %4, %5 : tensor<2x128x128xf32>
    %7 = "mhlo.gather"(%arg1, %3) {dimension_numbers = #mhlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = dense<[1, 128]> : tensor<2xi64>} : (tensor<512x128xf32>, tensor<1x128xi64>) -> tensor<1x128x128xf32>
    %8 = "mhlo.broadcast_in_dim"(%7) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<1x128x128xf32>) -> tensor<2x128x128xf32>
    %9 = mhlo.add %6, %8 : tensor<2x128x128xf32>
    %10:3 = mhlo.custom_call @byteir.layer_norm(%9, %arg3, %arg4) {backend_config = "", byteir_attrs = {axis = [2], epsilon = 9.9999999999999998E-13 : f64}} : (tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>) -> (tensor<2x128x128xf32>, tensor<2x128x1xf32>, tensor<2x128x1xf32>)
    %11 = "mhlo.transpose"(%arg5) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<128x128xf32>) -> tensor<128x128xf32>
    %12 = mhlo.reshape %10#0 : (tensor<2x128x128xf32>) -> tensor<256x128xf32>
    %13 = "mhlo.dot"(%12, %11) : (tensor<256x128xf32>, tensor<128x128xf32>) -> tensor<256x128xf32>
    %14 = "mhlo.broadcast_in_dim"(%arg6) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<256x128xf32>
    %15 = mhlo.add %14, %13 : tensor<256x128xf32>
    %16 = "mhlo.transpose"(%arg7) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<128x128xf32>) -> tensor<128x128xf32>
    %17 = "mhlo.dot"(%12, %16) : (tensor<256x128xf32>, tensor<128x128xf32>) -> tensor<256x128xf32>
    %18 = "mhlo.broadcast_in_dim"(%arg8) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<256x128xf32>
    %19 = mhlo.add %18, %17 : tensor<256x128xf32>
    %20 = mhlo.reshape %19 : (tensor<256x128xf32>) -> tensor<2x128x2x64xf32>
    %21 = "mhlo.transpose"(%arg9) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<128x128xf32>) -> tensor<128x128xf32>
    %22 = "mhlo.dot"(%12, %21) : (tensor<256x128xf32>, tensor<128x128xf32>) -> tensor<256x128xf32>
    %23 = "mhlo.broadcast_in_dim"(%arg10) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<256x128xf32>
    %24 = mhlo.add %23, %22 : tensor<256x128xf32>
    %25 = mhlo.reshape %24 : (tensor<256x128xf32>) -> tensor<2x128x2x64xf32>
    %26 = "mhlo.transpose"(%25) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<2x128x2x64xf32>) -> tensor<2x2x128x64xf32>
    %27 = mhlo.reshape %15 : (tensor<256x128xf32>) -> tensor<2x128x2x64xf32>
    %28 = "mhlo.transpose"(%27) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<2x128x2x64xf32>) -> tensor<2x2x128x64xf32>
    %29 = "mhlo.transpose"(%20) {permutation = dense<[0, 2, 3, 1]> : tensor<4xi64>} : (tensor<2x128x2x64xf32>) -> tensor<2x2x64x128xf32>
    %30 = mhlo.reshape %28 : (tensor<2x2x128x64xf32>) -> tensor<4x128x64xf32>
    %31 = mhlo.reshape %29 : (tensor<2x2x64x128xf32>) -> tensor<4x64x128xf32>
    %32 = "mhlo.dot_general"(%30, %31) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<4x128x64xf32>, tensor<4x64x128xf32>) -> tensor<4x128x128xf32>
    %33 = mhlo.reshape %32 : (tensor<4x128x128xf32>) -> tensor<2x2x128x128xf32>
    %34 = mhlo.divide %33, %0 : tensor<2x2x128x128xf32>
    %35 = mhlo.custom_call @byteir.softmax(%34) {backend_config = "", byteir_attrs = {axis = 3 : i64}} : (tensor<2x2x128x128xf32>) -> tensor<2x2x128x128xf32>
    %36 = mhlo.reshape %35 : (tensor<2x2x128x128xf32>) -> tensor<4x128x128xf32>
    %37 = mhlo.reshape %26 : (tensor<2x2x128x64xf32>) -> tensor<4x128x64xf32>
    %38 = "mhlo.dot_general"(%36, %37) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<4x128x128xf32>, tensor<4x128x64xf32>) -> tensor<4x128x64xf32>
    %39 = mhlo.reshape %38 : (tensor<4x128x64xf32>) -> tensor<2x2x128x64xf32>
    %40 = "mhlo.transpose"(%39) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<2x2x128x64xf32>) -> tensor<2x128x2x64xf32>
    %41 = "mhlo.transpose"(%arg11) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<128x128xf32>) -> tensor<128x128xf32>
    %42 = mhlo.reshape %40 : (tensor<2x128x2x64xf32>) -> tensor<256x128xf32>
    %43 = "mhlo.dot"(%42, %41) : (tensor<256x128xf32>, tensor<128x128xf32>) -> tensor<256x128xf32>
    %44 = "mhlo.broadcast_in_dim"(%arg12) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<256x128xf32>
    %45 = mhlo.add %44, %43 : tensor<256x128xf32>
    %46 = mhlo.reshape %45 : (tensor<256x128xf32>) -> tensor<2x128x128xf32>
    %47 = mhlo.add %46, %10#0 : tensor<2x128x128xf32>
    %48:3 = mhlo.custom_call @byteir.layer_norm(%47, %arg13, %arg14) {backend_config = "", byteir_attrs = {axis = [2], epsilon = 9.9999999999999998E-13 : f64}} : (tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>) -> (tensor<2x128x128xf32>, tensor<2x128x1xf32>, tensor<2x128x1xf32>)
    %49 = "mhlo.transpose"(%arg15) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<512x128xf32>) -> tensor<128x512xf32>
    %50 = mhlo.reshape %48#0 : (tensor<2x128x128xf32>) -> tensor<256x128xf32>
    %51 = "mhlo.dot"(%50, %49) : (tensor<256x128xf32>, tensor<128x512xf32>) -> tensor<256x512xf32>
    %52 = "mhlo.broadcast_in_dim"(%arg16) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<256x512xf32>
    %53 = mhlo.add %52, %51 : tensor<256x512xf32>
    %54 = mhlo.reshape %53 : (tensor<256x512xf32>) -> tensor<2x128x512xf32>
    %55 = mhlo.custom_call @byteir.gelu(%54) {backend_config = "", byteir_attrs = {approximate = "erf"}} : (tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
    %56 = "mhlo.transpose"(%arg17) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<128x512xf32>) -> tensor<512x128xf32>
    %57 = mhlo.reshape %55 : (tensor<2x128x512xf32>) -> tensor<256x512xf32>
    %58 = "mhlo.dot"(%57, %56) : (tensor<256x512xf32>, tensor<512x128xf32>) -> tensor<256x128xf32>
    %59 = "mhlo.broadcast_in_dim"(%arg18) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<256x128xf32>
    %60 = mhlo.add %59, %58 : tensor<256x128xf32>
    %61 = mhlo.reshape %60 : (tensor<256x128xf32>) -> tensor<2x128x128xf32>
    %62 = mhlo.add %61, %48#0 : tensor<2x128x128xf32>
    %63:3 = mhlo.custom_call @byteir.layer_norm(%62, %arg19, %arg20) {backend_config = "", byteir_attrs = {axis = [2], epsilon = 9.9999999999999998E-13 : f64}} : (tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>) -> (tensor<2x128x128xf32>, tensor<2x128x1xf32>, tensor<2x128x1xf32>)
    %64 = "mhlo.transpose"(%arg21) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<128x128xf32>) -> tensor<128x128xf32>
    %65 = mhlo.reshape %63#0 : (tensor<2x128x128xf32>) -> tensor<256x128xf32>
    %66 = "mhlo.dot"(%65, %64) : (tensor<256x128xf32>, tensor<128x128xf32>) -> tensor<256x128xf32>
    %67 = "mhlo.broadcast_in_dim"(%arg22) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<256x128xf32>
    %68 = mhlo.add %67, %66 : tensor<256x128xf32>
    %69 = "mhlo.transpose"(%arg23) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<128x128xf32>) -> tensor<128x128xf32>
    %70 = "mhlo.dot"(%65, %69) : (tensor<256x128xf32>, tensor<128x128xf32>) -> tensor<256x128xf32>
    %71 = "mhlo.broadcast_in_dim"(%arg24) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<256x128xf32>
    %72 = mhlo.add %71, %70 : tensor<256x128xf32>
    %73 = mhlo.reshape %72 : (tensor<256x128xf32>) -> tensor<2x128x2x64xf32>
    %74 = "mhlo.transpose"(%arg25) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<128x128xf32>) -> tensor<128x128xf32>
    %75 = "mhlo.dot"(%65, %74) : (tensor<256x128xf32>, tensor<128x128xf32>) -> tensor<256x128xf32>
    %76 = "mhlo.broadcast_in_dim"(%arg26) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<256x128xf32>
    %77 = mhlo.add %76, %75 : tensor<256x128xf32>
    %78 = mhlo.reshape %77 : (tensor<256x128xf32>) -> tensor<2x128x2x64xf32>
    %79 = "mhlo.transpose"(%78) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<2x128x2x64xf32>) -> tensor<2x2x128x64xf32>
    %80 = mhlo.reshape %68 : (tensor<256x128xf32>) -> tensor<2x128x2x64xf32>
    %81 = "mhlo.transpose"(%80) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<2x128x2x64xf32>) -> tensor<2x2x128x64xf32>
    %82 = "mhlo.transpose"(%73) {permutation = dense<[0, 2, 3, 1]> : tensor<4xi64>} : (tensor<2x128x2x64xf32>) -> tensor<2x2x64x128xf32>
    %83 = mhlo.reshape %81 : (tensor<2x2x128x64xf32>) -> tensor<4x128x64xf32>
    %84 = mhlo.reshape %82 : (tensor<2x2x64x128xf32>) -> tensor<4x64x128xf32>
    %85 = "mhlo.dot_general"(%83, %84) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<4x128x64xf32>, tensor<4x64x128xf32>) -> tensor<4x128x128xf32>
    %86 = mhlo.reshape %85 : (tensor<4x128x128xf32>) -> tensor<2x2x128x128xf32>
    %87 = mhlo.divide %86, %0 : tensor<2x2x128x128xf32>
    %88 = mhlo.custom_call @byteir.softmax(%87) {backend_config = "", byteir_attrs = {axis = 3 : i64}} : (tensor<2x2x128x128xf32>) -> tensor<2x2x128x128xf32>
    %89 = mhlo.reshape %88 : (tensor<2x2x128x128xf32>) -> tensor<4x128x128xf32>
    %90 = mhlo.reshape %79 : (tensor<2x2x128x64xf32>) -> tensor<4x128x64xf32>
    %91 = "mhlo.dot_general"(%89, %90) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<4x128x128xf32>, tensor<4x128x64xf32>) -> tensor<4x128x64xf32>
    %92 = mhlo.reshape %91 : (tensor<4x128x64xf32>) -> tensor<2x2x128x64xf32>
    %93 = "mhlo.transpose"(%92) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<2x2x128x64xf32>) -> tensor<2x128x2x64xf32>
    %94 = "mhlo.transpose"(%arg27) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<128x128xf32>) -> tensor<128x128xf32>
    %95 = mhlo.reshape %93 : (tensor<2x128x2x64xf32>) -> tensor<256x128xf32>
    %96 = "mhlo.dot"(%95, %94) : (tensor<256x128xf32>, tensor<128x128xf32>) -> tensor<256x128xf32>
    %97 = "mhlo.broadcast_in_dim"(%arg28) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<256x128xf32>
    %98 = mhlo.add %97, %96 : tensor<256x128xf32>
    %99 = mhlo.reshape %98 : (tensor<256x128xf32>) -> tensor<2x128x128xf32>
    %100 = mhlo.add %99, %63#0 : tensor<2x128x128xf32>
    %101:3 = mhlo.custom_call @byteir.layer_norm(%100, %arg29, %arg30) {backend_config = "", byteir_attrs = {axis = [2], epsilon = 9.9999999999999998E-13 : f64}} : (tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>) -> (tensor<2x128x128xf32>, tensor<2x128x1xf32>, tensor<2x128x1xf32>)
    %102 = "mhlo.transpose"(%arg31) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<512x128xf32>) -> tensor<128x512xf32>
    %103 = mhlo.reshape %101#0 : (tensor<2x128x128xf32>) -> tensor<256x128xf32>
    %104 = "mhlo.dot"(%103, %102) : (tensor<256x128xf32>, tensor<128x512xf32>) -> tensor<256x512xf32>
    %105 = "mhlo.broadcast_in_dim"(%arg32) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<512xf32>) -> tensor<256x512xf32>
    %106 = mhlo.add %105, %104 : tensor<256x512xf32>
    %107 = mhlo.reshape %106 : (tensor<256x512xf32>) -> tensor<2x128x512xf32>
    %108 = mhlo.custom_call @byteir.gelu(%107) {backend_config = "", byteir_attrs = {approximate = "erf"}} : (tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
    %109 = "mhlo.transpose"(%arg33) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<128x512xf32>) -> tensor<512x128xf32>
    %110 = mhlo.reshape %108 : (tensor<2x128x512xf32>) -> tensor<256x512xf32>
    %111 = "mhlo.dot"(%110, %109) : (tensor<256x512xf32>, tensor<512x128xf32>) -> tensor<256x128xf32>
    %112 = "mhlo.broadcast_in_dim"(%arg34) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<256x128xf32>
    %113 = mhlo.add %112, %111 : tensor<256x128xf32>
    %114 = mhlo.reshape %113 : (tensor<256x128xf32>) -> tensor<2x128x128xf32>
    %115 = mhlo.add %114, %101#0 : tensor<2x128x128xf32>
    %116:3 = mhlo.custom_call @byteir.layer_norm(%115, %arg35, %arg36) {backend_config = "", byteir_attrs = {axis = [2], epsilon = 9.9999999999999998E-13 : f64}} : (tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>) -> (tensor<2x128x128xf32>, tensor<2x128x1xf32>, tensor<2x128x1xf32>)
    %117 = "mhlo.transpose"(%arg38) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<128x128xf32>) -> tensor<128x128xf32>
    %118 = mhlo.reshape %116#0 : (tensor<2x128x128xf32>) -> tensor<256x128xf32>
    %119 = "mhlo.dot"(%118, %117) : (tensor<256x128xf32>, tensor<128x128xf32>) -> tensor<256x128xf32>
    %120 = "mhlo.broadcast_in_dim"(%arg39) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xf32>) -> tensor<256x128xf32>
    %121 = mhlo.add %120, %119 : tensor<256x128xf32>
    %122 = mhlo.reshape %121 : (tensor<256x128xf32>) -> tensor<2x128x128xf32>
    %123 = mhlo.custom_call @byteir.gelu(%122) {backend_config = "", byteir_attrs = {approximate = "erf"}} : (tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %124:3 = mhlo.custom_call @byteir.layer_norm(%123, %arg40, %arg41) {backend_config = "", byteir_attrs = {axis = [2], epsilon = 9.9999999999999998E-13 : f64}} : (tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>) -> (tensor<2x128x128xf32>, tensor<2x128x1xf32>, tensor<2x128x1xf32>)
    %125 = "mhlo.transpose"(%arg42) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<30522x128xf32>) -> tensor<128x30522xf32>
    %126 = mhlo.reshape %124#0 : (tensor<2x128x128xf32>) -> tensor<256x128xf32>
    %127 = "mhlo.dot"(%126, %125) : (tensor<256x128xf32>, tensor<128x30522xf32>) -> tensor<256x30522xf32>
    %128 = "mhlo.broadcast_in_dim"(%arg43) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<30522xf32>) -> tensor<256x30522xf32>
    %129 = mhlo.add %128, %127 : tensor<256x30522xf32>
    %130 = mhlo.reshape %129 : (tensor<256x30522xf32>) -> tensor<2x128x30522xf32>
    return %130, %129, %90, %arg13, %107, %arg3, %36, %69, %arg35, %48#1, %arg46, %35, %42, %126, %102, %49, %30, %2, %41, %48#2, %57, %16, %88, %54, %63#1, %84, %10#1, %37, %64, %21, %115, %12, %65, %65, %116#2, %124#1, %arg29, %47, %125, %50, %63#2, %arg19, %arg40, %12, %62, %65, %89, %83, %103, %124#2, %3, %110, %101#2, %122, %31, %74, %11, %101#1, %10#2, %56, %12, %100, %118, %95, %117, %94, %9, %109, %123, %116#1 : tensor<2x128x30522xf32>, tensor<256x30522xf32>, tensor<4x128x64xf32>, tensor<128xf32>, tensor<2x128x512xf32>, tensor<128xf32>, tensor<4x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<2x128x1xf32>, tensor<2x128xi64>, tensor<2x2x128x128xf32>, tensor<256x128xf32>, tensor<256x128xf32>, tensor<128x512xf32>, tensor<128x512xf32>, tensor<4x128x64xf32>, tensor<2x128xi64>, tensor<128x128xf32>, tensor<2x128x1xf32>, tensor<256x512xf32>, tensor<128x128xf32>, tensor<2x2x128x128xf32>, tensor<2x128x512xf32>, tensor<2x128x1xf32>, tensor<4x64x128xf32>, tensor<2x128x1xf32>, tensor<4x128x64xf32>, tensor<128x128xf32>, tensor<128x128xf32>, tensor<2x128x128xf32>, tensor<256x128xf32>, tensor<256x128xf32>, tensor<256x128xf32>, tensor<2x128x1xf32>, tensor<2x128x1xf32>, tensor<128xf32>, tensor<2x128x128xf32>, tensor<128x30522xf32>, tensor<256x128xf32>, tensor<2x128x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128xf32>, tensor<2x128x128xf32>, tensor<256x128xf32>, tensor<4x128x128xf32>, tensor<4x128x64xf32>, tensor<256x128xf32>, tensor<2x128x1xf32>, tensor<1x128xi64>, tensor<256x512xf32>, tensor<2x128x1xf32>, tensor<2x128x128xf32>, tensor<4x64x128xf32>, tensor<128x128xf32>, tensor<128x128xf32>, tensor<2x128x1xf32>, tensor<2x128x1xf32>, tensor<512x128xf32>, tensor<256x128xf32>, tensor<2x128x128xf32>, tensor<256x128xf32>, tensor<256x128xf32>, tensor<128x128xf32>, tensor<128x128xf32>, tensor<2x128x128xf32>, tensor<512x128xf32>, tensor<2x128x128xf32>, tensor<2x128x1xf32>
  }
}

