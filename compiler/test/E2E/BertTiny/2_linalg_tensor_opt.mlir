// RUN: byteir-opt %s -linalg-tensor-opt | FileCheck %s

// CHECK-LABEL: func.func @main
module {
  func.func private @Unknown0(%arg0: tensor<2x128xi64>) -> (tensor<256xi64>, tensor<256xi1>) attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<-100> : tensor<256xi64>
    %1 = mhlo.reshape %arg0 : (tensor<2x128xi64>) -> tensor<256xi64>
    %2 = mhlo.compare  NE, %1, %0 : (tensor<256xi64>, tensor<256xi64>) -> tensor<256xi1>
    return %1, %2 : tensor<256xi64>, tensor<256xi1>
  }
  func.func private @Unknown1(%arg0: tensor<2x128xi64>) -> (tensor<256xui32>, tensor<256x1xi64>, tensor<256xi1>) attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0> : tensor<256xi64>
    %1 = mhlo.constant dense<0.000000e+00> : tensor<256xf64>
    %2 = mhlo.constant dense<30522> : tensor<256xi64>
    %3 = mhlo.reshape %arg0 : (tensor<2x128xi64>) -> tensor<256xi64>
    %4 = mhlo.convert %3 : (tensor<256xi64>) -> tensor<256xui32>
    %5 = mhlo.compare  LT, %3, %0 : (tensor<256xi64>, tensor<256xi64>) -> tensor<256xi1>
    %6 = mhlo.add %3, %2 : tensor<256xi64>
    %7 = mhlo.select %5, %6, %3 : tensor<256xi1>, tensor<256xi64>
    %8 = mhlo.reshape %7 : (tensor<256xi64>) -> tensor<256x1xi64>
    %9 = mhlo.convert %3 : (tensor<256xi64>) -> tensor<256xf64>
    %10 = mhlo.compare  NE, %9, %1 : (tensor<256xf64>, tensor<256xf64>) -> tensor<256xi1>
    return %4, %8, %10 : tensor<256xui32>, tensor<256x1xi64>, tensor<256xi1>
  }
  func.func private @Unknown2(%arg0: tensor<128xi64>) -> (tensor<256xui32>, tensor<256x1xi64>, tensor<256xi1>) attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<-1.000000e+00> : tensor<256xf64>
    %1 = mhlo.constant dense<2> : tensor<256xi64>
    %2 = mhlo.constant dense<0> : tensor<256xi64>
    %3 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<128xi64>) -> tensor<2x128xi64>
    %4 = mhlo.reshape %3 : (tensor<2x128xi64>) -> tensor<256xi64>
    %5 = mhlo.convert %4 : (tensor<256xi64>) -> tensor<256xui32>
    %6 = mhlo.compare  LT, %4, %2 : (tensor<256xi64>, tensor<256xi64>) -> tensor<256xi1>
    %7 = mhlo.add %4, %1 : tensor<256xi64>
    %8 = mhlo.select %6, %7, %4 : tensor<256xi1>, tensor<256xi64>
    %9 = mhlo.reshape %8 : (tensor<256xi64>) -> tensor<256x1xi64>
    %10 = mhlo.convert %4 : (tensor<256xi64>) -> tensor<256xf64>
    %11 = mhlo.compare  NE, %10, %0 : (tensor<256xf64>, tensor<256xf64>) -> tensor<256xi1>
    return %5, %9, %11 : tensor<256xui32>, tensor<256x1xi64>, tensor<256xi1>
  }
  func.func private @Unknown3(%arg0: tensor<1x128xi64>) -> (tensor<128xui32>, tensor<128x1xi64>, tensor<128xi1>) attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<-1.000000e+00> : tensor<128xf64>
    %1 = mhlo.constant dense<512> : tensor<128xi64>
    %2 = mhlo.constant dense<0> : tensor<128xi64>
    %3 = mhlo.reshape %arg0 : (tensor<1x128xi64>) -> tensor<128xi64>
    %4 = mhlo.convert %3 : (tensor<128xi64>) -> tensor<128xui32>
    %5 = mhlo.compare  LT, %3, %2 : (tensor<128xi64>, tensor<128xi64>) -> tensor<128xi1>
    %6 = mhlo.add %3, %1 : tensor<128xi64>
    %7 = mhlo.select %5, %6, %3 : tensor<128xi1>, tensor<128xi64>
    %8 = mhlo.reshape %7 : (tensor<128xi64>) -> tensor<128x1xi64>
    %9 = mhlo.convert %3 : (tensor<128xi64>) -> tensor<128xf64>
    %10 = mhlo.compare  NE, %9, %0 : (tensor<128xf64>, tensor<128xf64>) -> tensor<128xi1>
    return %4, %8, %10 : tensor<128xui32>, tensor<128x1xi64>, tensor<128xi1>
  }
  func.func private @Unknown4(%arg0: tensor<256x128xf32>, %arg1: tensor<256x128xf32>, %arg2: tensor<128x128xf32>) -> tensor<2x128x128xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.reshape %arg0 : (tensor<256x128xf32>) -> tensor<2x128x128xf32>
    %1 = mhlo.reshape %arg1 : (tensor<256x128xf32>) -> tensor<2x128x128xf32>
    %2 = mhlo.add %0, %1 : tensor<2x128x128xf32>
    %3 = "mhlo.broadcast_in_dim"(%arg2) {broadcast_dimensions = dense<[1, 2]> : tensor<2xi64>} : (tensor<128x128xf32>) -> tensor<2x128x128xf32>
    %4 = mhlo.add %2, %3 : tensor<2x128x128xf32>
    return %4 : tensor<2x128x128xf32>
  }
  func.func private @Unknown5(%arg0: tensor<256x30522xf32>, %arg1: tensor<30522xf32>) -> (tensor<2x128x30522xf32>, tensor<256x30522xf32>) attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.reshape %arg0 : (tensor<256x30522xf32>) -> tensor<2x128x30522xf32>
    %1 = "mhlo.broadcast_in_dim"(%arg1) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<30522xf32>) -> tensor<2x128x30522xf32>
    %2 = mhlo.add %0, %1 : tensor<2x128x30522xf32>
    %3 = mhlo.reshape %2 : (tensor<2x128x30522xf32>) -> tensor<256x30522xf32>
    return %2, %3 : tensor<2x128x30522xf32>, tensor<256x30522xf32>
  }
  func.func private @Unknown6(%arg0: tensor<256xf32>, %arg1: tensor<256x30522xf32>) -> (tensor<256x30522xf32>, tensor<256x30522xf32>) attributes {__byteir_elementwise_fusion__} {
    %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<256xf32>) -> tensor<256x30522xf32>
    %1 = mhlo.subtract %arg1, %0 : tensor<256x30522xf32>
    %2 = mhlo.exponential %1 : tensor<256x30522xf32>
    return %1, %2 : tensor<256x30522xf32>, tensor<256x30522xf32>
  }
  func.func private @Unknown7(%arg0: tensor<256xf32>) -> tensor<256xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.log %arg0 : tensor<256xf32>
    return %0 : tensor<256xf32>
  }
  func.func private @Unknown8(%arg0: tensor<256xf32>, %arg1: tensor<256x30522xf32>, %arg2: tensor<256xi64>, %arg3: tensor<256xi1>) -> (tensor<256x30522xf32>, tensor<256x30522xf32>, tensor<256x30522xf32>, tensor<256x30522xf32>) attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<1.000000e+00> : tensor<256x30522xf32>
    %1 = mhlo.constant dense<0.000000e+00> : tensor<256x30522xf32>
    %2 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<256xf32>) -> tensor<256x30522xf32>
    %3 = mhlo.subtract %arg1, %2 : tensor<256x30522xf32>
    %4 = "mhlo.broadcast_in_dim"(%arg2) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<256xi64>) -> tensor<256x30522xi64>
    %5 = "mhlo.iota"() {iota_dimension = 0 : i64} : () -> tensor<30522xi64>
    %6 = "mhlo.broadcast_in_dim"(%5) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<30522xi64>) -> tensor<256x30522xi64>
    %7 = mhlo.compare  EQ, %4, %6 : (tensor<256x30522xi64>, tensor<256x30522xi64>) -> tensor<256x30522xi1>
    %8 = mhlo.select %7, %0, %1 : tensor<256x30522xi1>, tensor<256x30522xf32>
    %9 = mhlo.compare  NE, %8, %0 : (tensor<256x30522xf32>, tensor<256x30522xf32>) -> tensor<256x30522xi1>
    %10 = mhlo.negate %8 : tensor<256x30522xf32>
    %11 = mhlo.multiply %10, %3 : tensor<256x30522xf32>
    %12 = mhlo.select %9, %1, %11 : tensor<256x30522xi1>, tensor<256x30522xf32>
    %13 = "mhlo.broadcast_in_dim"(%arg3) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<256xi1>) -> tensor<256x30522xi1>
    %14 = mhlo.select %13, %0, %1 : tensor<256x30522xi1>, tensor<256x30522xf32>
    %15 = mhlo.multiply %14, %8 : tensor<256x30522xf32>
    %16 = mhlo.multiply %12, %15 : tensor<256x30522xf32>
    %17 = mhlo.multiply %10, %15 : tensor<256x30522xf32>
    %18 = mhlo.exponential %3 : tensor<256x30522xf32>
    return %15, %16, %17, %18 : tensor<256x30522xf32>, tensor<256x30522xf32>, tensor<256x30522xf32>, tensor<256x30522xf32>
  }
  func.func private @Unknown9(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.divide %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
  func.func private @Unknown10(%arg0: tensor<f32>) -> tensor<f32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %2 = mhlo.compare  NE, %arg0, %0 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %3 = mhlo.select %2, %arg0, %1 : tensor<i1>, tensor<f32>
    return %3 : tensor<f32>
  }
  func.func private @Unknown11(%arg0: tensor<f32>, %arg1: tensor<256x30522xf32>) -> tensor<256x30522xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<256x30522xf32>
    %1 = mhlo.divide %arg1, %0 : tensor<256x30522xf32>
    return %1 : tensor<256x30522xf32>
  }
  func.func private @Unknown12(%arg0: tensor<256xf32>, %arg1: tensor<256x30522xf32>, %arg2: tensor<256x30522xf32>) -> (tensor<256x30522xf32>, tensor<2x128x30522xf32>) attributes {__byteir_elementwise_fusion__} {
    %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<256xf32>) -> tensor<256x30522xf32>
    %1 = mhlo.multiply %arg1, %0 : tensor<256x30522xf32>
    %2 = mhlo.subtract %arg2, %1 : tensor<256x30522xf32>
    %3 = mhlo.reshape %2 : (tensor<256x30522xf32>) -> tensor<2x128x30522xf32>
    return %2, %3 : tensor<256x30522xf32>, tensor<2x128x30522xf32>
  }
  func.func private @MatmulOp13(%arg0: tensor<256x128xf32>, %arg1: tensor<256x30522xf32>) -> tensor<30522x128xf32> attributes {__byre__lhs_contracting_dimension = 0 : i64, __byre__output_transpose, __byre__rhs_contracting_dimension = 0 : i64, byre_compute_name = "MatmulOp"} {
    %0 = "mhlo.dot_general"(%arg0, %arg1) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [0], rhs_contracting_dimensions = [0]>, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<256x128xf32>, tensor<256x30522xf32>) -> tensor<128x30522xf32>
    %1 = "mhlo.transpose"(%0) {minor_to_major = dense<[0, 1]> : tensor<2xindex>, permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<128x30522xf32>) -> tensor<30522x128xf32>
    return %1 : tensor<30522x128xf32>
  }
  func.func private @Unknown14(%arg0: tensor<2x128x128xf32>, %arg1: tensor<2x128x128xf32>) -> tensor<2x128x128xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.add %arg0, %arg1 : tensor<2x128x128xf32>
    return %0 : tensor<2x128x128xf32>
  }
  func.func private @Unknown15(%arg0: tensor<2x128x128xf32>, %arg1: tensor<2x128x128xf32>, %arg2: tensor<2x128x128xf32>, %arg3: tensor<2x128x128xf32>) -> tensor<2x128x128xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.add %arg0, %arg1 : tensor<2x128x128xf32>
    %1 = mhlo.add %0, %arg2 : tensor<2x128x128xf32>
    %2 = mhlo.add %1, %arg3 : tensor<2x128x128xf32>
    return %2 : tensor<2x128x128xf32>
  }
  func.func private @Unknown16(%arg0: tensor<2x128x128xf32>, %arg1: tensor<2x128x128xf32>) -> tensor<2x128x128xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.add %arg0, %arg1 : tensor<2x128x128xf32>
    return %0 : tensor<2x128x128xf32>
  }
  func.func private @Unknown17(%arg0: tensor<2x128x128xf32>, %arg1: tensor<2x128x128xf32>, %arg2: tensor<2x128x128xf32>, %arg3: tensor<2x128x128xf32>) -> tensor<2x128x128xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.add %arg0, %arg1 : tensor<2x128x128xf32>
    %1 = mhlo.add %0, %arg2 : tensor<2x128x128xf32>
    %2 = mhlo.add %1, %arg3 : tensor<2x128x128xf32>
    return %2 : tensor<2x128x128xf32>
  }
  func.func private @Unknown18(%arg0: tensor<256xi1>, %arg1: tensor<2x128x128xf32>, %arg2: tensor<256xi1>) -> (tensor<256x128xf32>, tensor<256x128xf32>) attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<256x128xf32>
    %1 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<256xi1>) -> tensor<256x128xi1>
    %2 = mhlo.reshape %arg1 : (tensor<2x128x128xf32>) -> tensor<256x128xf32>
    %3 = mhlo.select %1, %2, %0 : tensor<256x128xi1>, tensor<256x128xf32>
    %4 = "mhlo.broadcast_in_dim"(%arg2) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<256xi1>) -> tensor<256x128xi1>
    %5 = mhlo.select %4, %2, %0 : tensor<256x128xi1>, tensor<256x128xf32>
    return %3, %5 : tensor<256x128xf32>, tensor<256x128xf32>
  }
  func.func private @Unknown19(%arg0: tensor<128xi1>, %arg1: tensor<128x128xf32>) -> tensor<128x128xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<128x128xf32>
    %1 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<128xi1>) -> tensor<128x128xi1>
    %2 = mhlo.select %1, %arg1, %0 : tensor<128x128xi1>, tensor<128x128xf32>
    return %2 : tensor<128x128xf32>
  }
  func.func @main(%arg0: tensor<2x128xi64>, %arg1: tensor<2x128xi64>, %arg2: tensor<1x512xi64>, %arg3: tensor<1x512xi64>, %arg4: tensor<30522x128xf32>, %arg5: tensor<2x128xf32>, %arg6: tensor<512x128xf32>, %arg7: tensor<128xf32>, %arg8: tensor<128xf32>, %arg9: tensor<128x128xf32>, %arg10: tensor<128xf32>, %arg11: tensor<128x128xf32>, %arg12: tensor<128xf32>, %arg13: tensor<128x128xf32>, %arg14: tensor<128xf32>, %arg15: tensor<128x128xf32>, %arg16: tensor<128xf32>, %arg17: tensor<128xf32>, %arg18: tensor<128xf32>, %arg19: tensor<512x128xf32>, %arg20: tensor<512xf32>, %arg21: tensor<128x512xf32>, %arg22: tensor<128xf32>, %arg23: tensor<128xf32>, %arg24: tensor<128xf32>, %arg25: tensor<128x128xf32>, %arg26: tensor<128xf32>, %arg27: tensor<128x128xf32>, %arg28: tensor<128xf32>, %arg29: tensor<128x128xf32>, %arg30: tensor<128xf32>, %arg31: tensor<128x128xf32>, %arg32: tensor<128xf32>, %arg33: tensor<128xf32>, %arg34: tensor<128xf32>, %arg35: tensor<512x128xf32>, %arg36: tensor<512xf32>, %arg37: tensor<128x512xf32>, %arg38: tensor<128xf32>, %arg39: tensor<128xf32>, %arg40: tensor<128xf32>, %arg41: tensor<128x128xf32>, %arg42: tensor<128xf32>, %arg43: tensor<128xf32>, %arg44: tensor<128xf32>, %arg45: tensor<30522xf32>) -> (tensor<2x128x30522xf32>, tensor<f32>, tensor<30522x128xf32>, tensor<2x128xf32>, tensor<512x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128xf32>, tensor<512xf32>, tensor<128x512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128xf32>, tensor<512xf32>, tensor<128x512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<30522xf32>) {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<512x128xf32>
    %1 = mhlo.constant dense<0.000000e+00> : tensor<2x128xf32>
    %2 = mhlo.constant dense<-0.000000e+00> : tensor<2x128x128xf32>
    %3 = mhlo.constant dense<0xFF800000> : tensor<f32>
    %4 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %5 = "mhlo.slice"(%arg2) {limit_indices = dense<[1, 128]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<1x512xi64>) -> tensor<1x128xi64>
    %6 = mhlo.reshape %5 : (tensor<1x128xi64>) -> tensor<128xi64>
    %7 = "mhlo.slice"(%arg3) {limit_indices = dense<[1, 128]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<1x512xi64>) -> tensor<1x128xi64>
    %8:2 = call @Unknown0(%arg1) : (tensor<2x128xi64>) -> (tensor<256xi64>, tensor<256xi1>)
    %9:3 = call @Unknown1(%arg0) : (tensor<2x128xi64>) -> (tensor<256xui32>, tensor<256x1xi64>, tensor<256xi1>)
    %10 = "mhlo.gather"(%arg4, %9#0) {dimension_numbers = #mhlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 128]> : tensor<2xi64>} : (tensor<30522x128xf32>, tensor<256xui32>) -> tensor<256x128xf32>
    %11:3 = call @Unknown2(%6) : (tensor<128xi64>) -> (tensor<256xui32>, tensor<256x1xi64>, tensor<256xi1>)
    %12 = "mhlo.gather"(%arg5, %11#0) {dimension_numbers = #mhlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 128]> : tensor<2xi64>} : (tensor<2x128xf32>, tensor<256xui32>) -> tensor<256x128xf32>
    %13:3 = call @Unknown3(%7) : (tensor<1x128xi64>) -> (tensor<128xui32>, tensor<128x1xi64>, tensor<128xi1>)
    %14 = "mhlo.gather"(%arg6, %13#0) {dimension_numbers = #mhlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 128]> : tensor<2xi64>} : (tensor<512x128xf32>, tensor<128xui32>) -> tensor<128x128xf32>
    %15 = call @Unknown4(%10, %12, %14) : (tensor<256x128xf32>, tensor<256x128xf32>, tensor<128x128xf32>) -> tensor<2x128x128xf32>
    %16:3 = mhlo.custom_call @ftv4.layernorm(%15, %arg7, %arg8) {backend_config = ""} : (tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>) -> (tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>)
    %17 = mhlo.custom_call @ftv4.linear_transpose(%16#0, %arg9, %arg10) {backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}"} : (tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>) -> tensor<2x2x128x64xf32>
    %18 = mhlo.custom_call @ftv4.linear_transpose(%16#0, %arg11, %arg12) {backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}"} : (tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>) -> tensor<2x2x128x64xf32>
    %19 = mhlo.custom_call @ftv4.matmul(%17, %18) {backend_config = "{scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true}"} : (tensor<2x2x128x64xf32>, tensor<2x2x128x64xf32>) -> tensor<2x2x128x128xf32>
    %20:3 = mhlo.custom_call @ftv4.softmax(%19, %2) {backend_config = "{batch_first = true, dropout_rate = 0.000000e+00 : f32, head_num = 2 : i32}"} : (tensor<2x2x128x128xf32>, tensor<2x128x128xf32>) -> (tensor<2x2x128x128xf32>, tensor<2x2x128x128xf32>, tensor<2x2x128x128xui8>)
    %21 = mhlo.custom_call @ftv4.linear_transpose(%16#0, %arg13, %arg14) {backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}"} : (tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>) -> tensor<2x2x128x64xf32>
    %22 = mhlo.custom_call @ftv4.matmul(%20#0, %21) {backend_config = "{scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false}"} : (tensor<2x2x128x128xf32>, tensor<2x2x128x64xf32>) -> tensor<2x2x128x64xf32>
    %23 = mhlo.custom_call @ftv4.transpose4d(%22) {backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22}"} : (tensor<2x2x128x64xf32>) -> tensor<2x128x2x64xf32>
    %24 = mhlo.reshape %23 : (tensor<2x128x2x64xf32>) -> tensor<2x128x128xf32>
    %25 = mhlo.custom_call @ftv4.linear(%24, %arg15, %arg16) {backend_config = ""} : (tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>) -> tensor<2x128x128xf32>
    %26:4 = mhlo.custom_call @ftv4.layernorm_residual(%25, %arg17, %arg18, %16#0) {backend_config = ""} : (tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>) -> (tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>)
    %27:3 = mhlo.custom_call @ftv4.linear_gelu_dropout(%26#0, %arg19, %arg20) {backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}"} : (tensor<2x128x128xf32>, tensor<512x128xf32>, tensor<512xf32>) -> (tensor<2x128x512xf32>, tensor<2x128x512xf32>, tensor<0xf32>)
    %28 = mhlo.custom_call @ftv4.linear(%27#0, %arg21, %arg22) {backend_config = ""} : (tensor<2x128x512xf32>, tensor<128x512xf32>, tensor<128xf32>) -> tensor<2x128x128xf32>
    %29:4 = mhlo.custom_call @ftv4.layernorm_residual(%28, %arg23, %arg24, %26#0) {backend_config = ""} : (tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>) -> (tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>)
    %30 = mhlo.custom_call @ftv4.linear_transpose(%29#0, %arg25, %arg26) {backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}"} : (tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>) -> tensor<2x2x128x64xf32>
    %31 = mhlo.custom_call @ftv4.linear_transpose(%29#0, %arg27, %arg28) {backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}"} : (tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>) -> tensor<2x2x128x64xf32>
    %32 = mhlo.custom_call @ftv4.matmul(%30, %31) {backend_config = "{scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true}"} : (tensor<2x2x128x64xf32>, tensor<2x2x128x64xf32>) -> tensor<2x2x128x128xf32>
    %33:3 = mhlo.custom_call @ftv4.softmax(%32, %2) {backend_config = "{batch_first = true, dropout_rate = 0.000000e+00 : f32, head_num = 2 : i32}"} : (tensor<2x2x128x128xf32>, tensor<2x128x128xf32>) -> (tensor<2x2x128x128xf32>, tensor<2x2x128x128xf32>, tensor<2x2x128x128xui8>)
    %34 = mhlo.custom_call @ftv4.linear_transpose(%29#0, %arg29, %arg30) {backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}"} : (tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>) -> tensor<2x2x128x64xf32>
    %35 = mhlo.custom_call @ftv4.matmul(%33#0, %34) {backend_config = "{scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false}"} : (tensor<2x2x128x128xf32>, tensor<2x2x128x64xf32>) -> tensor<2x2x128x64xf32>
    %36 = mhlo.custom_call @ftv4.transpose4d(%35) {backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22}"} : (tensor<2x2x128x64xf32>) -> tensor<2x128x2x64xf32>
    %37 = mhlo.reshape %36 : (tensor<2x128x2x64xf32>) -> tensor<2x128x128xf32>
    %38 = mhlo.custom_call @ftv4.linear(%37, %arg31, %arg32) {backend_config = ""} : (tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>) -> tensor<2x128x128xf32>
    %39:4 = mhlo.custom_call @ftv4.layernorm_residual(%38, %arg33, %arg34, %29#0) {backend_config = ""} : (tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>) -> (tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>)
    %40:3 = mhlo.custom_call @ftv4.linear_gelu_dropout(%39#0, %arg35, %arg36) {backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}"} : (tensor<2x128x128xf32>, tensor<512x128xf32>, tensor<512xf32>) -> (tensor<2x128x512xf32>, tensor<2x128x512xf32>, tensor<0xf32>)
    %41 = mhlo.custom_call @ftv4.linear(%40#0, %arg37, %arg38) {backend_config = ""} : (tensor<2x128x512xf32>, tensor<128x512xf32>, tensor<128xf32>) -> tensor<2x128x128xf32>
    %42:4 = mhlo.custom_call @ftv4.layernorm_residual(%41, %arg39, %arg40, %39#0) {backend_config = ""} : (tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>) -> (tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<2x128x128xf32>)
    %43:3 = mhlo.custom_call @ftv4.linear_gelu_dropout(%42#0, %arg41, %arg42) {backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}"} : (tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>) -> (tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<0xf32>)
    %44:3 = mhlo.custom_call @ftv4.layernorm(%43#0, %arg43, %arg44) {backend_config = ""} : (tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>) -> (tensor<2x128x128xf32>, tensor<256xf32>, tensor<256xf32>)
    %45 = mhlo.reshape %44#0 : (tensor<2x128x128xf32>) -> tensor<256x128xf32>
    %46 = "mhlo.dot_general"(%45, %arg4) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [1]>, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<256x128xf32>, tensor<30522x128xf32>) -> tensor<256x30522xf32>
    %47:2 = call @Unknown5(%46, %arg45) : (tensor<256x30522xf32>, tensor<30522xf32>) -> (tensor<2x128x30522xf32>, tensor<256x30522xf32>)
    %48 = mhlo.reduce(%47#1 init: %3) across dimensions = [1] : (tensor<256x30522xf32>, tensor<f32>) -> tensor<256xf32>
     reducer(%arg46: tensor<f32>, %arg47: tensor<f32>)  {
      %103 = mhlo.maximum %arg46, %arg47 : tensor<f32>
      mhlo.return %103 : tensor<f32>
    }
    %49:2 = call @Unknown6(%48, %47#1) : (tensor<256xf32>, tensor<256x30522xf32>) -> (tensor<256x30522xf32>, tensor<256x30522xf32>)
    %50 = mhlo.reduce(%49#1 init: %4) across dimensions = [1] : (tensor<256x30522xf32>, tensor<f32>) -> tensor<256xf32>
     reducer(%arg46: tensor<f32>, %arg47: tensor<f32>)  {
      %103 = mhlo.add %arg46, %arg47 : tensor<f32>
      mhlo.return %103 : tensor<f32>
    }
    %51 = call @Unknown7(%50) : (tensor<256xf32>) -> tensor<256xf32>
    %52:4 = call @Unknown8(%51, %49#0, %8#0, %8#1) : (tensor<256xf32>, tensor<256x30522xf32>, tensor<256xi64>, tensor<256xi1>) -> (tensor<256x30522xf32>, tensor<256x30522xf32>, tensor<256x30522xf32>, tensor<256x30522xf32>)
    %53 = mhlo.reduce(%52#1 init: %4) across dimensions = [0, 1] : (tensor<256x30522xf32>, tensor<f32>) -> tensor<f32>
     reducer(%arg46: tensor<f32>, %arg47: tensor<f32>)  {
      %103 = mhlo.add %arg46, %arg47 : tensor<f32>
      mhlo.return %103 : tensor<f32>
    }
    %54 = mhlo.reduce(%52#0 init: %4) across dimensions = [0, 1] : (tensor<256x30522xf32>, tensor<f32>) -> tensor<f32>
     reducer(%arg46: tensor<f32>, %arg47: tensor<f32>)  {
      %103 = mhlo.add %arg46, %arg47 : tensor<f32>
      mhlo.return %103 : tensor<f32>
    }
    %55 = call @Unknown9(%53, %54) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %56 = call @Unknown10(%54) : (tensor<f32>) -> tensor<f32>
    %57 = call @Unknown11(%56, %52#2) : (tensor<f32>, tensor<256x30522xf32>) -> tensor<256x30522xf32>
    %58 = mhlo.reduce(%57 init: %4) across dimensions = [1] : (tensor<256x30522xf32>, tensor<f32>) -> tensor<256xf32>
     reducer(%arg46: tensor<f32>, %arg47: tensor<f32>)  {
      %103 = mhlo.add %arg46, %arg47 : tensor<f32>
      mhlo.return %103 : tensor<f32>
    }
    %59:2 = call @Unknown12(%58, %52#3, %57) : (tensor<256xf32>, tensor<256x30522xf32>, tensor<256x30522xf32>) -> (tensor<256x30522xf32>, tensor<2x128x30522xf32>)
    %60 = call @MatmulOp13(%45, %59#0) : (tensor<256x128xf32>, tensor<256x30522xf32>) -> tensor<30522x128xf32>
    %61 = "mhlo.dot"(%59#0, %arg4) {precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<256x30522xf32>, tensor<30522x128xf32>) -> tensor<256x128xf32>
    %62 = mhlo.reshape %61 : (tensor<256x128xf32>) -> tensor<2x128x128xf32>
    %63:3 = mhlo.custom_call @ftv4.layernorm_backward(%62, %43#0, %arg43, %44#1, %44#2) {backend_config = ""} : (tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>) -> (tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>)
    %64:3 = mhlo.custom_call @ftv4.linear_gelu_dropout_backward(%63#0, %42#0, %arg41, %43#1, %43#2) {backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}"} : (tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<2x128x128xf32>, tensor<0xf32>) -> (tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>)
    %65:4 = mhlo.custom_call @ftv4.layernorm_backward_residual(%64#0, %42#3, %arg39, %42#1, %42#2) {backend_config = ""} : (tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>) -> (tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>)
    %66:3 = mhlo.custom_call @ftv4.linear_backward(%65#0, %40#0, %arg37) {backend_config = ""} : (tensor<2x128x128xf32>, tensor<2x128x512xf32>, tensor<128x512xf32>) -> (tensor<2x128x512xf32>, tensor<128x512xf32>, tensor<128xf32>)
    %67:3 = mhlo.custom_call @ftv4.linear_gelu_dropout_backward(%66#0, %39#0, %arg35, %40#1, %40#2) {backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}"} : (tensor<2x128x512xf32>, tensor<2x128x128xf32>, tensor<512x128xf32>, tensor<2x128x512xf32>, tensor<0xf32>) -> (tensor<2x128x128xf32>, tensor<512x128xf32>, tensor<512xf32>)
    %68 = call @Unknown14(%65#3, %67#0) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %69:4 = mhlo.custom_call @ftv4.layernorm_backward_residual(%68, %39#3, %arg33, %39#1, %39#2) {backend_config = ""} : (tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>) -> (tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>)
    %70:3 = mhlo.custom_call @ftv4.linear_backward(%69#0, %37, %arg31) {backend_config = ""} : (tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<128x128xf32>) -> (tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>)
    %71 = mhlo.reshape %70#0 : (tensor<2x128x128xf32>) -> tensor<2x128x2x64xf32>
    %72 = mhlo.custom_call @ftv4.transpose4d_backward(%71) {backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22}"} : (tensor<2x128x2x64xf32>) -> tensor<2x2x128x64xf32>
    %73:2 = mhlo.custom_call @ftv4.matmul_backward(%72, %33#0, %34) {backend_config = "{scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false}"} : (tensor<2x2x128x64xf32>, tensor<2x2x128x128xf32>, tensor<2x2x128x64xf32>) -> (tensor<2x2x128x128xf32>, tensor<2x2x128x64xf32>)
    %74 = mhlo.custom_call @ftv4.softmax_backward(%73#0, %33#0, %33#2) {backend_config = "{dropout_rate = 0.000000e+00 : f32}"} : (tensor<2x2x128x128xf32>, tensor<2x2x128x128xf32>, tensor<2x2x128x128xui8>) -> tensor<2x2x128x128xf32>
    %75:2 = mhlo.custom_call @ftv4.matmul_backward(%74, %30, %31) {backend_config = "{scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true}"} : (tensor<2x2x128x128xf32>, tensor<2x2x128x64xf32>, tensor<2x2x128x64xf32>) -> (tensor<2x2x128x64xf32>, tensor<2x2x128x64xf32>)
    %76:3 = mhlo.custom_call @ftv4.linear_transpose_backward(%75#0, %29#0, %arg25) {backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}"} : (tensor<2x2x128x64xf32>, tensor<2x128x128xf32>, tensor<128x128xf32>) -> (tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>)
    %77:3 = mhlo.custom_call @ftv4.linear_transpose_backward(%73#1, %29#0, %arg29) {backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}"} : (tensor<2x2x128x64xf32>, tensor<2x128x128xf32>, tensor<128x128xf32>) -> (tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>)
    %78:3 = mhlo.custom_call @ftv4.linear_transpose_backward(%75#1, %29#0, %arg27) {backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}"} : (tensor<2x2x128x64xf32>, tensor<2x128x128xf32>, tensor<128x128xf32>) -> (tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>)
    %79 = call @Unknown15(%69#3, %76#0, %77#0, %78#0) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %80:4 = mhlo.custom_call @ftv4.layernorm_backward_residual(%79, %29#3, %arg23, %29#1, %29#2) {backend_config = ""} : (tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>) -> (tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>)
    %81:3 = mhlo.custom_call @ftv4.linear_backward(%80#0, %27#0, %arg21) {backend_config = ""} : (tensor<2x128x128xf32>, tensor<2x128x512xf32>, tensor<128x512xf32>) -> (tensor<2x128x512xf32>, tensor<128x512xf32>, tensor<128xf32>)
    %82:3 = mhlo.custom_call @ftv4.linear_gelu_dropout_backward(%81#0, %26#0, %arg19, %27#1, %27#2) {backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}"} : (tensor<2x128x512xf32>, tensor<2x128x128xf32>, tensor<512x128xf32>, tensor<2x128x512xf32>, tensor<0xf32>) -> (tensor<2x128x128xf32>, tensor<512x128xf32>, tensor<512xf32>)
    %83 = call @Unknown16(%80#3, %82#0) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %84:4 = mhlo.custom_call @ftv4.layernorm_backward_residual(%83, %26#3, %arg17, %26#1, %26#2) {backend_config = ""} : (tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>) -> (tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<2x128x128xf32>)
    %85:3 = mhlo.custom_call @ftv4.linear_backward(%84#0, %24, %arg15) {backend_config = ""} : (tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<128x128xf32>) -> (tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>)
    %86 = mhlo.reshape %85#0 : (tensor<2x128x128xf32>) -> tensor<2x128x2x64xf32>
    %87 = mhlo.custom_call @ftv4.transpose4d_backward(%86) {backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22}"} : (tensor<2x128x2x64xf32>) -> tensor<2x2x128x64xf32>
    %88:2 = mhlo.custom_call @ftv4.matmul_backward(%87, %20#0, %21) {backend_config = "{scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false}"} : (tensor<2x2x128x64xf32>, tensor<2x2x128x128xf32>, tensor<2x2x128x64xf32>) -> (tensor<2x2x128x128xf32>, tensor<2x2x128x64xf32>)
    %89 = mhlo.custom_call @ftv4.softmax_backward(%88#0, %20#0, %20#2) {backend_config = "{dropout_rate = 0.000000e+00 : f32}"} : (tensor<2x2x128x128xf32>, tensor<2x2x128x128xf32>, tensor<2x2x128x128xui8>) -> tensor<2x2x128x128xf32>
    %90:2 = mhlo.custom_call @ftv4.matmul_backward(%89, %17, %18) {backend_config = "{scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true}"} : (tensor<2x2x128x128xf32>, tensor<2x2x128x64xf32>, tensor<2x2x128x64xf32>) -> (tensor<2x2x128x64xf32>, tensor<2x2x128x64xf32>)
    %91:3 = mhlo.custom_call @ftv4.linear_transpose_backward(%90#0, %16#0, %arg9) {backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}"} : (tensor<2x2x128x64xf32>, tensor<2x128x128xf32>, tensor<128x128xf32>) -> (tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>)
    %92:3 = mhlo.custom_call @ftv4.linear_transpose_backward(%88#1, %16#0, %arg13) {backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}"} : (tensor<2x2x128x64xf32>, tensor<2x128x128xf32>, tensor<128x128xf32>) -> (tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>)
    %93:3 = mhlo.custom_call @ftv4.linear_transpose_backward(%90#1, %16#0, %arg11) {backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}"} : (tensor<2x2x128x64xf32>, tensor<2x128x128xf32>, tensor<128x128xf32>) -> (tensor<2x128x128xf32>, tensor<128x128xf32>, tensor<128xf32>)
    %94 = call @Unknown17(%84#3, %91#0, %92#0, %93#0) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %95:3 = mhlo.custom_call @ftv4.layernorm_backward(%94, %15, %arg7, %16#1, %16#2) {backend_config = ""} : (tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>) -> (tensor<2x128x128xf32>, tensor<128xf32>, tensor<128xf32>)
    %96:2 = call @Unknown18(%9#2, %95#0, %11#2) : (tensor<256xi1>, tensor<2x128x128xf32>, tensor<256xi1>) -> (tensor<256x128xf32>, tensor<256x128xf32>)
    %97 = "mhlo.scatter"(%60, %9#1, %96#0) ({
    ^bb0(%arg46: tensor<f32>, %arg47: tensor<f32>):
      %103 = mhlo.add %arg46, %arg47 : tensor<f32>
      mhlo.return %103 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<30522x128xf32>, tensor<256x1xi64>, tensor<256x128xf32>) -> tensor<30522x128xf32>
    %98 = "mhlo.scatter"(%1, %11#1, %96#1) ({
    ^bb0(%arg46: tensor<f32>, %arg47: tensor<f32>):
      %103 = mhlo.add %arg46, %arg47 : tensor<f32>
      mhlo.return %103 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<2x128xf32>, tensor<256x1xi64>, tensor<256x128xf32>) -> tensor<2x128xf32>
    %99 = mhlo.reduce(%95#0 init: %4) across dimensions = [0] : (tensor<2x128x128xf32>, tensor<f32>) -> tensor<128x128xf32>
     reducer(%arg46: tensor<f32>, %arg47: tensor<f32>)  {
      %103 = mhlo.add %arg46, %arg47 : tensor<f32>
      mhlo.return %103 : tensor<f32>
    }
    %100 = call @Unknown19(%13#2, %99) : (tensor<128xi1>, tensor<128x128xf32>) -> tensor<128x128xf32>
    %101 = "mhlo.scatter"(%0, %13#1, %100) ({
    ^bb0(%arg46: tensor<f32>, %arg47: tensor<f32>):
      %103 = mhlo.add %arg46, %arg47 : tensor<f32>
      mhlo.return %103 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<512x128xf32>, tensor<128x1xi64>, tensor<128x128xf32>) -> tensor<512x128xf32>
    %102 = mhlo.reduce(%59#1 init: %4) across dimensions = [0, 1] : (tensor<2x128x30522xf32>, tensor<f32>) -> tensor<30522xf32>
     reducer(%arg46: tensor<f32>, %arg47: tensor<f32>)  {
      %103 = mhlo.add %arg46, %arg47 : tensor<f32>
      mhlo.return %103 : tensor<f32>
    }
    return %47#0, %55, %97, %98, %101, %95#1, %95#2, %91#1, %91#2, %93#1, %93#2, %92#1, %92#2, %85#1, %85#2, %84#1, %84#2, %82#1, %82#2, %81#1, %81#2, %80#1, %80#2, %76#1, %76#2, %78#1, %78#2, %77#1, %77#2, %70#1, %70#2, %69#1, %69#2, %67#1, %67#2, %66#1, %66#2, %65#1, %65#2, %64#1, %64#2, %63#1, %63#2, %102 : tensor<2x128x30522xf32>, tensor<f32>, tensor<30522x128xf32>, tensor<2x128xf32>, tensor<512x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128xf32>, tensor<512xf32>, tensor<128x512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128xf32>, tensor<512xf32>, tensor<128x512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<30522xf32>
  }
}

