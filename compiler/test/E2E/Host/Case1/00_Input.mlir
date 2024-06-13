// RUN: byteir-opt %s --hlo-graph-opt --hlo-opt="target=CPU" --linalg-tensor-opt="target=CPU" --byre-tensor-opt="entry-func=main append-arg-types" --byteir-bufferize-opt --scf-opt="target=CPU" | FileCheck %s

// CHECK-LABEL: func.func @main

module {
  func.func @main(%arg0: tensor<1x100x27x48x3xf32>) -> tensor<51200xi32> {
    %0 = mhlo.constant dense<1> : tensor<100x1296xi32>
    %1 = mhlo.constant dense<[0, 512, 1024, 1536, 2048, 2560, 3072, 3584, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192, 8704, 9216, 9728, 10240, 10752, 11264, 11776, 12288, 12800, 13312, 13824, 14336, 14848, 15360, 15872, 16384, 16896, 17408, 17920, 18432, 18944, 19456, 19968, 20480, 20992, 21504, 22016, 22528, 23040, 23552, 24064, 24576, 25088, 25600, 26112, 26624, 27136, 27648, 28160, 28672, 29184, 29696, 30208, 30720, 31232, 31744, 32256, 32768, 33280, 33792, 34304, 34816, 35328, 35840, 36352, 36864, 37376, 37888, 38400, 38912, 39424, 39936, 40448, 40960, 41472, 41984, 42496, 43008, 43520, 44032, 44544, 45056, 45568, 46080, 46592, 47104, 47616, 48128, 48640, 49152, 49664, 50176, 50688]> : tensor<100xi32>
    %2 = mhlo.constant dense<6> : tensor<100x1296xi32>
    %3 = mhlo.constant dense<3> : tensor<100x1296xi32>
    %4 = mhlo.constant dense<5> : tensor<100x1296xi32>
    %5 = mhlo.constant dense<0> : tensor<51200xi32>
    %6 = mhlo.convert %arg0 : (tensor<1x100x27x48x3xf32>) -> tensor<1x100x27x48x3xi32>
    %7 = mhlo.reshape %6 : (tensor<1x100x27x48x3xi32>) -> tensor<100x1296x3xi32>
    %8 = "mhlo.slice"(%7) {limit_indices = dense<[100, 1296, 1]> : tensor<3xi64>, start_indices = dense<0> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>} : (tensor<100x1296x3xi32>) -> tensor<100x1296x1xi32>
    %9 = mhlo.reshape %8 : (tensor<100x1296x1xi32>) -> tensor<100x1296xi32>
    %10 = mhlo.shift_right_arithmetic %9, %4 : tensor<100x1296xi32>
    %11 = mhlo.shift_left %10, %2 : tensor<100x1296xi32>
    %12 = "mhlo.slice"(%7) {limit_indices = dense<[100, 1296, 2]> : tensor<3xi64>, start_indices = dense<[0, 0, 1]> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>} : (tensor<100x1296x3xi32>) -> tensor<100x1296x1xi32>
    %13 = mhlo.reshape %12 : (tensor<100x1296x1xi32>) -> tensor<100x1296xi32>
    %14 = mhlo.shift_right_arithmetic %13, %4 : tensor<100x1296xi32>
    %15 = mhlo.shift_left %14, %3 : tensor<100x1296xi32>
    %16 = mhlo.add %11, %15 : tensor<100x1296xi32>
    %17 = "mhlo.slice"(%7) {limit_indices = dense<[100, 1296, 3]> : tensor<3xi64>, start_indices = dense<[0, 0, 2]> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>} : (tensor<100x1296x3xi32>) -> tensor<100x1296x1xi32>
    %18 = mhlo.reshape %17 : (tensor<100x1296x1xi32>) -> tensor<100x1296xi32>
    %19 = mhlo.shift_right_arithmetic %18, %4 : tensor<100x1296xi32>
    %20 = mhlo.add %19, %16 : tensor<100x1296xi32>
    %21 = "mhlo.broadcast_in_dim"(%1) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<100xi32>) -> tensor<100x1296xi32>
    %22 = mhlo.add %20, %21 : tensor<100x1296xi32>
    %23 = "mhlo.scatter"(%5, %22, %0) ({
    ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):
      %28 = mhlo.add %arg1, %arg2 : tensor<i32>
      mhlo.return %28 : tensor<i32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 2>, unique_indices = false} : (tensor<51200xi32>, tensor<100x1296xi32>, tensor<100x1296xi32>) -> tensor<51200xi32>
    %24 = mhlo.convert %23 : tensor<51200xi32>
    return %24 : tensor<51200xi32>
  }
}