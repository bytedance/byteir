// RUN: byteir-opt %s -byteir-bufferize-opt --split-input-file | FileCheck %s

//CHECK-LABEL: func.func @fuse_matmul_matmul
//CHECK-SAME:  (%[[ARG0:.+]]: {{.+}}, %[[ARG1:.+]]: {{.+}}, %[[ARG2:.+]]: {{.+}})
func.func @fuse_matmul_matmul(%arg0: tensor<1024x32xf32>, %arg1: tensor<32x512xf32>, %arg2: tensor<512x32xf32>) -> tensor<1024x32xf32> {
  %c512 = arith.constant 512 : index
  %c1024 = arith.constant 1024 : index
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  //CHECK-DAG: %[[A:.+]] = memref.alloc() : memref<1024x32xf32>
  %0 = tensor.empty() : tensor<1024x32xf32>
  //CHECK: scf.for %[[ARG3:.+]] =   
  %1 = scf.for %arg3 = %c0 to %c1024 step %c4 iter_args(%arg4 = %0) -> (tensor<1024x32xf32>) {
    //CHECK: scf.for %[[ARG4:.+]] =   
    %2 = scf.for %arg5 = %c0 to %c512 step %c8 iter_args(%arg6 = %arg4) -> (tensor<1024x32xf32>) {
      //CHECK-DAG: %[[S:.+]] = memref.subview %[[ARG0]][%[[ARG3]], 0] [4, 32]
      //CHECK-DAG: %[[S0:.+]] = memref.subview %[[ARG1]][0, %[[ARG4]]] [32, 8]
      //CHECK-DAG: %[[A1:.+]] = memref.alloc() : memref<4x8xf32>
      //CHECK-DAG: %[[S3:.+]] = memref.subview %[[A]][%[[ARG3]], 0] [4, 32]
      %extracted_slice = tensor.extract_slice %arg0[%arg3, 0] [4, 32] [1, 1] : tensor<1024x32xf32> to tensor<4x32xf32>
      %extracted_slice_0 = tensor.extract_slice %arg1[0, %arg5] [32, 8] [1, 1] : tensor<32x512xf32> to tensor<32x8xf32>
      %3 = tensor.empty() : tensor<4x8xf32>
      //CHECK-DAG: linalg.matmul ins(%[[S]], %[[S0]] : {{.+}}) outs(%[[A1]] : {{.+}})
      %4 = linalg.matmul ins(%extracted_slice, %extracted_slice_0 : tensor<4x32xf32>, tensor<32x8xf32>) outs(%3 : tensor<4x8xf32>) -> tensor<4x8xf32>
      %extracted_slice_1 = tensor.extract_slice %arg2[%arg5, 0] [8, 32] [1, 1] : tensor<512x32xf32> to tensor<8x32xf32>
      //CHECK-DAG: %[[S2:.+]] = memref.subview %[[ARG2]][%[[ARG4]], 0] [8, 32]
      %extracted_slice_2 = tensor.extract_slice %arg6[%arg3, 0] [4, 32] [1, 1] : tensor<1024x32xf32> to tensor<4x32xf32>
      //CHECK: linalg.matmul
      //CHECK-SAME: ins(%[[A1]], %[[S2]] : {{.+}}) outs(%[[S3]] : {{.+}})
      %5 = linalg.matmul ins(%4, %extracted_slice_1 : tensor<4x8xf32>, tensor<8x32xf32>) outs(%extracted_slice_2 : tensor<4x32xf32>) -> tensor<4x32xf32>
      %inserted_slice = tensor.insert_slice %5 into %arg6[%arg3, 0] [4, 32] [1, 1] : tensor<4x32xf32> into tensor<1024x32xf32>
      scf.yield %inserted_slice : tensor<1024x32xf32>
    }
    scf.yield %2 : tensor<1024x32xf32>
  } {__byteir_parallel__}
  return %1 : tensor<1024x32xf32>
}

// -----

//CHECK-LABEL: func.func @fuse_fork_add
//CHECK-SAME:  (%[[ARG0:.+]]: {{.+}}, %[[ARG1:.+]]: {{.+}}, %[[ARG2:.+]]: {{.+}})
func.func @fuse_fork_add(%arg0: tensor<1024x512xf32>, %arg1: tensor<1024x512xf32>, %arg2: tensor<1024x512xf32>) -> tensor<1024x512xf32> {
  %c1024 = arith.constant 1024 : index
  %c512 = arith.constant 512 : index
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  //CHECK-DAG: %[[A:.+]] = memref.alloc() : memref<1024x512xf32>
  %0 = tensor.empty() : tensor<1024x512xf32>
  //CHECK: scf.for %[[ARG3:.+]] =   
  %1 = scf.for %arg3 = %c0 to %c512 step %c8 iter_args(%arg4 = %0) -> (tensor<1024x512xf32>) {
    //CHECK: scf.for %[[ARG4:.+]] =   
    %2 = scf.for %arg5 = %c0 to %c1024 step %c4 iter_args(%arg6 = %arg4) -> (tensor<1024x512xf32>) {
      //CHECK-DAG: %[[S:.+]] = memref.subview %[[ARG0]][%[[ARG4]], %[[ARG3]]] [4, 8]
      //CHECK-DAG: %[[S0:.+]] = memref.subview %[[A]][%[[ARG4]], %[[ARG3]]] [4, 8]
      //CHECK-DAG: %[[S1:.+]] = memref.subview %[[ARG1]][%[[ARG4]], %[[ARG3]]] [4, 8]
      %extracted_slice = tensor.extract_slice %arg0[%arg5, %arg3] [4, 8] [1, 1] : tensor<1024x512xf32> to tensor<4x8xf32>
      %extracted_slice_0 = tensor.extract_slice %arg1[%arg5, %arg3] [4, 8] [1, 1] : tensor<1024x512xf32> to tensor<4x8xf32>
      //CHECK-DAG: %[[A3:.+]] = memref.alloc() : memref<4x8xf32>
      %3 = tensor.empty() : tensor<4x8xf32>
      //CHECK: linalg.elemwise_binary
      //CHECK-SAME: ins(%[[S]], %[[S1]] : {{.+}}) outs(%[[A3]] : {{.+}})
      %4 = linalg.elemwise_binary ins(%extracted_slice, %extracted_slice_0 : tensor<4x8xf32>, tensor<4x8xf32>) outs(%3 : tensor<4x8xf32>) -> tensor<4x8xf32>
      %extracted_slice_1 = tensor.extract_slice %arg1[%arg5, %arg3] [4, 8] [1, 1] : tensor<1024x512xf32> to tensor<4x8xf32>
      //CHECK-DAG: %[[S2:.+]] = memref.subview %[[ARG2]][%[[ARG4]], %[[ARG3]]] [4, 8]
      %extracted_slice_2 = tensor.extract_slice %arg2[%arg5, %arg3] [4, 8] [1, 1] : tensor<1024x512xf32> to tensor<4x8xf32>
      //CHECK-DAG: %[[A4:.+]] = memref.alloc() : memref<4x8xf32>
      %5 = tensor.empty() : tensor<4x8xf32>
      //CHECK: linalg.elemwise_binary
      //CHECK-SAME: ins(%[[S1]], %[[S2]] : {{.+}}) outs(%[[A4]] : {{.+}})
      //CHECK: linalg.elemwise_binary
      //CHECK-SAME: ins(%[[A3]], %[[A4]] : {{.+}}) outs(%[[S0]] : {{.+}})
      %6 = linalg.elemwise_binary ins(%extracted_slice_1, %extracted_slice_2 : tensor<4x8xf32>, tensor<4x8xf32>) outs(%5 : tensor<4x8xf32>) -> tensor<4x8xf32>
      %7 = tensor.empty() : tensor<4x8xf32>
      %8 = linalg.elemwise_binary ins(%4, %6 : tensor<4x8xf32>, tensor<4x8xf32>) outs(%7 : tensor<4x8xf32>) -> tensor<4x8xf32>
      %inserted_slice = tensor.insert_slice %8 into %arg6[%arg5, %arg3] [4, 8] [1, 1] : tensor<4x8xf32> into tensor<1024x512xf32>
      scf.yield %inserted_slice : tensor<1024x512xf32>
    } {__byteir_parallel__}
    scf.yield %2 : tensor<1024x512xf32>
  } {__byteir_parallel__}
  return %1 : tensor<1024x512xf32>
}

// -----

//CHECK-LABEL: func.func @max_pool
//CHECK-SAME:  (%[[ARG0:.+]]: {{.+}})
func.func @max_pool(%arg0: tensor<4x126x126x16xf32>) -> tensor<4x63x63x16xf32> {
  %c4 = arith.constant 4 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0xFF800000 : f32
  //CHECK-DAG: %[[CST:.+]] = arith.constant 0xFF800000 : f32
  //CHECK-DAG: %[[A:.+]] = memref.alloc() : memref<2x2xf32>
  //CHECK-DAG: %[[A0:.+]] = memref.alloc() : memref<4x63x63x16xf32>
  %0 = tensor.empty() : tensor<2x2xf32>
  %1 = tensor.empty() : tensor<4x63x63x16xf32>
  //CHECK: scf.for %[[ARG1:.+]] =    
  %2 = scf.for %arg1 = %c0 to %c4 step %c1 iter_args(%arg2 = %1) -> (tensor<4x63x63x16xf32>) {
    //CHECK-DAG: %[[S:.+]] = memref.subview %[[ARG0]][%[[ARG1]], 0, 0, 0] [1, 126, 126, 16]
    //CHECK-DAG: %[[S1:.+]] = memref.subview %[[A0]][%[[ARG1]], 0, 0, 0] [1, 63, 63, 16]
    %extracted_slice = tensor.extract_slice %arg0[%arg1, 0, 0, 0] [1, 126, 126, 16] [1, 1, 1, 1] : tensor<4x126x126x16xf32> to tensor<1x126x126x16xf32>
    %3 = tensor.empty() : tensor<1x63x63x16xf32>
    //CHECK-DAG: linalg.fill
    //CHECK-SAME: ins(%[[CST]] : {{.+}}) outs(%[[S1]] : {{.+}})
    %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<1x63x63x16xf32>) -> tensor<1x63x63x16xf32>
    //CHECK: linalg.pooling_nhwc_max
    //CHECK-SAME: ins(%[[S]], %[[A]] : {{.+}}) outs(%[[S1]] : {{.+}})
    %5 = linalg.pooling_nhwc_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%extracted_slice, %0 : tensor<1x126x126x16xf32>, tensor<2x2xf32>) outs(%4 : tensor<1x63x63x16xf32>) -> tensor<1x63x63x16xf32>
    %inserted_slice = tensor.insert_slice %5 into %arg2[%arg1, 0, 0, 0] [1, 63, 63, 16] [1, 1, 1, 1] : tensor<1x63x63x16xf32> into tensor<4x63x63x16xf32>
    scf.yield %inserted_slice : tensor<4x63x63x16xf32>
  }
  return %2 : tensor<4x63x63x16xf32>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

//CHECK-LABEL: func.func @add2_generic
//CHECK-SAME:  (%[[ARG0:.+]]: {{.+}}, %[[ARG1:.+]]: {{.+}}, %[[ARG2:.+]]: {{.+}})
func.func @add2_generic(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>, %arg2: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %c4 = arith.constant 4 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  //CHECK: %[[A:.+]] = memref.alloc()
  %0 = tensor.empty() : tensor<4x4xf32>
  //CHECK: scf.for %[[ARG3:.+]] =
  %1 = scf.for %arg3 = %c0 to %c4 step %c1 iter_args(%arg4 = %0) -> (tensor<4x4xf32>) {
    //CHECK-DAG: %[[S:.+]] = memref.subview %[[ARG0]][%[[ARG3]], 0] [1, 4]
    //CHECK-DAG: %[[S0:.+]] = memref.subview %[[A]][%[[ARG3]], 0] [1, 4]
    //CHECK-DAG: %[[S1:.+]] = memref.subview %[[ARG1]][%[[ARG3]], 0] [1, 4]
    //CHECK-DAG: %[[S2:.+]] = memref.subview %[[ARG2]][%[[ARG3]], 0] [1, 4]
    %extracted_slice = tensor.extract_slice %arg0[%arg3, 0] [1, 4] [1, 1] : tensor<4x4xf32> to tensor<1x4xf32>
    %extracted_slice_0 = tensor.extract_slice %arg1[%arg3, 0] [1, 4] [1, 1] : tensor<4x4xf32> to tensor<1x4xf32>
    %extracted_slice_1 = tensor.extract_slice %arg2[%arg3, 0] [1, 4] [1, 1] : tensor<4x4xf32> to tensor<1x4xf32>
    %2 = tensor.empty() : tensor<1x4xf32>
    //CHECK: linalg.generic 
    //CHECK-SAME: ins(%[[S]], %[[S1]], %[[S2]] : {{.+}}) outs(%[[S0]] : {{.+}})
    %3 = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%extracted_slice, %extracted_slice_0, %extracted_slice_1 : tensor<1x4xf32>, tensor<1x4xf32>, tensor<1x4xf32>) outs(%2 : tensor<1x4xf32>) {
    ^bb0(%in: f32, %in_2: f32, %in_3: f32, %out: f32):
      %4 = arith.addf %in, %in_2 : f32
      %5 = arith.addf %4, %in_3 : f32
      linalg.yield %5 : f32
    } -> tensor<1x4xf32>
    %inserted_slice = tensor.insert_slice %3 into %arg4[%arg3, 0] [1, 4] [1, 1] : tensor<1x4xf32> into tensor<4x4xf32>
    scf.yield %inserted_slice : tensor<4x4xf32>
  }
  return %1 : tensor<4x4xf32>
}

// -----

//CHECK-LABEL: func.func @tile_linalg_matmul
//CHECK-SAME:  (%[[ARG0:.+]]: memref<128x128xf32>, %[[ARG1:.+]]: memref<128x128xf32>, %[[ARG2:.+]]: memref<128x128xf32>)
func.func @tile_linalg_matmul(%arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>, %arg2: tensor<128x128xf32>) -> tensor<128x128xf32> {
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  //CHECK: scf.for %[[ARG3:.+]] =
  %0 = scf.for %arg3 = %c0 to %c128 step %c8 iter_args(%arg4 = %arg2) -> (tensor<128x128xf32>) {
    %c0_0 = arith.constant 0 : index
    %c128_1 = arith.constant 128 : index
    //CHECK: scf.for %[[ARG4:.+]] =
    %1 = scf.for %arg5 = %c0_0 to %c128_1 step %c4 iter_args(%arg6 = %arg4) -> (tensor<128x128xf32>) {
      %c0_2 = arith.constant 0 : index
      %c128_3 = arith.constant 128 : index
      //CHECK: scf.for %[[ARG5:.+]] =
      %2 = scf.for %arg7 = %c0_2 to %c128_3 step %c2 iter_args(%arg8 = %arg6) -> (tensor<128x128xf32>) {
        //CHECK-DAG: %[[S:.+]] = memref.subview %[[ARG0]][%[[ARG5]], %[[ARG3]]] [2, 8] [1, 1]
        //CHECK-DAG: %[[S0:.+]] = memref.subview %[[ARG1]][%[[ARG3]], %[[ARG4]]] [8, 4] [1, 1]
        //CHECK-DAG: %[[S1:.+]] = memref.subview %[[ARG2]][%[[ARG5]], %[[ARG4]]] [2, 4] [1, 1]
        //CHECK: linalg.matmul 
        //CHECK-SAME: ins(%[[S]], %[[S0]] : {{.+}}) outs(%[[S1]] : {{.+}})
        %extracted_slice = tensor.extract_slice %arg0[%arg7, %arg3] [2, 8] [1, 1] : tensor<128x128xf32> to tensor<2x8xf32>
        %extracted_slice_4 = tensor.extract_slice %arg1[%arg3, %arg5] [8, 4] [1, 1] : tensor<128x128xf32> to tensor<8x4xf32>
        %extracted_slice_5 = tensor.extract_slice %arg8[%arg7, %arg5] [2, 4] [1, 1] : tensor<128x128xf32> to tensor<2x4xf32>
        %3 = linalg.matmul ins(%extracted_slice, %extracted_slice_4 : tensor<2x8xf32>, tensor<8x4xf32>) outs(%extracted_slice_5 : tensor<2x4xf32>) -> tensor<2x4xf32>
        %inserted_slice = tensor.insert_slice %3 into %arg8[%arg7, %arg5] [2, 4] [1, 1] : tensor<2x4xf32> into tensor<128x128xf32>
        scf.yield %inserted_slice : tensor<128x128xf32>
      }
      scf.yield %2 : tensor<128x128xf32>
    }
    scf.yield %1 : tensor<128x128xf32>
  }
  return %0 : tensor<128x128xf32>
}

// -----

//CHECK-LABEL: func.func @fuse_dot_attention
//CHECK-SAME:  (%[[ARG0:.+]]: memref<1024x32xf32>, %[[ARG1:.+]]: memref<32x512xf32>, %[[ARG2:.+]]: memref<512x32xf32>)
func.func @fuse_dot_attention(%arg0: tensor<1024x32xf32>, %arg1: tensor<32x512xf32>, %arg2: tensor<512x32xf32>) -> tensor<1024x32xf32> {
  %c1024 = arith.constant 1024 : index
  %c512 = arith.constant 512 : index
  %c0 = arith.constant 0 : index
  //CHECK-DAG: %[[CST:.+]] = arith.constant 0.000000e+00 : f32
  //CHECK-DAG: %[[CST0:.+]] = arith.constant 0xFF800000 : f32
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant 0xFF800000 : f32
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %0 = tensor.empty() : tensor<1024x32xf32>
  %1 = tensor.empty() : tensor<1024xf32>
  //CHECK-DAG: %[[A1:.+]] = memref.alloc() : memref<1024x32xf32>
  //CHECK-DAG: %[[A:.+]] = memref.alloc() : memref<1024xf32>
  //CHECK-DAG: %[[A2:.+]] = memref.alloc() : memref<1024xf32>
  //CHECK-DAG: linalg.fill ins(%[[CST0]] : {{.+}}) outs(%[[A2]] : {{.+}})
  //CHECK-DAG: linalg.fill ins(%[[CST]] : {{.+}}) outs(%[[A1]] : {{.+}})
  //CHECK-DAG: linalg.fill ins(%[[CST]] : {{.+}}) outs(%[[A]] : {{.+}})
  %2 = linalg.fill ins(%cst_0 : f32) outs(%1 : tensor<1024xf32>) -> tensor<1024xf32>
  %3 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1024x32xf32>) -> tensor<1024x32xf32>
  %4 = linalg.fill ins(%cst : f32) outs(%1 : tensor<1024xf32>) -> tensor<1024xf32>
  //CHECK: scf.for %[[ARG3:.+]] =
  %5:3 = scf.for %arg3 = %c0 to %c512 step %c8 iter_args(%arg4 = %3, %arg5 = %2, %arg6 = %4) -> (tensor<1024x32xf32>, tensor<1024xf32>, tensor<1024xf32>) {
    //CHECK: scf.for %[[ARG4:.+]] =
    %6:3 = scf.for %arg7 = %c0 to %c1024 step %c4 iter_args(%arg8 = %arg4, %arg9 = %arg5, %arg10 = %arg6) -> (tensor<1024x32xf32>, tensor<1024xf32>, tensor<1024xf32>) {
      //CHECK-DAG: %[[S:.+]] = memref.subview %[[ARG0]][%[[ARG4]], 0] [4, 32]
      //CHECK-DAG: %[[S9:.+]] = memref.subview %[[ARG1]][0, %[[ARG3]]] [32, 8]
      //CHECK-DAG: %[[A9:.+]] = memref.alloc() : memref<4x8xf32>
      //CHECK-DAG: %[[A8:.+]] = memref.alloc() : memref<4x8xf32>
      //CHECK-DAG: linalg.fill ins(%[[CST]] : {{.+}}) outs(%[[A8]] : {{.+}})
      //CHECK-DAG: linalg.matmul ins(%[[S]], %[[S9]] : {{.+}}) outs(%[[A8]] : {{.+}})
      //CHECK-DAG: %[[S6:.+]] = memref.subview %[[A2]][%[[ARG4]]] [4]
      //CHECK-DAG: %[[S7:.+]] = memref.subview %[[A]][%[[ARG4]]] [4]
      //CHECK-DAG: %[[A11:.+]] = memref.alloc() : memref<4xf32>
      //CHECK-DAG: linalg_ext.softmax dimension(1) ins(%[[A8]] : {{.+}}) outs(%[[A9]], %[[S6]], %[[S7]], %[[A11]] : {{.+}})
      //CHECK-DAG: %[[S5:.+]] = memref.subview %[[ARG2]][%[[ARG3]], 0] [8, 32]
      //CHECK-DAG: %[[S4:.+]] = memref.subview %[[A1]][%[[ARG4]], 0] [4, 32]
      //CHECK-DAG: %[[A12:.+]] = memref.alloc() : memref<4x4xf32>
      //CHECK-DAG: linalg_ext.diag ins(%[[A11]] : {{.+}}) outs(%[[A12]] : {{.+}})
      //CHECK-DAG: %[[A3:.+]] = memref.alloc() : memref<4x32xf32>
      //CHECK-DAG: linalg.fill ins(%[[CST]] : {{.+}}) outs(%[[A3]] : {{.+}})
      //CHECK-DAG: linalg.matmul ins(%[[A12]], %[[S4]] : {{.+}}) outs(%[[A3]] : {{.+}})
      //CHECK-DAG: linalg.matmul {__root__} ins(%[[A9]], %[[S5]] : {{.+}}) outs(%[[A3]] : {{.+}})
      //CHECK: memref.copy %[[A3]], %[[S4]]
      %extracted_slice = tensor.extract_slice %arg0[%arg7, 0] [4, 32] [1, 1] : tensor<1024x32xf32> to tensor<4x32xf32>
      %extracted_slice_1 = tensor.extract_slice %arg1[0, %arg3] [32, 8] [1, 1] : tensor<32x512xf32> to tensor<32x8xf32>
      %7 = tensor.empty() : tensor<4x8xf32>
      %8 = linalg.fill ins(%cst : f32) outs(%7 : tensor<4x8xf32>) -> tensor<4x8xf32>
      %9 = linalg.matmul ins(%extracted_slice, %extracted_slice_1 : tensor<4x32xf32>, tensor<32x8xf32>) outs(%8 : tensor<4x8xf32>) -> tensor<4x8xf32>
      %extracted_slice_2 = tensor.extract_slice %arg9[%arg7] [4] [1] : tensor<1024xf32> to tensor<4xf32>
      %extracted_slice_3 = tensor.extract_slice %arg10[%arg7] [4] [1] : tensor<1024xf32> to tensor<4xf32>
      %10 = tensor.empty() : tensor<4xf32>
      %11:4 = linalg_ext.softmax dimension(1) ins(%9 : tensor<4x8xf32>) outs(%7, %extracted_slice_2, %extracted_slice_3, %10 : tensor<4x8xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) : tensor<4x8xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>
      %extracted_slice_4 = tensor.extract_slice %arg2[%arg3, 0] [8, 32] [1, 1] : tensor<512x32xf32> to tensor<8x32xf32>
      %extracted_slice_5 = tensor.extract_slice %arg8[%arg7, 0] [4, 32] [1, 1] : tensor<1024x32xf32> to tensor<4x32xf32>
      %12 = tensor.empty() : tensor<4x4xf32>
      %13 = linalg_ext.diag ins(%11#3 : tensor<4xf32>) outs(%12 : tensor<4x4xf32>) : tensor<4x4xf32>
      %14 = tensor.empty() : tensor<4x32xf32>
      %15 = linalg.fill ins(%cst : f32) outs(%14 : tensor<4x32xf32>) -> tensor<4x32xf32>
      %16 = linalg.matmul ins(%13, %extracted_slice_5 : tensor<4x4xf32>, tensor<4x32xf32>) outs(%15 : tensor<4x32xf32>) -> tensor<4x32xf32>
      %17 = linalg.matmul {__root__} ins(%11#0, %extracted_slice_4 : tensor<4x8xf32>, tensor<8x32xf32>) outs(%16 : tensor<4x32xf32>) -> tensor<4x32xf32>
      %inserted_slice = tensor.insert_slice %17 into %arg8[%arg7, 0] [4, 32] [1, 1] : tensor<4x32xf32> into tensor<1024x32xf32>
      %inserted_slice_6 = tensor.insert_slice %11#1 into %arg9[%arg7] [4] [1] : tensor<4xf32> into tensor<1024xf32>
      %inserted_slice_7 = tensor.insert_slice %11#2 into %arg10[%arg7] [4] [1] : tensor<4xf32> into tensor<1024xf32>
      scf.yield %inserted_slice, %inserted_slice_6, %inserted_slice_7 : tensor<1024x32xf32>, tensor<1024xf32>, tensor<1024xf32>
    } {__byteir_parallel__}
    scf.yield %6#0, %6#1, %6#2 : tensor<1024x32xf32>, tensor<1024xf32>, tensor<1024xf32>
  }
  return %5#0 : tensor<1024x32xf32>
}