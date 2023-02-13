// RUN: byteir-opt %s -byteir-bufferize-opt | FileCheck %s

// CHECK-LABEL: func.func @main
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map4 = affine_map<(d0, d1, d2) -> (d2)>
#map5 = affine_map<(d0, d1) -> (d0)>
#map6 = affine_map<(d0) -> (d0)>
#map7 = affine_map<() -> ()>
#map8 = affine_map<(d0, d1) -> ()>
#map9 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func private @Unknown0(%arg0: tensor<2x128xi64>) -> (tensor<256xi64>, tensor<256xi1>) attributes {__byteir_elementwise_fusion__} {
    %c-100_i64 = arith.constant -100 : i64
    %collapsed = tensor.collapse_shape %arg0 [[0, 1]] : tensor<2x128xi64> into tensor<256xi64>
    %0 = tensor.empty() : tensor<2x128xi1>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<2x128xi64>) outs(%0 : tensor<2x128xi1>) {
    ^bb0(%in: i64, %out: i1):
      %2 = arith.cmpi ne, %in, %c-100_i64 : i64
      linalg.yield %2 : i1
    } -> tensor<2x128xi1>
    %collapsed_0 = tensor.collapse_shape %1 [[0, 1]] : tensor<2x128xi1> into tensor<256xi1>
    return %collapsed, %collapsed_0 : tensor<256xi64>, tensor<256xi1>
  }
  func.func private @Unknown1(%arg0: tensor<2x128xi64>) -> (tensor<256xui32>, tensor<256x1xi64>, tensor<256xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f64
    %c30522_i64 = arith.constant 30522 : i64
    %c0_i64 = arith.constant 0 : i64
    %0 = tensor.empty() : tensor<2x128xui32>
    %1 = tensor.empty() : tensor<2x128xi64>
    %2 = tensor.empty() : tensor<2x128xi1>
    %3:3 = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<2x128xi64>) outs(%0, %1, %2 : tensor<2x128xui32>, tensor<2x128xi64>, tensor<2x128xi1>) {
    ^bb0(%in: i64, %out: ui32, %out_2: i64, %out_3: i1):
      %4 = arith.trunci %in : i64 to i32
      %5 = builtin.unrealized_conversion_cast %4 : i32 to ui32
      %6 = arith.addi %in, %c30522_i64 : i64
      %7 = arith.cmpi slt, %in, %c0_i64 : i64
      %8 = arith.select %7, %6, %in : i64
      %9 = arith.sitofp %in : i64 to f64
      %10 = arith.cmpf une, %9, %cst : f64
      linalg.yield %5, %8, %10 : ui32, i64, i1
    } -> (tensor<2x128xui32>, tensor<2x128xi64>, tensor<2x128xi1>)
    %collapsed = tensor.collapse_shape %3#2 [[0, 1]] : tensor<2x128xi1> into tensor<256xi1>
    %collapsed_0 = tensor.collapse_shape %3#0 [[0, 1]] : tensor<2x128xui32> into tensor<256xui32>
    %collapsed_1 = tensor.collapse_shape %3#1 [[0, 1]] : tensor<2x128xi64> into tensor<256xi64>
    %expanded = tensor.expand_shape %collapsed_1 [[0, 1]] : tensor<256xi64> into tensor<256x1xi64>
    return %collapsed_0, %expanded, %collapsed : tensor<256xui32>, tensor<256x1xi64>, tensor<256xi1>
  }
  func.func private @Unknown2(%arg0: tensor<128xi64>) -> (tensor<256xui32>, tensor<256x1xi64>, tensor<256xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant -1.000000e+00 : f64
    %c2_i64 = arith.constant 2 : i64
    %c0_i64 = arith.constant 0 : i64
    %0 = tensor.empty() : tensor<2x128xui32>
    %1 = tensor.empty() : tensor<2x128xi64>
    %2 = tensor.empty() : tensor<2x128xi1>
    %3:3 = linalg.generic {indexing_maps = [#map1, #map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<128xi64>) outs(%0, %1, %2 : tensor<2x128xui32>, tensor<2x128xi64>, tensor<2x128xi1>) {
    ^bb0(%in: i64, %out: ui32, %out_2: i64, %out_3: i1):
      %4 = arith.addi %in, %c2_i64 : i64
      %5 = arith.trunci %in : i64 to i32
      %6 = builtin.unrealized_conversion_cast %5 : i32 to ui32
      %7 = arith.cmpi slt, %in, %c0_i64 : i64
      %8 = arith.select %7, %4, %in : i64
      %9 = arith.sitofp %in : i64 to f64
      %10 = arith.cmpf une, %9, %cst : f64
      linalg.yield %6, %8, %10 : ui32, i64, i1
    } -> (tensor<2x128xui32>, tensor<2x128xi64>, tensor<2x128xi1>)
    %collapsed = tensor.collapse_shape %3#2 [[0, 1]] : tensor<2x128xi1> into tensor<256xi1>
    %collapsed_0 = tensor.collapse_shape %3#0 [[0, 1]] : tensor<2x128xui32> into tensor<256xui32>
    %collapsed_1 = tensor.collapse_shape %3#1 [[0, 1]] : tensor<2x128xi64> into tensor<256xi64>
    %expanded = tensor.expand_shape %collapsed_1 [[0, 1]] : tensor<256xi64> into tensor<256x1xi64>
    return %collapsed_0, %expanded, %collapsed : tensor<256xui32>, tensor<256x1xi64>, tensor<256xi1>
  }
  func.func private @Unknown3(%arg0: tensor<1x128xi64>) -> (tensor<128xui32>, tensor<128x1xi64>, tensor<128xi1>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant -1.000000e+00 : f64
    %c512_i64 = arith.constant 512 : i64
    %c0_i64 = arith.constant 0 : i64
    %0 = tensor.empty() : tensor<1x128xui32>
    %1 = tensor.empty() : tensor<1x128xi64>
    %2 = tensor.empty() : tensor<1x128xi1>
    %3:3 = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<1x128xi64>) outs(%0, %1, %2 : tensor<1x128xui32>, tensor<1x128xi64>, tensor<1x128xi1>) {
    ^bb0(%in: i64, %out: ui32, %out_2: i64, %out_3: i1):
      %4 = arith.trunci %in : i64 to i32
      %5 = builtin.unrealized_conversion_cast %4 : i32 to ui32
      %6 = arith.addi %in, %c512_i64 : i64
      %7 = arith.cmpi slt, %in, %c0_i64 : i64
      %8 = arith.select %7, %6, %in : i64
      %9 = arith.sitofp %in : i64 to f64
      %10 = arith.cmpf une, %9, %cst : f64
      linalg.yield %5, %8, %10 : ui32, i64, i1
    } -> (tensor<1x128xui32>, tensor<1x128xi64>, tensor<1x128xi1>)
    %collapsed = tensor.collapse_shape %3#2 [[0, 1]] : tensor<1x128xi1> into tensor<128xi1>
    %collapsed_0 = tensor.collapse_shape %3#0 [[0, 1]] : tensor<1x128xui32> into tensor<128xui32>
    %collapsed_1 = tensor.collapse_shape %3#1 [[0, 1]] : tensor<1x128xi64> into tensor<128xi64>
    %expanded = tensor.expand_shape %collapsed_1 [[0, 1]] : tensor<128xi64> into tensor<128x1xi64>
    return %collapsed_0, %expanded, %collapsed : tensor<128xui32>, tensor<128x1xi64>, tensor<128xi1>
  }
  func.func private @Unknown4(%arg0: tensor<256x128xf32>, %arg1: tensor<256x128xf32>, %arg2: tensor<128x128xf32>) -> tensor<2x128x128xf32> attributes {__byteir_elementwise_fusion__} {
    %expanded = tensor.expand_shape %arg0 [[0, 1], [2]] : tensor<256x128xf32> into tensor<2x128x128xf32>
    %expanded_0 = tensor.expand_shape %arg1 [[0, 1], [2]] : tensor<256x128xf32> into tensor<2x128x128xf32>
    %0 = tensor.empty() : tensor<2x128x128xf32>
    %1 = linalg.generic {indexing_maps = [#map2, #map2, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded, %expanded_0, %arg2 : tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<128x128xf32>) outs(%0 : tensor<2x128x128xf32>) {
    ^bb0(%in: f32, %in_1: f32, %in_2: f32, %out: f32):
      %2 = arith.addf %in, %in_1 : f32
      %3 = arith.addf %2, %in_2 : f32
      linalg.yield %3 : f32
    } -> tensor<2x128x128xf32>
    return %1 : tensor<2x128x128xf32>
  }
  func.func private @Unknown5(%arg0: tensor<256x30522xf32>, %arg1: tensor<30522xf32>) -> (tensor<2x128x30522xf32>, tensor<256x30522xf32>) attributes {__byteir_elementwise_fusion__} {
    %expanded = tensor.expand_shape %arg0 [[0, 1], [2]] : tensor<256x30522xf32> into tensor<2x128x30522xf32>
    %0 = tensor.empty() : tensor<2x128x30522xf32>
    %1 = linalg.generic {indexing_maps = [#map2, #map4, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded, %arg1 : tensor<2x128x30522xf32>, tensor<30522xf32>) outs(%0 : tensor<2x128x30522xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %2 = arith.addf %in, %in_0 : f32
      linalg.yield %2 : f32
    } -> tensor<2x128x30522xf32>
    %collapsed = tensor.collapse_shape %1 [[0, 1], [2]] : tensor<2x128x30522xf32> into tensor<256x30522xf32>
    return %1, %collapsed : tensor<2x128x30522xf32>, tensor<256x30522xf32>
  }
  func.func private @Unknown6(%arg0: tensor<256xf32>, %arg1: tensor<256x30522xf32>) -> (tensor<256x30522xf32>, tensor<256x30522xf32>) attributes {__byteir_elementwise_fusion__} {
    %0 = tensor.empty() : tensor<256x30522xf32>
    %1:2 = linalg.generic {indexing_maps = [#map, #map5, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg1, %arg0 : tensor<256x30522xf32>, tensor<256xf32>) outs(%0, %0 : tensor<256x30522xf32>, tensor<256x30522xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32, %out_1: f32):
      %2 = arith.subf %in, %in_0 : f32
      %3 = math.exp %2 : f32
      linalg.yield %2, %3 : f32, f32
    } -> (tensor<256x30522xf32>, tensor<256x30522xf32>)
    return %1#0, %1#1 : tensor<256x30522xf32>, tensor<256x30522xf32>
  }
  func.func private @Unknown7(%arg0: tensor<256xf32>) -> tensor<256xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = tensor.empty() : tensor<256xf32>
    %1 = linalg.generic {indexing_maps = [#map6, #map6], iterator_types = ["parallel"]} ins(%arg0 : tensor<256xf32>) outs(%0 : tensor<256xf32>) {
    ^bb0(%in: f32, %out: f32):
      %2 = math.log %in : f32
      linalg.yield %2 : f32
    } -> tensor<256xf32>
    return %1 : tensor<256xf32>
  }
  func.func private @Unknown8(%arg0: tensor<256xf32>, %arg1: tensor<256x30522xf32>, %arg2: tensor<256xi64>, %arg3: tensor<256xi1>) -> (tensor<256x30522xf32>, tensor<256x30522xf32>, tensor<256x30522xf32>, tensor<256x30522xf32>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<256x30522xf32>
    %1:4 = linalg.generic {indexing_maps = [#map5, #map5, #map, #map5, #map, #map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg3, %arg2, %arg1, %arg0 : tensor<256xi1>, tensor<256xi64>, tensor<256x30522xf32>, tensor<256xf32>) outs(%0, %0, %0, %0 : tensor<256x30522xf32>, tensor<256x30522xf32>, tensor<256x30522xf32>, tensor<256x30522xf32>) {
    ^bb0(%in: i1, %in_1: i64, %in_2: f32, %in_3: f32, %out: f32, %out_4: f32, %out_5: f32, %out_6: f32):
      %2 = linalg.index 1 : index
      %3 = arith.index_cast %2 : index to i64
      %4 = arith.cmpi eq, %in_1, %3 : i64
      %5 = arith.select %4, %cst, %cst_0 : f32
      %6 = arith.select %in, %cst, %cst_0 : f32
      %7 = arith.mulf %6, %5 : f32
      %8 = arith.subf %in_2, %in_3 : f32
      %9 = arith.negf %5 : f32
      %10 = arith.mulf %9, %8 : f32
      %11 = arith.cmpf une, %5, %cst : f32
      %12 = arith.select %11, %cst_0, %10 : f32
      %13 = arith.mulf %12, %7 : f32
      %14 = arith.mulf %9, %7 : f32
      %15 = math.exp %8 : f32
      linalg.yield %7, %13, %14, %15 : f32, f32, f32, f32
    } -> (tensor<256x30522xf32>, tensor<256x30522xf32>, tensor<256x30522xf32>, tensor<256x30522xf32>)
    return %1#0, %1#1, %1#2, %1#3 : tensor<256x30522xf32>, tensor<256x30522xf32>, tensor<256x30522xf32>, tensor<256x30522xf32>
  }
  func.func private @Unknown9(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> attributes {__byteir_elementwise_fusion__} {
    %0 = tensor.empty() : tensor<f32>
    %1 = linalg.generic {indexing_maps = [#map7, #map7, #map7], iterator_types = []} ins(%arg0, %arg1 : tensor<f32>, tensor<f32>) outs(%0 : tensor<f32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %2 = arith.divf %in, %in_0 : f32
      linalg.yield %2 : f32
    } -> tensor<f32>
    return %1 : tensor<f32>
  }
  func.func private @Unknown10(%arg0: tensor<f32>) -> tensor<f32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<f32>
    %1 = linalg.generic {indexing_maps = [#map7, #map7], iterator_types = []} ins(%arg0 : tensor<f32>) outs(%0 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %2 = arith.cmpf une, %in, %cst_0 : f32
      %3 = arith.select %2, %in, %cst : f32
      linalg.yield %3 : f32
    } -> tensor<f32>
    return %1 : tensor<f32>
  }
  func.func private @Unknown11(%arg0: tensor<f32>, %arg1: tensor<256x30522xf32>) -> tensor<256x30522xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = tensor.empty() : tensor<256x30522xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map8, #map], iterator_types = ["parallel", "parallel"]} ins(%arg1, %arg0 : tensor<256x30522xf32>, tensor<f32>) outs(%0 : tensor<256x30522xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %2 = arith.divf %in, %in_0 : f32
      linalg.yield %2 : f32
    } -> tensor<256x30522xf32>
    return %1 : tensor<256x30522xf32>
  }
  func.func private @Unknown12(%arg0: tensor<256xf32>, %arg1: tensor<256x30522xf32>, %arg2: tensor<256x30522xf32>) -> (tensor<256x30522xf32>, tensor<2x128x30522xf32>) attributes {__byteir_elementwise_fusion__} {
    %expanded = tensor.expand_shape %arg2 [[0, 1], [2]] : tensor<256x30522xf32> into tensor<2x128x30522xf32>
    %expanded_0 = tensor.expand_shape %arg1 [[0, 1], [2]] : tensor<256x30522xf32> into tensor<2x128x30522xf32>
    %expanded_1 = tensor.expand_shape %arg0 [[0, 1]] : tensor<256xf32> into tensor<2x128xf32>
    %0 = tensor.empty() : tensor<2x128x30522xf32>
    %1 = linalg.generic {indexing_maps = [#map2, #map2, #map9, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded, %expanded_0, %expanded_1 : tensor<2x128x30522xf32>, tensor<2x128x30522xf32>, tensor<2x128xf32>) outs(%0 : tensor<2x128x30522xf32>) {
    ^bb0(%in: f32, %in_2: f32, %in_3: f32, %out: f32):
      %2 = arith.mulf %in_2, %in_3 : f32
      %3 = arith.subf %in, %2 : f32
      linalg.yield %3 : f32
    } -> tensor<2x128x30522xf32>
    %collapsed = tensor.collapse_shape %1 [[0, 1], [2]] : tensor<2x128x30522xf32> into tensor<256x30522xf32>
    return %collapsed, %1 : tensor<256x30522xf32>, tensor<2x128x30522xf32>
  }
  func.func private @MatmulOp13(%arg0: tensor<256x128xf32>, %arg1: tensor<256x30522xf32>) -> tensor<30522x128xf32> attributes {__byre__lhs_contracting_dimension = 0 : i64, __byre__output_transpose, __byre__rhs_contracting_dimension = 0 : i64, byre_compute_name = "MatmulOp"} {
    %0 = "mhlo.dot_general"(%arg0, %arg1) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [0], rhs_contracting_dimensions = [0]>, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<256x128xf32>, tensor<256x30522xf32>) -> tensor<128x30522xf32>
    %1 = "mhlo.transpose"(%0) {minor_to_major = dense<[0, 1]> : tensor<2xindex>, permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<128x30522xf32>) -> tensor<30522x128xf32>
    return %1 : tensor<30522x128xf32>
  }
  func.func private @Unknown14(%arg0: tensor<2x128x128xf32>, %arg1: tensor<2x128x128xf32>) -> tensor<2x128x128xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = tensor.empty() : tensor<2x128x128xf32>
    %1 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : tensor<2x128x128xf32>, tensor<2x128x128xf32>) outs(%0 : tensor<2x128x128xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %2 = arith.addf %in, %in_0 : f32
      linalg.yield %2 : f32
    } -> tensor<2x128x128xf32>
    return %1 : tensor<2x128x128xf32>
  }
  func.func private @Unknown15(%arg0: tensor<2x128x128xf32>, %arg1: tensor<2x128x128xf32>, %arg2: tensor<2x128x128xf32>, %arg3: tensor<2x128x128xf32>) -> tensor<2x128x128xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = tensor.empty() : tensor<2x128x128xf32>
    %1 = linalg.generic {indexing_maps = [#map2, #map2, #map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0, %arg1, %arg2, %arg3 : tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xf32>) outs(%0 : tensor<2x128x128xf32>) {
    ^bb0(%in: f32, %in_0: f32, %in_1: f32, %in_2: f32, %out: f32):
      %2 = arith.addf %in, %in_0 : f32
      %3 = arith.addf %2, %in_1 : f32
      %4 = arith.addf %3, %in_2 : f32
      linalg.yield %4 : f32
    } -> tensor<2x128x128xf32>
    return %1 : tensor<2x128x128xf32>
  }
  func.func private @Unknown16(%arg0: tensor<2x128x128xf32>, %arg1: tensor<2x128x128xf32>) -> tensor<2x128x128xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = tensor.empty() : tensor<2x128x128xf32>
    %1 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : tensor<2x128x128xf32>, tensor<2x128x128xf32>) outs(%0 : tensor<2x128x128xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %2 = arith.addf %in, %in_0 : f32
      linalg.yield %2 : f32
    } -> tensor<2x128x128xf32>
    return %1 : tensor<2x128x128xf32>
  }
  func.func private @Unknown17(%arg0: tensor<2x128x128xf32>, %arg1: tensor<2x128x128xf32>, %arg2: tensor<2x128x128xf32>, %arg3: tensor<2x128x128xf32>) -> tensor<2x128x128xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = tensor.empty() : tensor<2x128x128xf32>
    %1 = linalg.generic {indexing_maps = [#map2, #map2, #map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0, %arg1, %arg2, %arg3 : tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xf32>, tensor<2x128x128xf32>) outs(%0 : tensor<2x128x128xf32>) {
    ^bb0(%in: f32, %in_0: f32, %in_1: f32, %in_2: f32, %out: f32):
      %2 = arith.addf %in, %in_0 : f32
      %3 = arith.addf %2, %in_1 : f32
      %4 = arith.addf %3, %in_2 : f32
      linalg.yield %4 : f32
    } -> tensor<2x128x128xf32>
    return %1 : tensor<2x128x128xf32>
  }
  func.func private @Unknown18(%arg0: tensor<256xi1>, %arg1: tensor<2x128x128xf32>, %arg2: tensor<256xi1>) -> (tensor<256x128xf32>, tensor<256x128xf32>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f32
    %expanded = tensor.expand_shape %arg0 [[0, 1]] : tensor<256xi1> into tensor<2x128xi1>
    %0 = tensor.empty() : tensor<2x128x128xf32>
    %expanded_0 = tensor.expand_shape %arg2 [[0, 1]] : tensor<256xi1> into tensor<2x128xi1>
    %1:2 = linalg.generic {indexing_maps = [#map9, #map9, #map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_0, %expanded, %arg1 : tensor<2x128xi1>, tensor<2x128xi1>, tensor<2x128x128xf32>) outs(%0, %0 : tensor<2x128x128xf32>, tensor<2x128x128xf32>) {
    ^bb0(%in: i1, %in_2: i1, %in_3: f32, %out: f32, %out_4: f32):
      %2 = arith.select %in_2, %in_3, %cst : f32
      %3 = arith.select %in, %in_3, %cst : f32
      linalg.yield %2, %3 : f32, f32
    } -> (tensor<2x128x128xf32>, tensor<2x128x128xf32>)
    %collapsed = tensor.collapse_shape %1#1 [[0, 1], [2]] : tensor<2x128x128xf32> into tensor<256x128xf32>
    %collapsed_1 = tensor.collapse_shape %1#0 [[0, 1], [2]] : tensor<2x128x128xf32> into tensor<256x128xf32>
    return %collapsed_1, %collapsed : tensor<256x128xf32>, tensor<256x128xf32>
  }
  func.func private @Unknown19(%arg0: tensor<128xi1>, %arg1: tensor<128x128xf32>) -> tensor<128x128xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<128x128xf32>
    %1 = linalg.generic {indexing_maps = [#map5, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<128xi1>, tensor<128x128xf32>) outs(%0 : tensor<128x128xf32>) {
    ^bb0(%in: i1, %in_0: f32, %out: f32):
      %2 = arith.select %in, %in_0, %cst : f32
      linalg.yield %2 : f32
    } -> tensor<128x128xf32>
    return %1 : tensor<128x128xf32>
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

