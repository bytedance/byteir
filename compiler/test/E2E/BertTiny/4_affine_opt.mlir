// RUN: byteir-opt %s -affine-opt | FileCheck %s

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
  func.func private @Unknown0(%arg0: memref<2x128xi64>) -> (memref<256xi64>, memref<256xi1>) attributes {__byteir_elementwise_fusion__} {
    %c-100_i64 = arith.constant -100 : i64
    %collapse_shape = memref.collapse_shape %arg0 [[0, 1]] : memref<2x128xi64> into memref<256xi64>
    %alloc = memref.alloc() : memref<2x128xi1>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0 : memref<2x128xi64>) outs(%alloc : memref<2x128xi1>) {
    ^bb0(%in: i64, %out: i1):
      %0 = arith.cmpi ne, %in, %c-100_i64 : i64
      linalg.yield %0 : i1
    }
    %collapse_shape_0 = memref.collapse_shape %alloc [[0, 1]] : memref<2x128xi1> into memref<256xi1>
    return %collapse_shape, %collapse_shape_0 : memref<256xi64>, memref<256xi1>
  }
  func.func private @Unknown1(%arg0: memref<2x128xi64>) -> (memref<256xui32>, memref<256x1xi64>, memref<256xi1>) attributes {__byteir_elementwise_fusion__} {
    %c30522_i64 = arith.constant 30522 : i64
    %c0_i64 = arith.constant 0 : i64
    %alloc = memref.alloc() : memref<2x128xui32>
    %alloc_0 = memref.alloc() : memref<2x128xi64>
    %alloc_1 = memref.alloc() : memref<2x128xi1>
    linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0 : memref<2x128xi64>) outs(%alloc, %alloc_0, %alloc_1 : memref<2x128xui32>, memref<2x128xi64>, memref<2x128xi1>) {
    ^bb0(%in: i64, %out: ui32, %out_4: i64, %out_5: i1):
      %0 = arith.trunci %in : i64 to i32
      %1 = builtin.unrealized_conversion_cast %0 : i32 to ui32
      %2 = arith.addi %in, %c30522_i64 : i64
      %3 = arith.cmpi slt, %in, %c0_i64 : i64
      %4 = arith.select %3, %2, %in : i64
      %5 = arith.cmpi ne, %in, %c0_i64 : i64
      linalg.yield %1, %4, %5 : ui32, i64, i1
    }
    %collapse_shape = memref.collapse_shape %alloc_1 [[0, 1]] : memref<2x128xi1> into memref<256xi1>
    %collapse_shape_2 = memref.collapse_shape %alloc [[0, 1]] : memref<2x128xui32> into memref<256xui32>
    %collapse_shape_3 = memref.collapse_shape %alloc_0 [[0, 1]] : memref<2x128xi64> into memref<256xi64>
    %expand_shape = memref.expand_shape %collapse_shape_3 [[0, 1]] : memref<256xi64> into memref<256x1xi64>
    return %collapse_shape_2, %expand_shape, %collapse_shape : memref<256xui32>, memref<256x1xi64>, memref<256xi1>
  }
  func.func private @Unknown2(%arg0: memref<128xi64>) -> (memref<256xui32>, memref<256x1xi64>, memref<256xi1>) attributes {__byteir_elementwise_fusion__} {
    %c-1_i64 = arith.constant -1 : i64
    %c2_i64 = arith.constant 2 : i64
    %c0_i64 = arith.constant 0 : i64
    %alloc = memref.alloc() : memref<2x128xui32>
    %alloc_0 = memref.alloc() : memref<2x128xi64>
    %alloc_1 = memref.alloc() : memref<2x128xi1>
    linalg.generic {indexing_maps = [#map1, #map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0 : memref<128xi64>) outs(%alloc, %alloc_0, %alloc_1 : memref<2x128xui32>, memref<2x128xi64>, memref<2x128xi1>) {
    ^bb0(%in: i64, %out: ui32, %out_4: i64, %out_5: i1):
      %0 = arith.addi %in, %c2_i64 : i64
      %1 = arith.trunci %in : i64 to i32
      %2 = builtin.unrealized_conversion_cast %1 : i32 to ui32
      %3 = arith.cmpi slt, %in, %c0_i64 : i64
      %4 = arith.select %3, %0, %in : i64
      %5 = arith.cmpi ne, %in, %c-1_i64 : i64
      linalg.yield %2, %4, %5 : ui32, i64, i1
    }
    %collapse_shape = memref.collapse_shape %alloc_1 [[0, 1]] : memref<2x128xi1> into memref<256xi1>
    %collapse_shape_2 = memref.collapse_shape %alloc [[0, 1]] : memref<2x128xui32> into memref<256xui32>
    %collapse_shape_3 = memref.collapse_shape %alloc_0 [[0, 1]] : memref<2x128xi64> into memref<256xi64>
    %expand_shape = memref.expand_shape %collapse_shape_3 [[0, 1]] : memref<256xi64> into memref<256x1xi64>
    return %collapse_shape_2, %expand_shape, %collapse_shape : memref<256xui32>, memref<256x1xi64>, memref<256xi1>
  }
  func.func private @Unknown3(%arg0: memref<1x128xi64>) -> (memref<128xui32>, memref<128x1xi64>, memref<128xi1>) attributes {__byteir_elementwise_fusion__} {
    %c-1_i64 = arith.constant -1 : i64
    %c512_i64 = arith.constant 512 : i64
    %c0_i64 = arith.constant 0 : i64
    %alloc = memref.alloc() : memref<1x128xui32>
    %alloc_0 = memref.alloc() : memref<1x128xi64>
    %alloc_1 = memref.alloc() : memref<1x128xi1>
    linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0 : memref<1x128xi64>) outs(%alloc, %alloc_0, %alloc_1 : memref<1x128xui32>, memref<1x128xi64>, memref<1x128xi1>) {
    ^bb0(%in: i64, %out: ui32, %out_4: i64, %out_5: i1):
      %0 = arith.trunci %in : i64 to i32
      %1 = builtin.unrealized_conversion_cast %0 : i32 to ui32
      %2 = arith.addi %in, %c512_i64 : i64
      %3 = arith.cmpi slt, %in, %c0_i64 : i64
      %4 = arith.select %3, %2, %in : i64
      %5 = arith.cmpi ne, %in, %c-1_i64 : i64
      linalg.yield %1, %4, %5 : ui32, i64, i1
    }
    %collapse_shape = memref.collapse_shape %alloc_1 [[0, 1]] : memref<1x128xi1> into memref<128xi1>
    %collapse_shape_2 = memref.collapse_shape %alloc [[0, 1]] : memref<1x128xui32> into memref<128xui32>
    %collapse_shape_3 = memref.collapse_shape %alloc_0 [[0, 1]] : memref<1x128xi64> into memref<128xi64>
    %expand_shape = memref.expand_shape %collapse_shape_3 [[0, 1]] : memref<128xi64> into memref<128x1xi64>
    return %collapse_shape_2, %expand_shape, %collapse_shape : memref<128xui32>, memref<128x1xi64>, memref<128xi1>
  }
  func.func private @Unknown4(%arg0: memref<256x128xf32>, %arg1: memref<256x128xf32>, %arg2: memref<128x128xf32>) -> memref<2x128x128xf32> attributes {__byteir_elementwise_fusion__} {
    %expand_shape = memref.expand_shape %arg0 [[0, 1], [2]] : memref<256x128xf32> into memref<2x128x128xf32>
    %expand_shape_0 = memref.expand_shape %arg1 [[0, 1], [2]] : memref<256x128xf32> into memref<2x128x128xf32>
    %alloc = memref.alloc() : memref<2x128x128xf32>
    linalg.generic {indexing_maps = [#map2, #map2, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expand_shape, %expand_shape_0, %arg2 : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>) outs(%alloc : memref<2x128x128xf32>) {
    ^bb0(%in: f32, %in_1: f32, %in_2: f32, %out: f32):
      %0 = arith.addf %in, %in_1 : f32
      %1 = arith.addf %0, %in_2 : f32
      linalg.yield %1 : f32
    }
    return %alloc : memref<2x128x128xf32>
  }
  func.func private @Unknown5(%arg0: memref<256x30522xf32>, %arg1: memref<30522xf32>) -> (memref<2x128x30522xf32>, memref<256x30522xf32>) attributes {__byteir_elementwise_fusion__} {
    %expand_shape = memref.expand_shape %arg0 [[0, 1], [2]] : memref<256x30522xf32> into memref<2x128x30522xf32>
    %alloc = memref.alloc() : memref<2x128x30522xf32>
    linalg.generic {indexing_maps = [#map2, #map4, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expand_shape, %arg1 : memref<2x128x30522xf32>, memref<30522xf32>) outs(%alloc : memref<2x128x30522xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %0 = arith.addf %in, %in_0 : f32
      linalg.yield %0 : f32
    }
    %collapse_shape = memref.collapse_shape %alloc [[0, 1], [2]] : memref<2x128x30522xf32> into memref<256x30522xf32>
    return %alloc, %collapse_shape : memref<2x128x30522xf32>, memref<256x30522xf32>
  }
  func.func private @Unknown6(%arg0: memref<256xf32>, %arg1: memref<256x30522xf32>) -> (memref<256x30522xf32>, memref<256x30522xf32>) attributes {__byteir_elementwise_fusion__} {
    %alloc = memref.alloc() : memref<256x30522xf32>
    %alloc_0 = memref.alloc() : memref<256x30522xf32>
    linalg.generic {indexing_maps = [#map, #map5, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg1, %arg0 : memref<256x30522xf32>, memref<256xf32>) outs(%alloc, %alloc_0 : memref<256x30522xf32>, memref<256x30522xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32, %out_2: f32):
      %0 = arith.subf %in, %in_1 : f32
      %1 = math.exp %0 : f32
      linalg.yield %0, %1 : f32, f32
    }
    return %alloc, %alloc_0 : memref<256x30522xf32>, memref<256x30522xf32>
  }
  func.func private @Unknown7(%arg0: memref<256xf32>) -> memref<256xf32> attributes {__byteir_elementwise_fusion__} {
    %alloc = memref.alloc() : memref<256xf32>
    linalg.generic {indexing_maps = [#map6, #map6], iterator_types = ["parallel"]} ins(%arg0 : memref<256xf32>) outs(%alloc : memref<256xf32>) {
    ^bb0(%in: f32, %out: f32):
      %0 = math.log %in : f32
      linalg.yield %0 : f32
    }
    return %alloc : memref<256xf32>
  }
  func.func private @Unknown8(%arg0: memref<256xf32>, %arg1: memref<256x30522xf32>, %arg2: memref<256xi64>, %arg3: memref<256xi1>) -> (memref<256x30522xf32>, memref<256x30522xf32>, memref<256x30522xf32>, memref<256x30522xf32>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() : memref<256x30522xf32>
    %alloc_1 = memref.alloc() : memref<256x30522xf32>
    %alloc_2 = memref.alloc() : memref<256x30522xf32>
    %alloc_3 = memref.alloc() : memref<256x30522xf32>
    linalg.generic {indexing_maps = [#map5, #map5, #map, #map5, #map, #map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg3, %arg2, %arg1, %arg0 : memref<256xi1>, memref<256xi64>, memref<256x30522xf32>, memref<256xf32>) outs(%alloc, %alloc_1, %alloc_2, %alloc_3 : memref<256x30522xf32>, memref<256x30522xf32>, memref<256x30522xf32>, memref<256x30522xf32>) {
    ^bb0(%in: i1, %in_4: i64, %in_5: f32, %in_6: f32, %out: f32, %out_7: f32, %out_8: f32, %out_9: f32):
      %0 = linalg.index 1 : index
      %1 = arith.index_cast %0 : index to i64
      %2 = arith.cmpi eq, %in_4, %1 : i64
      %3 = arith.select %2, %cst, %cst_0 : f32
      %4 = arith.select %in, %cst, %cst_0 : f32
      %5 = arith.mulf %4, %3 : f32
      %6 = arith.subf %in_5, %in_6 : f32
      %7 = arith.negf %3 : f32
      %8 = arith.mulf %7, %6 : f32
      %9 = arith.cmpf une, %3, %cst : f32
      %10 = arith.select %9, %cst_0, %8 : f32
      %11 = arith.mulf %10, %5 : f32
      %12 = arith.mulf %7, %5 : f32
      %13 = math.exp %6 : f32
      linalg.yield %5, %11, %12, %13 : f32, f32, f32, f32
    }
    return %alloc, %alloc_1, %alloc_2, %alloc_3 : memref<256x30522xf32>, memref<256x30522xf32>, memref<256x30522xf32>, memref<256x30522xf32>
  }
  func.func private @Unknown9(%arg0: memref<f32>, %arg1: memref<f32>) -> memref<f32> attributes {__byteir_elementwise_fusion__} {
    %alloc = memref.alloc() : memref<f32>
    linalg.generic {indexing_maps = [#map7, #map7, #map7], iterator_types = []} ins(%arg0, %arg1 : memref<f32>, memref<f32>) outs(%alloc : memref<f32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %0 = arith.divf %in, %in_0 : f32
      linalg.yield %0 : f32
    }
    return %alloc : memref<f32>
  }
  func.func private @Unknown10(%arg0: memref<f32>) -> memref<f32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() : memref<f32>
    linalg.generic {indexing_maps = [#map7, #map7], iterator_types = []} ins(%arg0 : memref<f32>) outs(%alloc : memref<f32>) {
    ^bb0(%in: f32, %out: f32):
      %0 = arith.cmpf une, %in, %cst_0 : f32
      %1 = arith.select %0, %in, %cst : f32
      linalg.yield %1 : f32
    }
    return %alloc : memref<f32>
  }
  func.func private @Unknown11(%arg0: memref<f32>, %arg1: memref<256x30522xf32>) -> memref<256x30522xf32> attributes {__byteir_elementwise_fusion__} {
    %alloc = memref.alloc() : memref<256x30522xf32>
    linalg.generic {indexing_maps = [#map, #map8, #map], iterator_types = ["parallel", "parallel"]} ins(%arg1, %arg0 : memref<256x30522xf32>, memref<f32>) outs(%alloc : memref<256x30522xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %0 = arith.divf %in, %in_0 : f32
      linalg.yield %0 : f32
    }
    return %alloc : memref<256x30522xf32>
  }
  func.func private @Unknown12(%arg0: memref<256xf32>, %arg1: memref<256x30522xf32>, %arg2: memref<256x30522xf32>) -> (memref<256x30522xf32>, memref<2x128x30522xf32>) attributes {__byteir_elementwise_fusion__} {
    %expand_shape = memref.expand_shape %arg2 [[0, 1], [2]] : memref<256x30522xf32> into memref<2x128x30522xf32>
    %expand_shape_0 = memref.expand_shape %arg1 [[0, 1], [2]] : memref<256x30522xf32> into memref<2x128x30522xf32>
    %expand_shape_1 = memref.expand_shape %arg0 [[0, 1]] : memref<256xf32> into memref<2x128xf32>
    %alloc = memref.alloc() : memref<2x128x30522xf32>
    linalg.generic {indexing_maps = [#map2, #map2, #map9, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expand_shape, %expand_shape_0, %expand_shape_1 : memref<2x128x30522xf32>, memref<2x128x30522xf32>, memref<2x128xf32>) outs(%alloc : memref<2x128x30522xf32>) {
    ^bb0(%in: f32, %in_2: f32, %in_3: f32, %out: f32):
      %0 = arith.mulf %in_2, %in_3 : f32
      %1 = arith.subf %in, %0 : f32
      linalg.yield %1 : f32
    }
    %collapse_shape = memref.collapse_shape %alloc [[0, 1], [2]] : memref<2x128x30522xf32> into memref<256x30522xf32>
    return %collapse_shape, %alloc : memref<256x30522xf32>, memref<2x128x30522xf32>
  }
  func.func private @MatmulOp13(%arg0: memref<256x128xf32>, %arg1: memref<256x30522xf32>) -> memref<30522x128xf32> attributes {__byre__lhs_contracting_dimension = 0 : i64, __byre__output_transpose, __byre__rhs_contracting_dimension = 0 : i64, byre_compute_name = "MatmulOp"} {
    %alloc = memref.alloc() : memref<128x30522xf32>
    "lmhlo.dot"(%arg0, %arg1, %alloc) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [0], rhs_contracting_dimensions = [0]>, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<256x128xf32>, memref<256x30522xf32>, memref<128x30522xf32>) -> ()
    %alloc_0 = memref.alloc() : memref<30522x128xf32>
    "lmhlo.transpose"(%alloc, %alloc_0) {minor_to_major = dense<[0, 1]> : tensor<2xindex>, permutation = dense<[1, 0]> : tensor<2xi64>} : (memref<128x30522xf32>, memref<30522x128xf32>) -> ()
    return %alloc_0 : memref<30522x128xf32>
  }
  func.func private @Unknown14(%arg0: memref<2x128x128xf32>, %arg1: memref<2x128x128xf32>) -> memref<2x128x128xf32> attributes {__byteir_elementwise_fusion__} {
    %alloc = memref.alloc() : memref<2x128x128xf32>
    linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : memref<2x128x128xf32>, memref<2x128x128xf32>) outs(%alloc : memref<2x128x128xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %0 = arith.addf %in, %in_0 : f32
      linalg.yield %0 : f32
    }
    return %alloc : memref<2x128x128xf32>
  }
  func.func private @Unknown15(%arg0: memref<2x128x128xf32>, %arg1: memref<2x128x128xf32>, %arg2: memref<2x128x128xf32>, %arg3: memref<2x128x128xf32>) -> memref<2x128x128xf32> attributes {__byteir_elementwise_fusion__} {
    %alloc = memref.alloc() : memref<2x128x128xf32>
    linalg.generic {indexing_maps = [#map2, #map2, #map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0, %arg1, %arg2, %arg3 : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>) outs(%alloc : memref<2x128x128xf32>) {
    ^bb0(%in: f32, %in_0: f32, %in_1: f32, %in_2: f32, %out: f32):
      %0 = arith.addf %in, %in_0 : f32
      %1 = arith.addf %0, %in_1 : f32
      %2 = arith.addf %1, %in_2 : f32
      linalg.yield %2 : f32
    }
    return %alloc : memref<2x128x128xf32>
  }
  func.func private @Unknown16(%arg0: memref<2x128x128xf32>, %arg1: memref<2x128x128xf32>) -> memref<2x128x128xf32> attributes {__byteir_elementwise_fusion__} {
    %alloc = memref.alloc() : memref<2x128x128xf32>
    linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : memref<2x128x128xf32>, memref<2x128x128xf32>) outs(%alloc : memref<2x128x128xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %0 = arith.addf %in, %in_0 : f32
      linalg.yield %0 : f32
    }
    return %alloc : memref<2x128x128xf32>
  }
  func.func private @Unknown17(%arg0: memref<2x128x128xf32>, %arg1: memref<2x128x128xf32>, %arg2: memref<2x128x128xf32>, %arg3: memref<2x128x128xf32>) -> memref<2x128x128xf32> attributes {__byteir_elementwise_fusion__} {
    %alloc = memref.alloc() : memref<2x128x128xf32>
    linalg.generic {indexing_maps = [#map2, #map2, #map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0, %arg1, %arg2, %arg3 : memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>) outs(%alloc : memref<2x128x128xf32>) {
    ^bb0(%in: f32, %in_0: f32, %in_1: f32, %in_2: f32, %out: f32):
      %0 = arith.addf %in, %in_0 : f32
      %1 = arith.addf %0, %in_1 : f32
      %2 = arith.addf %1, %in_2 : f32
      linalg.yield %2 : f32
    }
    return %alloc : memref<2x128x128xf32>
  }
  func.func private @Unknown18(%arg0: memref<256xi1>, %arg1: memref<2x128x128xf32>, %arg2: memref<256xi1>) -> (memref<256x128xf32>, memref<256x128xf32>) attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f32
    %expand_shape = memref.expand_shape %arg0 [[0, 1]] : memref<256xi1> into memref<2x128xi1>
    %expand_shape_0 = memref.expand_shape %arg2 [[0, 1]] : memref<256xi1> into memref<2x128xi1>
    %alloc = memref.alloc() : memref<2x128x128xf32>
    %alloc_1 = memref.alloc() : memref<2x128x128xf32>
    linalg.generic {indexing_maps = [#map9, #map9, #map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expand_shape_0, %expand_shape, %arg1 : memref<2x128xi1>, memref<2x128xi1>, memref<2x128x128xf32>) outs(%alloc, %alloc_1 : memref<2x128x128xf32>, memref<2x128x128xf32>) {
    ^bb0(%in: i1, %in_3: i1, %in_4: f32, %out: f32, %out_5: f32):
      %0 = arith.select %in_3, %in_4, %cst : f32
      %1 = arith.select %in, %in_4, %cst : f32
      linalg.yield %0, %1 : f32, f32
    }
    %collapse_shape = memref.collapse_shape %alloc_1 [[0, 1], [2]] : memref<2x128x128xf32> into memref<256x128xf32>
    %collapse_shape_2 = memref.collapse_shape %alloc [[0, 1], [2]] : memref<2x128x128xf32> into memref<256x128xf32>
    return %collapse_shape_2, %collapse_shape : memref<256x128xf32>, memref<256x128xf32>
  }
  func.func private @Unknown19(%arg0: memref<128xi1>, %arg1: memref<128x128xf32>) -> memref<128x128xf32> attributes {__byteir_elementwise_fusion__} {
    %cst = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() : memref<128x128xf32>
    linalg.generic {indexing_maps = [#map5, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : memref<128xi1>, memref<128x128xf32>) outs(%alloc : memref<128x128xf32>) {
    ^bb0(%in: i1, %in_0: f32, %out: f32):
      %0 = arith.select %in, %in_0, %cst : f32
      linalg.yield %0 : f32
    }
    return %alloc : memref<128x128xf32>
  }
  func.func @main(%arg0: memref<2x128xi64>, %arg1: memref<2x128xi64>, %arg2: memref<1x512xi64>, %arg3: memref<1x512xi64>, %arg4: memref<30522x128xf32>, %arg5: memref<2x128xf32>, %arg6: memref<512x128xf32>, %arg7: memref<128xf32>, %arg8: memref<128xf32>, %arg9: memref<128x128xf32>, %arg10: memref<128xf32>, %arg11: memref<128x128xf32>, %arg12: memref<128xf32>, %arg13: memref<128x128xf32>, %arg14: memref<128xf32>, %arg15: memref<128x128xf32>, %arg16: memref<128xf32>, %arg17: memref<128xf32>, %arg18: memref<128xf32>, %arg19: memref<512x128xf32>, %arg20: memref<512xf32>, %arg21: memref<128x512xf32>, %arg22: memref<128xf32>, %arg23: memref<128xf32>, %arg24: memref<128xf32>, %arg25: memref<128x128xf32>, %arg26: memref<128xf32>, %arg27: memref<128x128xf32>, %arg28: memref<128xf32>, %arg29: memref<128x128xf32>, %arg30: memref<128xf32>, %arg31: memref<128x128xf32>, %arg32: memref<128xf32>, %arg33: memref<128xf32>, %arg34: memref<128xf32>, %arg35: memref<512x128xf32>, %arg36: memref<512xf32>, %arg37: memref<128x512xf32>, %arg38: memref<128xf32>, %arg39: memref<128xf32>, %arg40: memref<128xf32>, %arg41: memref<128x128xf32>, %arg42: memref<128xf32>, %arg43: memref<128xf32>, %arg44: memref<128xf32>, %arg45: memref<30522xf32>) -> (memref<2x128x30522xf32>, memref<f32>, memref<30522x128xf32>, memref<2x128xf32>, memref<512x128xf32>, memref<128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<512x128xf32>, memref<512xf32>, memref<128x512xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<512x128xf32>, memref<512xf32>, memref<128x512xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<30522xf32>) {
    %alloc = memref.alloc() : memref<512x128xf32>
    "lmhlo.constant"(%alloc) {value = dense<0.000000e+00> : tensor<512x128xf32>} : (memref<512x128xf32>) -> ()
    %alloc_0 = memref.alloc() : memref<2x128xf32>
    "lmhlo.constant"(%alloc_0) {value = dense<0.000000e+00> : tensor<2x128xf32>} : (memref<2x128xf32>) -> ()
    %alloc_1 = memref.alloc() : memref<2x128x128xf32>
    "lmhlo.constant"(%alloc_1) {value = dense<-0.000000e+00> : tensor<2x128x128xf32>} : (memref<2x128x128xf32>) -> ()
    %alloc_2 = memref.alloc() : memref<f32>
    "lmhlo.constant"(%alloc_2) {value = dense<0xFF800000> : tensor<f32>} : (memref<f32>) -> ()
    %alloc_3 = memref.alloc() : memref<f32>
    "lmhlo.constant"(%alloc_3) {value = dense<0.000000e+00> : tensor<f32>} : (memref<f32>) -> ()
    %alloc_4 = memref.alloc() : memref<1x128xi64>
    "lmhlo.slice"(%arg2, %alloc_4) {limit_indices = dense<[1, 128]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (memref<1x512xi64>, memref<1x128xi64>) -> ()
    %alloc_5 = memref.alloc() : memref<128xi64>
    "lmhlo.reshape"(%alloc_4, %alloc_5) : (memref<1x128xi64>, memref<128xi64>) -> ()
    %alloc_6 = memref.alloc() : memref<1x128xi64>
    "lmhlo.slice"(%arg3, %alloc_6) {limit_indices = dense<[1, 128]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (memref<1x512xi64>, memref<1x128xi64>) -> ()
    %0:2 = call @Unknown0(%arg1) : (memref<2x128xi64>) -> (memref<256xi64>, memref<256xi1>)
    %1:3 = call @Unknown1(%arg0) : (memref<2x128xi64>) -> (memref<256xui32>, memref<256x1xi64>, memref<256xi1>)
    %alloc_7 = memref.alloc() : memref<256x128xf32>
    "lmhlo.gather"(%arg4, %1#0, %alloc_7) {dimension_numbers = #mhlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 128]> : tensor<2xi64>} : (memref<30522x128xf32>, memref<256xui32>, memref<256x128xf32>) -> ()
    %2:3 = call @Unknown2(%alloc_5) : (memref<128xi64>) -> (memref<256xui32>, memref<256x1xi64>, memref<256xi1>)
    %alloc_8 = memref.alloc() : memref<256x128xf32>
    "lmhlo.gather"(%arg5, %2#0, %alloc_8) {dimension_numbers = #mhlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 128]> : tensor<2xi64>} : (memref<2x128xf32>, memref<256xui32>, memref<256x128xf32>) -> ()
    %3:3 = call @Unknown3(%alloc_6) : (memref<1x128xi64>) -> (memref<128xui32>, memref<128x1xi64>, memref<128xi1>)
    %alloc_9 = memref.alloc() : memref<128x128xf32>
    "lmhlo.gather"(%arg6, %3#0, %alloc_9) {dimension_numbers = #mhlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 128]> : tensor<2xi64>} : (memref<512x128xf32>, memref<128xui32>, memref<128x128xf32>) -> ()
    %4 = call @Unknown4(%alloc_7, %alloc_8, %alloc_9) : (memref<256x128xf32>, memref<256x128xf32>, memref<128x128xf32>) -> memref<2x128x128xf32>
    %alloc_10 = memref.alloc() : memref<2x128x128xf32>
    %alloc_11 = memref.alloc() : memref<256xf32>
    %alloc_12 = memref.alloc() : memref<256xf32>
    "lmhlo.custom_call"(%4, %arg7, %arg8, %alloc_10, %alloc_11, %alloc_12) {backend_config = "", call_target_name = "ftv4.layernorm", operand_segment_sizes = array<i32: 3, 3>} : (memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>) -> ()
    %alloc_13 = memref.alloc() : memref<2x2x128x64xf32>
    "lmhlo.custom_call"(%alloc_10, %arg9, %arg10, %alloc_13) {backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", operand_segment_sizes = array<i32: 3, 1>} : (memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>) -> ()
    %alloc_14 = memref.alloc() : memref<2x2x128x64xf32>
    "lmhlo.custom_call"(%alloc_10, %arg11, %arg12, %alloc_14) {backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", operand_segment_sizes = array<i32: 3, 1>} : (memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>) -> ()
    %alloc_15 = memref.alloc() : memref<2x2x128x128xf32>
    "lmhlo.custom_call"(%alloc_13, %alloc_14, %alloc_15) {backend_config = "{scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true}", call_target_name = "ftv4.matmul", operand_segment_sizes = array<i32: 2, 1>} : (memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x128xf32>) -> ()
    %alloc_16 = memref.alloc() : memref<2x2x128x128xf32>
    %alloc_17 = memref.alloc() : memref<2x2x128x128xf32>
    %alloc_18 = memref.alloc() : memref<2x2x128x128xui8>
    "lmhlo.custom_call"(%alloc_15, %alloc_1, %alloc_16, %alloc_17, %alloc_18) {backend_config = "{batch_first = true, dropout_rate = 0.000000e+00 : f32, head_num = 2 : i32}", call_target_name = "ftv4.softmax", operand_segment_sizes = array<i32: 2, 3>} : (memref<2x2x128x128xf32>, memref<2x128x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xui8>) -> ()
    %alloc_19 = memref.alloc() : memref<2x2x128x64xf32>
    "lmhlo.custom_call"(%alloc_10, %arg13, %arg14, %alloc_19) {backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", operand_segment_sizes = array<i32: 3, 1>} : (memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>) -> ()
    %alloc_20 = memref.alloc() : memref<2x2x128x64xf32>
    "lmhlo.custom_call"(%alloc_16, %alloc_19, %alloc_20) {backend_config = "{scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false}", call_target_name = "ftv4.matmul", operand_segment_sizes = array<i32: 2, 1>} : (memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>) -> ()
    %alloc_21 = memref.alloc() : memref<2x128x2x64xf32>
    "lmhlo.custom_call"(%alloc_20, %alloc_21) {backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22}", call_target_name = "ftv4.transpose4d", operand_segment_sizes = array<i32: 1, 1>} : (memref<2x2x128x64xf32>, memref<2x128x2x64xf32>) -> ()
    %alloc_22 = memref.alloc() : memref<2x128x128xf32>
    "lmhlo.reshape"(%alloc_21, %alloc_22) : (memref<2x128x2x64xf32>, memref<2x128x128xf32>) -> ()
    %alloc_23 = memref.alloc() : memref<2x128x128xf32>
    "lmhlo.custom_call"(%alloc_22, %arg15, %arg16, %alloc_23) {backend_config = "", call_target_name = "ftv4.linear", operand_segment_sizes = array<i32: 3, 1>} : (memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x128x128xf32>) -> ()
    %alloc_24 = memref.alloc() : memref<2x128x128xf32>
    %alloc_25 = memref.alloc() : memref<256xf32>
    %alloc_26 = memref.alloc() : memref<256xf32>
    %alloc_27 = memref.alloc() : memref<2x128x128xf32>
    "lmhlo.custom_call"(%alloc_23, %arg17, %arg18, %alloc_10, %alloc_24, %alloc_25, %alloc_26, %alloc_27) {backend_config = "", call_target_name = "ftv4.layernorm_residual", operand_segment_sizes = array<i32: 4, 4>} : (memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>) -> ()
    %alloc_28 = memref.alloc() : memref<2x128x512xf32>
    %alloc_29 = memref.alloc() : memref<2x128x512xf32>
    %alloc_30 = memref.alloc() : memref<0xf32>
    "lmhlo.custom_call"(%alloc_24, %arg19, %arg20, %alloc_28, %alloc_29, %alloc_30) {backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout", operand_segment_sizes = array<i32: 3, 3>} : (memref<2x128x128xf32>, memref<512x128xf32>, memref<512xf32>, memref<2x128x512xf32>, memref<2x128x512xf32>, memref<0xf32>) -> ()
    %alloc_31 = memref.alloc() : memref<2x128x128xf32>
    "lmhlo.custom_call"(%alloc_28, %arg21, %arg22, %alloc_31) {backend_config = "", call_target_name = "ftv4.linear", operand_segment_sizes = array<i32: 3, 1>} : (memref<2x128x512xf32>, memref<128x512xf32>, memref<128xf32>, memref<2x128x128xf32>) -> ()
    %alloc_32 = memref.alloc() : memref<2x128x128xf32>
    %alloc_33 = memref.alloc() : memref<256xf32>
    %alloc_34 = memref.alloc() : memref<256xf32>
    %alloc_35 = memref.alloc() : memref<2x128x128xf32>
    "lmhlo.custom_call"(%alloc_31, %arg23, %arg24, %alloc_24, %alloc_32, %alloc_33, %alloc_34, %alloc_35) {backend_config = "", call_target_name = "ftv4.layernorm_residual", operand_segment_sizes = array<i32: 4, 4>} : (memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>) -> ()
    %alloc_36 = memref.alloc() : memref<2x2x128x64xf32>
    "lmhlo.custom_call"(%alloc_32, %arg25, %arg26, %alloc_36) {backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", operand_segment_sizes = array<i32: 3, 1>} : (memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>) -> ()
    %alloc_37 = memref.alloc() : memref<2x2x128x64xf32>
    "lmhlo.custom_call"(%alloc_32, %arg27, %arg28, %alloc_37) {backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", operand_segment_sizes = array<i32: 3, 1>} : (memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>) -> ()
    %alloc_38 = memref.alloc() : memref<2x2x128x128xf32>
    "lmhlo.custom_call"(%alloc_36, %alloc_37, %alloc_38) {backend_config = "{scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true}", call_target_name = "ftv4.matmul", operand_segment_sizes = array<i32: 2, 1>} : (memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x128xf32>) -> ()
    %alloc_39 = memref.alloc() : memref<2x2x128x128xf32>
    %alloc_40 = memref.alloc() : memref<2x2x128x128xf32>
    %alloc_41 = memref.alloc() : memref<2x2x128x128xui8>
    "lmhlo.custom_call"(%alloc_38, %alloc_1, %alloc_39, %alloc_40, %alloc_41) {backend_config = "{batch_first = true, dropout_rate = 0.000000e+00 : f32, head_num = 2 : i32}", call_target_name = "ftv4.softmax", operand_segment_sizes = array<i32: 2, 3>} : (memref<2x2x128x128xf32>, memref<2x128x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xui8>) -> ()
    %alloc_42 = memref.alloc() : memref<2x2x128x64xf32>
    "lmhlo.custom_call"(%alloc_32, %arg29, %arg30, %alloc_42) {backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose", operand_segment_sizes = array<i32: 3, 1>} : (memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x2x128x64xf32>) -> ()
    %alloc_43 = memref.alloc() : memref<2x2x128x64xf32>
    "lmhlo.custom_call"(%alloc_39, %alloc_42, %alloc_43) {backend_config = "{scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false}", call_target_name = "ftv4.matmul", operand_segment_sizes = array<i32: 2, 1>} : (memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>) -> ()
    %alloc_44 = memref.alloc() : memref<2x128x2x64xf32>
    "lmhlo.custom_call"(%alloc_43, %alloc_44) {backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22}", call_target_name = "ftv4.transpose4d", operand_segment_sizes = array<i32: 1, 1>} : (memref<2x2x128x64xf32>, memref<2x128x2x64xf32>) -> ()
    %alloc_45 = memref.alloc() : memref<2x128x128xf32>
    "lmhlo.reshape"(%alloc_44, %alloc_45) : (memref<2x128x2x64xf32>, memref<2x128x128xf32>) -> ()
    %alloc_46 = memref.alloc() : memref<2x128x128xf32>
    "lmhlo.custom_call"(%alloc_45, %arg31, %arg32, %alloc_46) {backend_config = "", call_target_name = "ftv4.linear", operand_segment_sizes = array<i32: 3, 1>} : (memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x128x128xf32>) -> ()
    %alloc_47 = memref.alloc() : memref<2x128x128xf32>
    %alloc_48 = memref.alloc() : memref<256xf32>
    %alloc_49 = memref.alloc() : memref<256xf32>
    %alloc_50 = memref.alloc() : memref<2x128x128xf32>
    "lmhlo.custom_call"(%alloc_46, %arg33, %arg34, %alloc_32, %alloc_47, %alloc_48, %alloc_49, %alloc_50) {backend_config = "", call_target_name = "ftv4.layernorm_residual", operand_segment_sizes = array<i32: 4, 4>} : (memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>) -> ()
    %alloc_51 = memref.alloc() : memref<2x128x512xf32>
    %alloc_52 = memref.alloc() : memref<2x128x512xf32>
    %alloc_53 = memref.alloc() : memref<0xf32>
    "lmhlo.custom_call"(%alloc_47, %arg35, %arg36, %alloc_51, %alloc_52, %alloc_53) {backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout", operand_segment_sizes = array<i32: 3, 3>} : (memref<2x128x128xf32>, memref<512x128xf32>, memref<512xf32>, memref<2x128x512xf32>, memref<2x128x512xf32>, memref<0xf32>) -> ()
    %alloc_54 = memref.alloc() : memref<2x128x128xf32>
    "lmhlo.custom_call"(%alloc_51, %arg37, %arg38, %alloc_54) {backend_config = "", call_target_name = "ftv4.linear", operand_segment_sizes = array<i32: 3, 1>} : (memref<2x128x512xf32>, memref<128x512xf32>, memref<128xf32>, memref<2x128x128xf32>) -> ()
    %alloc_55 = memref.alloc() : memref<2x128x128xf32>
    %alloc_56 = memref.alloc() : memref<256xf32>
    %alloc_57 = memref.alloc() : memref<256xf32>
    %alloc_58 = memref.alloc() : memref<2x128x128xf32>
    "lmhlo.custom_call"(%alloc_54, %arg39, %arg40, %alloc_47, %alloc_55, %alloc_56, %alloc_57, %alloc_58) {backend_config = "", call_target_name = "ftv4.layernorm_residual", operand_segment_sizes = array<i32: 4, 4>} : (memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>) -> ()
    %alloc_59 = memref.alloc() : memref<2x128x128xf32>
    %alloc_60 = memref.alloc() : memref<2x128x128xf32>
    %alloc_61 = memref.alloc() : memref<0xf32>
    "lmhlo.custom_call"(%alloc_55, %arg41, %arg42, %alloc_59, %alloc_60, %alloc_61) {backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout", operand_segment_sizes = array<i32: 3, 3>} : (memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<0xf32>) -> ()
    %alloc_62 = memref.alloc() : memref<2x128x128xf32>
    %alloc_63 = memref.alloc() : memref<256xf32>
    %alloc_64 = memref.alloc() : memref<256xf32>
    "lmhlo.custom_call"(%alloc_59, %arg43, %arg44, %alloc_62, %alloc_63, %alloc_64) {backend_config = "", call_target_name = "ftv4.layernorm", operand_segment_sizes = array<i32: 3, 3>} : (memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<256xf32>, memref<256xf32>) -> ()
    %alloc_65 = memref.alloc() : memref<256x128xf32>
    "lmhlo.reshape"(%alloc_62, %alloc_65) : (memref<2x128x128xf32>, memref<256x128xf32>) -> ()
    %alloc_66 = memref.alloc() : memref<256x30522xf32>
    "lmhlo.dot"(%alloc_65, %arg4, %alloc_66) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [1]>, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<256x128xf32>, memref<30522x128xf32>, memref<256x30522xf32>) -> ()
    %5:2 = call @Unknown5(%alloc_66, %arg45) : (memref<256x30522xf32>, memref<30522xf32>) -> (memref<2x128x30522xf32>, memref<256x30522xf32>)
    %alloc_67 = memref.alloc() : memref<256xf32>
    "lmhlo.reduce"(%5#1, %alloc_2, %alloc_67) ({
    ^bb0(%arg46: memref<f32>, %arg47: memref<f32>, %arg48: memref<f32>):
      "lmhlo.maximum"(%arg46, %arg47, %arg48) : (memref<f32>, memref<f32>, memref<f32>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (memref<256x30522xf32>, memref<f32>, memref<256xf32>) -> ()
    %6:2 = call @Unknown6(%alloc_67, %5#1) : (memref<256xf32>, memref<256x30522xf32>) -> (memref<256x30522xf32>, memref<256x30522xf32>)
    %alloc_68 = memref.alloc() : memref<256xf32>
    "lmhlo.reduce"(%6#1, %alloc_3, %alloc_68) ({
    ^bb0(%arg46: memref<f32>, %arg47: memref<f32>, %arg48: memref<f32>):
      "lmhlo.add"(%arg46, %arg47, %arg48) : (memref<f32>, memref<f32>, memref<f32>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (memref<256x30522xf32>, memref<f32>, memref<256xf32>) -> ()
    %7 = call @Unknown7(%alloc_68) : (memref<256xf32>) -> memref<256xf32>
    %8:4 = call @Unknown8(%7, %6#0, %0#0, %0#1) : (memref<256xf32>, memref<256x30522xf32>, memref<256xi64>, memref<256xi1>) -> (memref<256x30522xf32>, memref<256x30522xf32>, memref<256x30522xf32>, memref<256x30522xf32>)
    %alloc_69 = memref.alloc() : memref<f32>
    "lmhlo.reduce"(%8#1, %alloc_3, %alloc_69) ({
    ^bb0(%arg46: memref<f32>, %arg47: memref<f32>, %arg48: memref<f32>):
      "lmhlo.add"(%arg46, %arg47, %arg48) : (memref<f32>, memref<f32>, memref<f32>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (memref<256x30522xf32>, memref<f32>, memref<f32>) -> ()
    %alloc_70 = memref.alloc() : memref<f32>
    "lmhlo.reduce"(%8#0, %alloc_3, %alloc_70) ({
    ^bb0(%arg46: memref<f32>, %arg47: memref<f32>, %arg48: memref<f32>):
      "lmhlo.add"(%arg46, %arg47, %arg48) : (memref<f32>, memref<f32>, memref<f32>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (memref<256x30522xf32>, memref<f32>, memref<f32>) -> ()
    %9 = call @Unknown9(%alloc_69, %alloc_70) : (memref<f32>, memref<f32>) -> memref<f32>
    %10 = call @Unknown10(%alloc_70) : (memref<f32>) -> memref<f32>
    %11 = call @Unknown11(%10, %8#2) : (memref<f32>, memref<256x30522xf32>) -> memref<256x30522xf32>
    %alloc_71 = memref.alloc() : memref<256xf32>
    "lmhlo.reduce"(%11, %alloc_3, %alloc_71) ({
    ^bb0(%arg46: memref<f32>, %arg47: memref<f32>, %arg48: memref<f32>):
      "lmhlo.add"(%arg46, %arg47, %arg48) : (memref<f32>, memref<f32>, memref<f32>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (memref<256x30522xf32>, memref<f32>, memref<256xf32>) -> ()
    %12:2 = call @Unknown12(%alloc_71, %8#3, %11) : (memref<256xf32>, memref<256x30522xf32>, memref<256x30522xf32>) -> (memref<256x30522xf32>, memref<2x128x30522xf32>)
    %13 = call @MatmulOp13(%alloc_65, %12#0) : (memref<256x128xf32>, memref<256x30522xf32>) -> memref<30522x128xf32>
    %alloc_72 = memref.alloc() : memref<256x128xf32>
    "lmhlo.dot"(%12#0, %arg4, %alloc_72) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (memref<256x30522xf32>, memref<30522x128xf32>, memref<256x128xf32>) -> ()
    %alloc_73 = memref.alloc() : memref<2x128x128xf32>
    "lmhlo.reshape"(%alloc_72, %alloc_73) : (memref<256x128xf32>, memref<2x128x128xf32>) -> ()
    %alloc_74 = memref.alloc() : memref<2x128x128xf32>
    %alloc_75 = memref.alloc() : memref<128xf32>
    %alloc_76 = memref.alloc() : memref<128xf32>
    "lmhlo.custom_call"(%alloc_73, %alloc_59, %arg43, %alloc_63, %alloc_64, %alloc_74, %alloc_75, %alloc_76) {backend_config = "", call_target_name = "ftv4.layernorm_backward", operand_segment_sizes = array<i32: 5, 3>} : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>) -> ()
    %alloc_77 = memref.alloc() : memref<2x128x128xf32>
    %alloc_78 = memref.alloc() : memref<128x128xf32>
    %alloc_79 = memref.alloc() : memref<128xf32>
    "lmhlo.custom_call"(%alloc_74, %alloc_55, %arg41, %alloc_60, %alloc_61, %alloc_77, %alloc_78, %alloc_79) {backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout_backward", operand_segment_sizes = array<i32: 5, 3>} : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<0xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>) -> ()
    %alloc_80 = memref.alloc() : memref<2x128x128xf32>
    %alloc_81 = memref.alloc() : memref<128xf32>
    %alloc_82 = memref.alloc() : memref<128xf32>
    %alloc_83 = memref.alloc() : memref<2x128x128xf32>
    "lmhlo.custom_call"(%alloc_77, %alloc_58, %arg39, %alloc_56, %alloc_57, %alloc_80, %alloc_81, %alloc_82, %alloc_83) {backend_config = "", call_target_name = "ftv4.layernorm_backward_residual", operand_segment_sizes = array<i32: 5, 4>} : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>) -> ()
    %alloc_84 = memref.alloc() : memref<2x128x512xf32>
    %alloc_85 = memref.alloc() : memref<128x512xf32>
    %alloc_86 = memref.alloc() : memref<128xf32>
    "lmhlo.custom_call"(%alloc_80, %alloc_51, %arg37, %alloc_84, %alloc_85, %alloc_86) {backend_config = "", call_target_name = "ftv4.linear_backward", operand_segment_sizes = array<i32: 3, 3>} : (memref<2x128x128xf32>, memref<2x128x512xf32>, memref<128x512xf32>, memref<2x128x512xf32>, memref<128x512xf32>, memref<128xf32>) -> ()
    %alloc_87 = memref.alloc() : memref<2x128x128xf32>
    %alloc_88 = memref.alloc() : memref<512x128xf32>
    %alloc_89 = memref.alloc() : memref<512xf32>
    "lmhlo.custom_call"(%alloc_84, %alloc_47, %arg35, %alloc_52, %alloc_53, %alloc_87, %alloc_88, %alloc_89) {backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout_backward", operand_segment_sizes = array<i32: 5, 3>} : (memref<2x128x512xf32>, memref<2x128x128xf32>, memref<512x128xf32>, memref<2x128x512xf32>, memref<0xf32>, memref<2x128x128xf32>, memref<512x128xf32>, memref<512xf32>) -> ()
    %14 = call @Unknown14(%alloc_83, %alloc_87) : (memref<2x128x128xf32>, memref<2x128x128xf32>) -> memref<2x128x128xf32>
    %alloc_90 = memref.alloc() : memref<2x128x128xf32>
    %alloc_91 = memref.alloc() : memref<128xf32>
    %alloc_92 = memref.alloc() : memref<128xf32>
    %alloc_93 = memref.alloc() : memref<2x128x128xf32>
    "lmhlo.custom_call"(%14, %alloc_50, %arg33, %alloc_48, %alloc_49, %alloc_90, %alloc_91, %alloc_92, %alloc_93) {backend_config = "", call_target_name = "ftv4.layernorm_backward_residual", operand_segment_sizes = array<i32: 5, 4>} : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>) -> ()
    %alloc_94 = memref.alloc() : memref<2x128x128xf32>
    %alloc_95 = memref.alloc() : memref<128x128xf32>
    %alloc_96 = memref.alloc() : memref<128xf32>
    "lmhlo.custom_call"(%alloc_90, %alloc_45, %arg31, %alloc_94, %alloc_95, %alloc_96) {backend_config = "", call_target_name = "ftv4.linear_backward", operand_segment_sizes = array<i32: 3, 3>} : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>) -> ()
    %alloc_97 = memref.alloc() : memref<2x128x2x64xf32>
    "lmhlo.reshape"(%alloc_94, %alloc_97) : (memref<2x128x128xf32>, memref<2x128x2x64xf32>) -> ()
    %alloc_98 = memref.alloc() : memref<2x2x128x64xf32>
    "lmhlo.custom_call"(%alloc_97, %alloc_98) {backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22}", call_target_name = "ftv4.transpose4d_backward", operand_segment_sizes = array<i32: 1, 1>} : (memref<2x128x2x64xf32>, memref<2x2x128x64xf32>) -> ()
    %alloc_99 = memref.alloc() : memref<2x2x128x128xf32>
    %alloc_100 = memref.alloc() : memref<2x2x128x64xf32>
    "lmhlo.custom_call"(%alloc_98, %alloc_39, %alloc_42, %alloc_99, %alloc_100) {backend_config = "{scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false}", call_target_name = "ftv4.matmul_backward", operand_segment_sizes = array<i32: 3, 2>} : (memref<2x2x128x64xf32>, memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x128xf32>, memref<2x2x128x64xf32>) -> ()
    %alloc_101 = memref.alloc() : memref<2x2x128x128xf32>
    "lmhlo.custom_call"(%alloc_99, %alloc_39, %alloc_41, %alloc_101) {backend_config = "{dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.softmax_backward", operand_segment_sizes = array<i32: 3, 1>} : (memref<2x2x128x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xui8>, memref<2x2x128x128xf32>) -> ()
    %alloc_102 = memref.alloc() : memref<2x2x128x64xf32>
    %alloc_103 = memref.alloc() : memref<2x2x128x64xf32>
    "lmhlo.custom_call"(%alloc_101, %alloc_36, %alloc_37, %alloc_102, %alloc_103) {backend_config = "{scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true}", call_target_name = "ftv4.matmul_backward", operand_segment_sizes = array<i32: 3, 2>} : (memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>) -> ()
    %alloc_104 = memref.alloc() : memref<2x128x128xf32>
    %alloc_105 = memref.alloc() : memref<128x128xf32>
    %alloc_106 = memref.alloc() : memref<128xf32>
    "lmhlo.custom_call"(%alloc_102, %alloc_32, %arg25, %alloc_104, %alloc_105, %alloc_106) {backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", operand_segment_sizes = array<i32: 3, 3>} : (memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>) -> ()
    %alloc_107 = memref.alloc() : memref<2x128x128xf32>
    %alloc_108 = memref.alloc() : memref<128x128xf32>
    %alloc_109 = memref.alloc() : memref<128xf32>
    "lmhlo.custom_call"(%alloc_100, %alloc_32, %arg29, %alloc_107, %alloc_108, %alloc_109) {backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", operand_segment_sizes = array<i32: 3, 3>} : (memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>) -> ()
    %alloc_110 = memref.alloc() : memref<2x128x128xf32>
    %alloc_111 = memref.alloc() : memref<128x128xf32>
    %alloc_112 = memref.alloc() : memref<128xf32>
    "lmhlo.custom_call"(%alloc_103, %alloc_32, %arg27, %alloc_110, %alloc_111, %alloc_112) {backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", operand_segment_sizes = array<i32: 3, 3>} : (memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>) -> ()
    %15 = call @Unknown15(%alloc_93, %alloc_104, %alloc_107, %alloc_110) : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>) -> memref<2x128x128xf32>
    %alloc_113 = memref.alloc() : memref<2x128x128xf32>
    %alloc_114 = memref.alloc() : memref<128xf32>
    %alloc_115 = memref.alloc() : memref<128xf32>
    %alloc_116 = memref.alloc() : memref<2x128x128xf32>
    "lmhlo.custom_call"(%15, %alloc_35, %arg23, %alloc_33, %alloc_34, %alloc_113, %alloc_114, %alloc_115, %alloc_116) {backend_config = "", call_target_name = "ftv4.layernorm_backward_residual", operand_segment_sizes = array<i32: 5, 4>} : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>) -> ()
    %alloc_117 = memref.alloc() : memref<2x128x512xf32>
    %alloc_118 = memref.alloc() : memref<128x512xf32>
    %alloc_119 = memref.alloc() : memref<128xf32>
    "lmhlo.custom_call"(%alloc_113, %alloc_28, %arg21, %alloc_117, %alloc_118, %alloc_119) {backend_config = "", call_target_name = "ftv4.linear_backward", operand_segment_sizes = array<i32: 3, 3>} : (memref<2x128x128xf32>, memref<2x128x512xf32>, memref<128x512xf32>, memref<2x128x512xf32>, memref<128x512xf32>, memref<128xf32>) -> ()
    %alloc_120 = memref.alloc() : memref<2x128x128xf32>
    %alloc_121 = memref.alloc() : memref<512x128xf32>
    %alloc_122 = memref.alloc() : memref<512xf32>
    "lmhlo.custom_call"(%alloc_117, %alloc_24, %arg19, %alloc_29, %alloc_30, %alloc_120, %alloc_121, %alloc_122) {backend_config = "{act_gelu = true, dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.linear_gelu_dropout_backward", operand_segment_sizes = array<i32: 5, 3>} : (memref<2x128x512xf32>, memref<2x128x128xf32>, memref<512x128xf32>, memref<2x128x512xf32>, memref<0xf32>, memref<2x128x128xf32>, memref<512x128xf32>, memref<512xf32>) -> ()
    %16 = call @Unknown16(%alloc_116, %alloc_120) : (memref<2x128x128xf32>, memref<2x128x128xf32>) -> memref<2x128x128xf32>
    %alloc_123 = memref.alloc() : memref<2x128x128xf32>
    %alloc_124 = memref.alloc() : memref<128xf32>
    %alloc_125 = memref.alloc() : memref<128xf32>
    %alloc_126 = memref.alloc() : memref<2x128x128xf32>
    "lmhlo.custom_call"(%16, %alloc_27, %arg17, %alloc_25, %alloc_26, %alloc_123, %alloc_124, %alloc_125, %alloc_126) {backend_config = "", call_target_name = "ftv4.layernorm_backward_residual", operand_segment_sizes = array<i32: 5, 4>} : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>) -> ()
    %alloc_127 = memref.alloc() : memref<2x128x128xf32>
    %alloc_128 = memref.alloc() : memref<128x128xf32>
    %alloc_129 = memref.alloc() : memref<128xf32>
    "lmhlo.custom_call"(%alloc_123, %alloc_22, %arg15, %alloc_127, %alloc_128, %alloc_129) {backend_config = "", call_target_name = "ftv4.linear_backward", operand_segment_sizes = array<i32: 3, 3>} : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>) -> ()
    %alloc_130 = memref.alloc() : memref<2x128x2x64xf32>
    "lmhlo.reshape"(%alloc_127, %alloc_130) : (memref<2x128x128xf32>, memref<2x128x2x64xf32>) -> ()
    %alloc_131 = memref.alloc() : memref<2x2x128x64xf32>
    "lmhlo.custom_call"(%alloc_130, %alloc_131) {backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22}", call_target_name = "ftv4.transpose4d_backward", operand_segment_sizes = array<i32: 1, 1>} : (memref<2x128x2x64xf32>, memref<2x2x128x64xf32>) -> ()
    %alloc_132 = memref.alloc() : memref<2x2x128x128xf32>
    %alloc_133 = memref.alloc() : memref<2x2x128x64xf32>
    "lmhlo.custom_call"(%alloc_131, %alloc_16, %alloc_19, %alloc_132, %alloc_133) {backend_config = "{scale = 1.000000e+00 : f32, transpose_a = false, transpose_b = false}", call_target_name = "ftv4.matmul_backward", operand_segment_sizes = array<i32: 3, 2>} : (memref<2x2x128x64xf32>, memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x128xf32>, memref<2x2x128x64xf32>) -> ()
    %alloc_134 = memref.alloc() : memref<2x2x128x128xf32>
    "lmhlo.custom_call"(%alloc_132, %alloc_16, %alloc_18, %alloc_134) {backend_config = "{dropout_rate = 0.000000e+00 : f32}", call_target_name = "ftv4.softmax_backward", operand_segment_sizes = array<i32: 3, 1>} : (memref<2x2x128x128xf32>, memref<2x2x128x128xf32>, memref<2x2x128x128xui8>, memref<2x2x128x128xf32>) -> ()
    %alloc_135 = memref.alloc() : memref<2x2x128x64xf32>
    %alloc_136 = memref.alloc() : memref<2x2x128x64xf32>
    "lmhlo.custom_call"(%alloc_134, %alloc_13, %alloc_14, %alloc_135, %alloc_136) {backend_config = "{scale = 1.250000e-01 : f32, transpose_a = false, transpose_b = true}", call_target_name = "ftv4.matmul_backward", operand_segment_sizes = array<i32: 3, 2>} : (memref<2x2x128x128xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>, memref<2x2x128x64xf32>) -> ()
    %alloc_137 = memref.alloc() : memref<2x128x128xf32>
    %alloc_138 = memref.alloc() : memref<128x128xf32>
    %alloc_139 = memref.alloc() : memref<128xf32>
    "lmhlo.custom_call"(%alloc_135, %alloc_10, %arg9, %alloc_137, %alloc_138, %alloc_139) {backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", operand_segment_sizes = array<i32: 3, 3>} : (memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>) -> ()
    %alloc_140 = memref.alloc() : memref<2x128x128xf32>
    %alloc_141 = memref.alloc() : memref<128x128xf32>
    %alloc_142 = memref.alloc() : memref<128xf32>
    "lmhlo.custom_call"(%alloc_133, %alloc_10, %arg13, %alloc_140, %alloc_141, %alloc_142) {backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", operand_segment_sizes = array<i32: 3, 3>} : (memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>) -> ()
    %alloc_143 = memref.alloc() : memref<2x128x128xf32>
    %alloc_144 = memref.alloc() : memref<128x128xf32>
    %alloc_145 = memref.alloc() : memref<128xf32>
    "lmhlo.custom_call"(%alloc_136, %alloc_10, %arg11, %alloc_143, %alloc_144, %alloc_145) {backend_config = "{forward_transpose_type = \22TRANSPOSE0213\22, head_num = 2 : i32}", call_target_name = "ftv4.linear_transpose_backward", operand_segment_sizes = array<i32: 3, 3>} : (memref<2x2x128x64xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<2x128x128xf32>, memref<128x128xf32>, memref<128xf32>) -> ()
    %17 = call @Unknown17(%alloc_126, %alloc_137, %alloc_140, %alloc_143) : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>) -> memref<2x128x128xf32>
    %alloc_146 = memref.alloc() : memref<2x128x128xf32>
    %alloc_147 = memref.alloc() : memref<128xf32>
    %alloc_148 = memref.alloc() : memref<128xf32>
    "lmhlo.custom_call"(%17, %4, %arg7, %alloc_11, %alloc_12, %alloc_146, %alloc_147, %alloc_148) {backend_config = "", call_target_name = "ftv4.layernorm_backward", operand_segment_sizes = array<i32: 5, 3>} : (memref<2x128x128xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<256xf32>, memref<256xf32>, memref<2x128x128xf32>, memref<128xf32>, memref<128xf32>) -> ()
    %18:2 = call @Unknown18(%1#2, %alloc_146, %2#2) : (memref<256xi1>, memref<2x128x128xf32>, memref<256xi1>) -> (memref<256x128xf32>, memref<256x128xf32>)
    %alloc_149 = memref.alloc() : memref<30522x128xf32>
    "lmhlo.scatter"(%13, %1#1, %18#0, %alloc_149) ({
    ^bb0(%arg46: tensor<f32>, %arg47: tensor<f32>):
      %20 = mhlo.add %arg46, %arg47 : tensor<f32>
      mhlo.return %20 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (memref<30522x128xf32>, memref<256x1xi64>, memref<256x128xf32>, memref<30522x128xf32>) -> ()
    %alloc_150 = memref.alloc() : memref<2x128xf32>
    "lmhlo.scatter"(%alloc_0, %2#1, %18#1, %alloc_150) ({
    ^bb0(%arg46: tensor<f32>, %arg47: tensor<f32>):
      %20 = mhlo.add %arg46, %arg47 : tensor<f32>
      mhlo.return %20 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (memref<2x128xf32>, memref<256x1xi64>, memref<256x128xf32>, memref<2x128xf32>) -> ()
    %alloc_151 = memref.alloc() : memref<128x128xf32>
    "lmhlo.reduce"(%alloc_146, %alloc_3, %alloc_151) ({
    ^bb0(%arg46: memref<f32>, %arg47: memref<f32>, %arg48: memref<f32>):
      "lmhlo.add"(%arg46, %arg47, %arg48) : (memref<f32>, memref<f32>, memref<f32>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (memref<2x128x128xf32>, memref<f32>, memref<128x128xf32>) -> ()
    %19 = call @Unknown19(%3#2, %alloc_151) : (memref<128xi1>, memref<128x128xf32>) -> memref<128x128xf32>
    %alloc_152 = memref.alloc() : memref<512x128xf32>
    "lmhlo.scatter"(%alloc, %3#1, %19, %alloc_152) ({
    ^bb0(%arg46: tensor<f32>, %arg47: tensor<f32>):
      %20 = mhlo.add %arg46, %arg47 : tensor<f32>
      mhlo.return %20 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (memref<512x128xf32>, memref<128x1xi64>, memref<128x128xf32>, memref<512x128xf32>) -> ()
    %alloc_153 = memref.alloc() : memref<30522xf32>
    "lmhlo.reduce"(%12#1, %alloc_3, %alloc_153) ({
    ^bb0(%arg46: memref<f32>, %arg47: memref<f32>, %arg48: memref<f32>):
      "lmhlo.add"(%arg46, %arg47, %arg48) : (memref<f32>, memref<f32>, memref<f32>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (memref<2x128x30522xf32>, memref<f32>, memref<30522xf32>) -> ()
    return %5#0, %9, %alloc_149, %alloc_150, %alloc_152, %alloc_147, %alloc_148, %alloc_138, %alloc_139, %alloc_144, %alloc_145, %alloc_141, %alloc_142, %alloc_128, %alloc_129, %alloc_124, %alloc_125, %alloc_121, %alloc_122, %alloc_118, %alloc_119, %alloc_114, %alloc_115, %alloc_105, %alloc_106, %alloc_111, %alloc_112, %alloc_108, %alloc_109, %alloc_95, %alloc_96, %alloc_91, %alloc_92, %alloc_88, %alloc_89, %alloc_85, %alloc_86, %alloc_81, %alloc_82, %alloc_78, %alloc_79, %alloc_75, %alloc_76, %alloc_153 : memref<2x128x30522xf32>, memref<f32>, memref<30522x128xf32>, memref<2x128xf32>, memref<512x128xf32>, memref<128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<512x128xf32>, memref<512xf32>, memref<128x512xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<512x128xf32>, memref<512xf32>, memref<128x512xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<128x128xf32>, memref<128xf32>, memref<128xf32>, memref<128xf32>, memref<30522xf32>
  }
}

