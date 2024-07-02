// RUN: byteir-opt %s -shape-opt --dynamic-shape-clustering | FileCheck %s

func.func @dynamic_partition_and_stitch(%arg0: tensor<4x4xf32>, %arg1: tensor<4xi32>) -> tensor<?x4xf32> {
  %0 = shape.const_shape [4] : tensor<1xindex>
  %1 = mhlo.constant dense<[[-0.705530286, 0.87041223, 0.972314774, -0.0584422052], [-1.43617868, 6.772900e-01, 0.880922436, 0.56821847], [0.57929492, 0.470399499, -1.0485183, -1.27004325], [-0.32425791, 1.88410747, 0.220974803, -0.238485783]]> : tensor<4x4xf32>
  %2 = mhlo.constant dense<[0.553816557, -0.920699775, 0.418103188, -0.261674613]> : tensor<4xf32>
  %3 = mhlo.constant dense<[[-0.916170597, -0.884184718, 1.60242105, -1.19678485], [0.33643803, -0.431175768, 1.71861267, 0.126368985], [-1.07191086, -1.00517535, -0.666032254, 0.776807785], [1.53380013, 0.83925873, -0.24277249, 1.53341103]]> : tensor<4x4xf32>
  %4 = mhlo.constant dense<[2.63629675, 2.68127704, 2.14741468, -1.6519475]> : tensor<4xf32>
  %5 = mhlo.constant dense<[[0.473984271, 0.173930168, 0.465745121, 1.14254773], [-0.384602815, -0.673360229, 1.13109767, 0.761463344], [-0.171464354, -0.908823907, 1.19337058, -1.78143835], [1.40376866, -0.529214859, -1.9030931, 1.25083804]]> : tensor<4x4xf32>
  %6 = mhlo.constant dense<[0.478572756, 0.458867788, -1.44476604, 0.189240679]> : tensor<4xf32>
  %7 = mhlo.constant dense<[[-1.87686706, 0.286330104, -0.044809185, -0.178677231], [-1.14233077, -0.446333855, -1.2957921, 0.446576297], [0.985618114, 0.699275255, 0.609199941, -0.726590812], [0.0366623849, -0.640842735, -1.72003555, -0.383472085]]> : tensor<4x4xf32>
  %8 = mhlo.constant dense<[1.56364501, -0.948736965, 0.0843383893, 0.502355933]> : tensor<4xf32>
  %9 = mhlo.constant dense<[0, 1, 2, 3]> : tensor<4xi32>
  %10 = "mhlo.dot"(%arg0, %1) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  %11 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<4xf32>) -> tensor<4x4xf32>
  %12 = mhlo.add %10, %11 : tensor<4x4xf32>
  %13:2 = "mhlo.custom_call"(%12, %arg1) {byteir_attrs = {num_partitions = 2 : i64}, call_target_name = "tf.DynamicPartition", has_side_effect = false} : (tensor<4x4xf32>, tensor<4xi32>) -> (tensor<?x4xf32>, tensor<?x4xf32>)
  %14 = "mhlo.dot"(%13#0, %3) : (tensor<?x4xf32>, tensor<4x4xf32>) -> tensor<?x4xf32>
  %15 = shape.shape_of %14 : tensor<?x4xf32> -> tensor<2xindex>
  %18 = "mhlo.dynamic_broadcast_in_dim"(%4, %15) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<4xf32>, tensor<2xindex>) -> tensor<?x4xf32>
  %19 = mhlo.add %14, %18 : tensor<?x4xf32>
  %20 = "mhlo.dot"(%13#1, %5) : (tensor<?x4xf32>, tensor<4x4xf32>) -> tensor<?x4xf32>
  %21 = shape.shape_of %20 : tensor<?x4xf32> -> tensor<2xindex>
  %24 = "mhlo.dynamic_broadcast_in_dim"(%6, %21) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<4xf32>, tensor<2xindex>) -> tensor<?x4xf32>
  %25 = mhlo.add %20, %24 : tensor<?x4xf32>
  %26:2 = "mhlo.custom_call"(%9, %arg1) {byteir_attrs = {num_partitions = 2 : i64}, call_target_name = "tf.DynamicPartition", has_side_effect = false} : (tensor<4xi32>, tensor<4xi32>) -> (tensor<?xi32>, tensor<?xi32>)
  %27 = "mhlo.custom_call"(%26#0, %26#1, %19, %25) {call_target_name = "tf.DynamicStitch", has_side_effect = false} : (tensor<?xi32>, tensor<?xi32>, tensor<?x4xf32>, tensor<?x4xf32>) -> tensor<?x4xf32>
  %28 = "mhlo.dot"(%27, %7) : (tensor<?x4xf32>, tensor<4x4xf32>) -> tensor<?x4xf32>
  %29 = shape.shape_of %28 : tensor<?x4xf32> -> tensor<2xindex>
  %32 = "mhlo.dynamic_broadcast_in_dim"(%8, %29) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<4xf32>, tensor<2xindex>) -> tensor<?x4xf32>
  %33 = mhlo.add %28, %32 : tensor<?x4xf32>
  return %33 : tensor<?x4xf32>
}
// CHECK-LABEL: @dynamic_partition_and_stitch
// CHECK: call @dynamic_partition_and_stitch_sub_1({{.*}}) : (tensor<?x4xf32, {byteir.bounded_shape = [4, 4]}>) -> tensor<?x4xf32, {byteir.bounded_shape = [4, 4]}>
// CHECK: call @dynamic_partition_and_stitch_sub_2({{.*}}) : (tensor<?x4xf32, {byteir.bounded_shape = [4, 4]}>) -> tensor<?x4xf32, {byteir.bounded_shape = [4, 4]}>
// CHECK: call @dynamic_partition_and_stitch_sub_0({{.*}}) : (tensor<?xi32, {byteir.bounded_shape = [4]}>, tensor<?xi32, {byteir.bounded_shape = [4]}>, tensor<?x4xf32, {byteir.bounded_shape = [4, 4]}>, tensor<?x4xf32, {byteir.bounded_shape = [4, 4]}>) -> tensor<4x4xf32>

// CHECK-LABEL:  func.func @dynamic_partition_and_stitch_sub_2(%arg0: tensor<?x4xf32, {byteir.bounded_shape = [4, 4]}>) -> tensor<?x4xf32, {byteir.bounded_shape = [4, 4]}> attributes {__byteir_dynamic_sub_function} {
// CHECK-DAG:     %[[V0:.*]] = mhlo.constant {{.*}} : tensor<4x4xf32>
// CHECK-DAG:     %[[V1:.*]] = mhlo.constant dense<[0.478572756, 0.458867788, -1.44476604, 0.189240679]> : tensor<4xf32>
// CHECK-DAG:     %[[V2:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[V3:.*]] = arith.constant 4 : index
// CHECK-DAG:     %[[V4:.*]] = tensor.dim %arg0, %[[V2]] : tensor<?x4xf32, {byteir.bounded_shape = [4, 4]}>
// CHECK-DAG:     %[[V5:.*]] = "mhlo.dot"(%arg0, %[[V0]]) : (tensor<?x4xf32, {byteir.bounded_shape = [4, 4]}>, tensor<4x4xf32>) -> tensor<?x4xf32, {byteir.bounded_shape = [4, 4]}>
// CHECK-DAG:     %[[V6:.*]] = tensor.from_elements %[[V4]], %[[V3]] : tensor<2xindex>
// CHECK-DAG:     %[[V7:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[V1]], %[[V6]]) <{broadcast_dimensions = dense<1> : tensor<1xi64>}> : (tensor<4xf32>, tensor<2xindex>) -> tensor<?x4xf32, {byteir.bounded_shape = [4, 4]}>
// CHECK-DAG:     %[[V8:.*]] = mhlo.add %[[V5]], %[[V7]] : tensor<?x4xf32, {byteir.bounded_shape = [4, 4]}>
// CHECK-DAG:    return %[[V8]] : tensor<?x4xf32, {byteir.bounded_shape = [4, 4]}>

// CHECK-LABEL:  func.func @dynamic_partition_and_stitch_sub_1(%arg0: tensor<?x4xf32, {byteir.bounded_shape = [4, 4]}>) -> tensor<?x4xf32, {byteir.bounded_shape = [4, 4]}> attributes {__byteir_dynamic_sub_function} {
// CHECK-DAG:     %[[V0:.*]] = mhlo.constant {{.*}} : tensor<4x4xf32>
// CHECK-DAG:     %[[V1:.*]] = mhlo.constant dense<[2.63629675, 2.68127704, 2.14741468, -1.6519475]> : tensor<4xf32>
// CHECK-DAG:     %[[V2:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[V3:.*]] = arith.constant 4 : index
// CHECK-DAG:     %[[V4:.*]] = tensor.dim %arg0, %[[V2]] : tensor<?x4xf32, {byteir.bounded_shape = [4, 4]}>
// CHECK-DAG:     %[[V5:.*]] = "mhlo.dot"(%arg0, %[[V0]]) : (tensor<?x4xf32, {byteir.bounded_shape = [4, 4]}>, tensor<4x4xf32>) -> tensor<?x4xf32, {byteir.bounded_shape = [4, 4]}>
// CHECK-DAG:     %[[V6:.*]] = tensor.from_elements %[[V4]], %[[V3]] : tensor<2xindex>
// CHECK-DAG:     %[[V7:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[V1]], %[[V6]]) <{broadcast_dimensions = dense<1> : tensor<1xi64>}> : (tensor<4xf32>, tensor<2xindex>) -> tensor<?x4xf32, {byteir.bounded_shape = [4, 4]}>
// CHECK-DAG:     %[[V8:.*]] = mhlo.add %[[V5]], %[[V7]] : tensor<?x4xf32, {byteir.bounded_shape = [4, 4]}>
// CHECK-DAG:     return %[[V8]] : tensor<?x4xf32, {byteir.bounded_shape = [4, 4]}>

// CHECK-LABEL: func.func @dynamic_partition_and_stitch_sub_0(%arg0: tensor<?xi32, {byteir.bounded_shape = [4]}>, %arg1: tensor<?xi32, {byteir.bounded_shape = [4]}>, %arg2: tensor<?x4xf32, {byteir.bounded_shape = [4, 4]}>, %arg3: tensor<?x4xf32, {byteir.bounded_shape = [4, 4]}>) -> tensor<4x4xf32> attributes {__byteir_dynamic_sub_function} {
// CHECK-NEXT:   %0 = mhlo.custom_call @tf.DynamicStitch(%arg0, %arg1, %arg2, %arg3) : (tensor<?xi32, {byteir.bounded_shape = [4]}>, tensor<?xi32, {byteir.bounded_shape = [4]}>, tensor<?x4xf32, {byteir.bounded_shape = [4, 4]}>, tensor<?x4xf32, {byteir.bounded_shape = [4, 4]}>) -> tensor<4x4xf32>
// CHECK-NEXT:   return %0 : tensor<4x4xf32>
// CHECK-NEXT: }


func.func @dynamic_partition_and_mask_stitch(%arg0: tensor<4x4xf32>, %arg1: tensor<4xi32>) -> tensor<?x4xf32> {
  %0 = shape.const_shape [4] : tensor<1xindex>
  %1 = mhlo.constant dense<[[-1.3570565, -1.31949258, 6.790670e-01, 0.936488509], [-0.182958096, 0.316772372, -0.942083597, -0.00917604845], [-0.692627251, 0.642019153, 0.296407491, 0.514329314], [0.207706019, 0.15501003, -1.54203725, 0.680027544]]> : tensor<4x4xf32>
  %2 = mhlo.constant dense<[0.0650718957, -1.22261572, 0.241638973, -2.67625213]> : tensor<4xf32>
  %3 = mhlo.constant dense<[[-0.570340514, 0.117151208, -0.135694504, -1.57919896], [0.520053327, 0.762166619, 0.322875232, -1.69871449], [-1.26622009, 0.63558042, 5.698780e-01, 0.954656243], [0.776482939, 0.348752886, 2.03235912, 0.837243676]]> : tensor<4x4xf32>
  %4 = mhlo.constant dense<[0.0399630852, -1.36038888, -0.804892302, -0.616697788]> : tensor<4xf32>
  %5 = mhlo.constant dense<[[-0.676586568, -0.905907988, -0.766016423, 0.0220800452], [0.143005088, 1.34524262, -5.903690e-01, -1.23822224], [-1.51192784, -1.63904226, -1.10326695, -1.37864411], [2.1795454, 2.26776671, -0.449136406, -0.12559399]]> : tensor<4x4xf32>
  %6 = mhlo.constant dense<[-0.0970678701, 0.326490194, 0.730893254, -1.14204574]> : tensor<4xf32>
  %7 = mhlo.constant dense<[[-1.8936193, 0.430917233, -2.53207135, 0.347485185], [-1.5415138, -1.08674419, 0.828085601, 0.260659158], [-2.31310606, 0.937817812, -1.20469058, -0.882931053], [-0.742933631, -0.802833378, 0.0828524753, -0.348286659]]> : tensor<4x4xf32>
  %8 = mhlo.constant dense<[1.55462444, -0.558249593, 0.921237349, 2.072270e+00]> : tensor<4xf32>
  %9 = "mhlo.dot"(%arg0, %1) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  %10 = "mhlo.broadcast_in_dim"(%2) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<4xf32>) -> tensor<4x4xf32>
  %11 = mhlo.add %9, %10 : tensor<4x4xf32>
  %12:2 = "mhlo.custom_call"(%11, %arg1) {byteir_attrs = {num_partitions = 2 : i64}, call_target_name = "tf.DynamicPartition", has_side_effect = false} : (tensor<4x4xf32>, tensor<4xi32>) -> (tensor<?x4xf32>, tensor<?x4xf32>)
  %13 = "mhlo.dot"(%12#0, %3) : (tensor<?x4xf32>, tensor<4x4xf32>) -> tensor<?x4xf32>
  %14 = shape.shape_of %13 : tensor<?x4xf32> -> tensor<2xindex>
  %15 = shape.broadcast %14, %0 : tensor<2xindex>, tensor<1xindex> -> tensor<2xindex>
  %16 = "mhlo.dynamic_broadcast_in_dim"(%13, %15) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<?x4xf32>, tensor<2xindex>) -> tensor<?x4xf32>
  %17 = "mhlo.dynamic_broadcast_in_dim"(%4, %15) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<4xf32>, tensor<2xindex>) -> tensor<?x4xf32>
  %18 = mhlo.add %16, %17 : tensor<?x4xf32>
  %19 = "mhlo.dot"(%12#1, %5) : (tensor<?x4xf32>, tensor<4x4xf32>) -> tensor<?x4xf32>
  %20 = shape.shape_of %19 : tensor<?x4xf32> -> tensor<2xindex>
  %21 = shape.broadcast %20, %0 : tensor<2xindex>, tensor<1xindex> -> tensor<2xindex>
  %22 = "mhlo.dynamic_broadcast_in_dim"(%19, %21) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<?x4xf32>, tensor<2xindex>) -> tensor<?x4xf32>
  %23 = "mhlo.dynamic_broadcast_in_dim"(%6, %21) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<4xf32>, tensor<2xindex>) -> tensor<?x4xf32>
  %24 = mhlo.add %22, %23 : tensor<?x4xf32>
  %25 = "mhlo.custom_call"(%18, %24, %arg1) {api_version = 1 : i32, backend_config = "", call_target_name = "tf.DynamicMaskStitch", called_computations = [], has_side_effect = false} : (tensor<?x4xf32>, tensor<?x4xf32>, tensor<4xi32>) -> tensor<?x4xf32>
  %26 = "mhlo.dot"(%25, %7) : (tensor<?x4xf32>, tensor<4x4xf32>) -> tensor<?x4xf32>
  %27 = shape.shape_of %26 : tensor<?x4xf32> -> tensor<2xindex>
  %28 = shape.broadcast %27, %0 : tensor<2xindex>, tensor<1xindex> -> tensor<2xindex>
  %29 = "mhlo.dynamic_broadcast_in_dim"(%26, %28) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<?x4xf32>, tensor<2xindex>) -> tensor<?x4xf32>
  %30 = "mhlo.dynamic_broadcast_in_dim"(%8, %28) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<4xf32>, tensor<2xindex>) -> tensor<?x4xf32>
  %31 = mhlo.add %29, %30 : tensor<?x4xf32>
  return %31 : tensor<?x4xf32>
}

// CHECK-LABEL: func.func @dynamic_partition_and_mask_stitch
// CHECK: call @dynamic_partition_and_mask_stitch_sub_1({{.*}}) : (tensor<?x4xf32, {byteir.bounded_shape = [4, 4]}>) -> tensor<?x4xf32, {byteir.bounded_shape = [4, 4]}>
// CHECK: call @dynamic_partition_and_mask_stitch_sub_2({{.*}}) : (tensor<?x4xf32, {byteir.bounded_shape = [4, 4]}>) -> tensor<?x4xf32, {byteir.bounded_shape = [4, 4]}>
// CHECK: call @dynamic_partition_and_mask_stitch_sub_0({{.*}}) : (tensor<?x4xf32, {byteir.bounded_shape = [4, 4]}>, tensor<?x4xf32, {byteir.bounded_shape = [4, 4]}>, tensor<4xi32>) -> tensor<4x4xf32>

// CHECK-LABEL: func.func @dynamic_partition_and_mask_stitch_sub_2(%arg0: tensor<?x4xf32, {byteir.bounded_shape = [4, 4]}>) -> tensor<?x4xf32, {byteir.bounded_shape = [4, 4]}> attributes {__byteir_dynamic_sub_function} {
// CHECK-DAG:   %[[V0:.*]] = mhlo.constant dense<[-0.0970678701, 0.326490194, 0.730893254, -1.14204574]> : tensor<4xf32>
// CHECK-DAG:   %[[V1:.*]] = mhlo.constant {{.*}} : tensor<4x4xf32>
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C4:.*]] = arith.constant 4 : index
// CHECK-DAG:   %[[D0:.*]] = tensor.dim %arg0, %[[C0]] : tensor<?x4xf32, {byteir.bounded_shape = [4, 4]}>
// CHECK-DAG:   %[[V2:.*]] = "mhlo.dot"(%arg0, %[[V1]]) : (tensor<?x4xf32, {byteir.bounded_shape = [4, 4]}>, tensor<4x4xf32>) -> tensor<?x4xf32, {byteir.bounded_shape = [4, 4]}>
// CHECK-DAG:   %[[F0:.*]] = tensor.from_elements %[[D0]], %[[C4]] : tensor<2xindex>
// CHECK-DAG:   %[[V3:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[V0]], %[[F0]]) <{broadcast_dimensions = dense<1> : tensor<1xi64>}> : (tensor<4xf32>, tensor<2xindex>) -> tensor<?x4xf32, {byteir.bounded_shape = [4, 4]}>
// CHECK-DAG:   %[[V4:.*]] = mhlo.add %[[V2]], %[[V3]] : tensor<?x4xf32, {byteir.bounded_shape = [4, 4]}>
// CHECK:     return %[[V4]] : tensor<?x4xf32, {byteir.bounded_shape = [4, 4]}>

// CHECK-LABEL: func.func @dynamic_partition_and_mask_stitch_sub_1(%arg0: tensor<?x4xf32, {byteir.bounded_shape = [4, 4]}>) -> tensor<?x4xf32, {byteir.bounded_shape = [4, 4]}> attributes {__byteir_dynamic_sub_function} {
// CHECK-DAG:   %[[V0:.*]] = mhlo.constant dense<[0.0399630852, -1.36038888, -0.804892302, -0.616697788]> : tensor<4xf32>
// CHECK-DAG:   %[[V1:.*]] = mhlo.constant {{.*}} : tensor<4x4xf32>
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C4:.*]] = arith.constant 4 : index
// CHECK-DAG:   %[[D0:.*]] = tensor.dim %arg0, %[[C0]] : tensor<?x4xf32, {byteir.bounded_shape = [4, 4]}>
// CHECK-DAG:   %[[V2:.*]] = "mhlo.dot"(%arg0, %[[V1]]) : (tensor<?x4xf32, {byteir.bounded_shape = [4, 4]}>, tensor<4x4xf32>) -> tensor<?x4xf32, {byteir.bounded_shape = [4, 4]}>
// CHECK-DAG:   %[[F0:.*]] = tensor.from_elements %[[D0]], %[[C4]] : tensor<2xindex>
// CHECK-DAG:   %[[V3:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[V0]], %[[F0]]) <{broadcast_dimensions = dense<1> : tensor<1xi64>}> : (tensor<4xf32>, tensor<2xindex>) -> tensor<?x4xf32, {byteir.bounded_shape = [4, 4]}>
// CHECK-DAG:   %[[V4:.*]] = mhlo.add %[[V2]], %[[V3]] : tensor<?x4xf32, {byteir.bounded_shape = [4, 4]}>
// CHECK:     return %[[V4]] : tensor<?x4xf32, {byteir.bounded_shape = [4, 4]}>

// CHECK-LABEL:  func.func @dynamic_partition_and_mask_stitch_sub_0(%arg0: tensor<?x4xf32, {byteir.bounded_shape = [4, 4]}>, %arg1: tensor<?x4xf32, {byteir.bounded_shape = [4, 4]}>, %arg2: tensor<4xi32>) -> tensor<4x4xf32> attributes {__byteir_dynamic_sub_function} {
// CHECK-NEXT:    %0 = mhlo.custom_call @tf.DynamicMaskStitch(%arg0, %arg1, %arg2) {backend_config = ""} : (tensor<?x4xf32, {byteir.bounded_shape = [4, 4]}>, tensor<?x4xf32, {byteir.bounded_shape = [4, 4]}>, tensor<4xi32>) -> tensor<4x4xf32>
// CHECK-NEXT:    return %0 : tensor<4x4xf32>
