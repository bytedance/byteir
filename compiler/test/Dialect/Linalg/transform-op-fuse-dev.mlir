// RUN: byteir-opt %s --transform-dialect-interpreter --split-input-file | FileCheck %s

// this is called dev, since it is not perfect yet.

// CHECK-LABEL: func.func @fuse_2_add_sharing_input
func.func @fuse_2_add_sharing_input(%arg0: tensor<1024x512xf32>, %arg1: tensor<1024x512xf32>, %arg2: tensor<1024x512xf32>) -> (tensor<1024x512xf32>,tensor<1024x512xf32>) {
// CHECK: linalg.elemwise_binary {__other__}
// CHECK: scf.for
// CHECK:   scf.for
// CHECK:     linalg.elemwise_binary {__root__}
// CHECK:     scf.yield
// CHECK:   } {__byteir_parallel__}
// CHECK:   scf.yield
// CHECK: } {__byteir_parallel__}
  %0 = tensor.empty() : tensor<1024x512xf32>
  %1 = tensor.empty() : tensor<1024x512xf32>
  %2 = tensor.empty() : tensor<1024x512xf32>
  %3 = linalg.elemwise_binary {__other__} ins(%arg0, %arg1 : tensor<1024x512xf32>, tensor<1024x512xf32>)
                             outs(%0: tensor<1024x512xf32>) -> tensor<1024x512xf32>
  %4 = linalg.elemwise_binary {__root__} ins(%arg1, %arg2 : tensor<1024x512xf32>, tensor<1024x512xf32>)
                             outs(%1: tensor<1024x512xf32>) -> tensor<1024x512xf32>
  return %3, %4: tensor<1024x512xf32>, tensor<1024x512xf32>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match attributes{"__root__"} in %arg1 : (!pdl.operation) -> !pdl.operation
  %1, %loops:2 = transform.structured.fuse_ext %0 {tile_sizes = [4, 8], tile_interchange = [1, 0]}
  transform.structured.tile_loop_hint %1 
  cleanup
}

// -----

// CHECK-LABEL: func.func @diamond
func.func @diamond(%arg0: tensor<1024x512xf32>) -> tensor<1024x512xf32> {
  // CHECK: scf.for
  // CHECK:     __revisited__
  // CHECK-NOT: __revisited__
  // CHECK:     scf.yield
  %0 = tensor.empty() : tensor<1024x512xf32>
  %1 = linalg.elemwise_unary {__revisited__} ins(%arg0 : tensor<1024x512xf32>) outs(%0 : tensor<1024x512xf32>) -> tensor<1024x512xf32>
  %2 = linalg.elemwise_unary {__path_0__} ins(%1 : tensor<1024x512xf32>) outs(%0 : tensor<1024x512xf32>) -> tensor<1024x512xf32>
  %3 = linalg.elemwise_unary {__path_1__} ins(%1 : tensor<1024x512xf32>) outs(%0 : tensor<1024x512xf32>) -> tensor<1024x512xf32>
  %4 = linalg.elemwise_binary {__root__} ins(%2, %3 : tensor<1024x512xf32>, tensor<1024x512xf32>) outs(%0 : tensor<1024x512xf32>) -> tensor<1024x512xf32>
  return %4 : tensor<1024x512xf32>
}
transform.sequence  failures(propagate) {
^bb0(%arg0: !pdl.operation):
  %0 = transform.structured.match attributes {__root__} in %arg0 : (!pdl.operation) -> !pdl.operation
  %transformed, %loops = transform.structured.fuse_ext %0 {tile_interchange = [], tile_sizes = [4]}
}

// -----

// CHECK-LABEL: func.func @fuse_with_stop
func.func @fuse_with_stop(%arg0: tensor<1024x512xf32>) -> tensor<1024x512xf32> {
  %0 = tensor.empty() : tensor<1024x512xf32>
  %1 = linalg.elemwise_unary {__stop_1__} ins(%arg0 : tensor<1024x512xf32>) outs(%0 : tensor<1024x512xf32>) -> tensor<1024x512xf32>
  // CHECK: scf.for
  // CHECK-NOT: __stop_1__
  // CHECK-DAG:     __block_1_path_0__
  // CHECK-DAG:     __block_1_path_1__
  // CHECK:     __root_1__
  // CHECK:     scf.yield
  %2 = linalg.elemwise_unary {__block_1_path_0__} ins(%1 : tensor<1024x512xf32>) outs(%0 : tensor<1024x512xf32>) -> tensor<1024x512xf32>
  %3 = linalg.elemwise_unary {__block_1_path_1__} ins(%1 : tensor<1024x512xf32>) outs(%0 : tensor<1024x512xf32>) -> tensor<1024x512xf32>
  %4 = linalg.elemwise_binary {__root_1__, __stop_0__} ins(%2, %3 : tensor<1024x512xf32>, tensor<1024x512xf32>) outs(%0 : tensor<1024x512xf32>) -> tensor<1024x512xf32>
  // CHECK: scf.for
  // CHECK-NOT: __stop_0__
  // CHECK-DAG:     __block_0_path_0__
  // CHECK-DAG:     __block_0_path_1__
  // CHECK:     __root_0__
  // CHECK:     scf.yield
  %5 = linalg.elemwise_unary {__block_0_path_0__} ins(%4 : tensor<1024x512xf32>) outs(%0 : tensor<1024x512xf32>) -> tensor<1024x512xf32>
  %6 = linalg.elemwise_unary {__block_0_path_1__} ins(%4 : tensor<1024x512xf32>) outs(%0 : tensor<1024x512xf32>) -> tensor<1024x512xf32>
  %7 = linalg.elemwise_binary {__root_0__} ins(%5, %6 : tensor<1024x512xf32>, tensor<1024x512xf32>) outs(%0 : tensor<1024x512xf32>) -> tensor<1024x512xf32>
  return %7 : tensor<1024x512xf32>
}
transform.sequence  failures(propagate) {
^bb0(%arg0: !pdl.operation):
  %root_0 = transform.structured.match attributes {__root_0__} in %arg0 : (!pdl.operation) -> !pdl.operation
  %stop_0 = transform.structured.match attributes {__stop_0__} in %arg0 : (!pdl.operation) -> !pdl.operation
  %root_1 = transform.structured.match attributes {__root_1__} in %arg0 : (!pdl.operation) -> !pdl.operation
  %stop_1 = transform.structured.match attributes {__stop_1__} in %arg0 : (!pdl.operation) -> !pdl.operation
  %transformed_0, %loops_0 = transform.structured.fuse_ext %root_0, %stop_0 {tile_interchange = [], tile_sizes = [4]}
  %transformed_1, %loops_1 = transform.structured.fuse_ext %root_1, %stop_1 {tile_interchange = [], tile_sizes = [4]}
  cleanup
}

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: func.func @resnet_block
func.func @resnet_block(%arg0: tensor<1x56x56x256xf16>) -> tensor<1x56x56x256xf16> {
  %cst = arith.constant dense_resource<__elided__> : tensor<256x1x1x64xf32>
  %cst_0 = arith.constant dense_resource<__elided__> : tensor<64x3x3x64xf32>
  %cst_1 = arith.constant dense_resource<__elided__> : tensor<64x1x1x256xf32>
  %cst_2 = arith.constant 0.000000e+00 : f16
  %0 = tensor.empty() : tensor<1x56x56x256xf16>
  // CHECK: scf.for
  // CHECK:   scf.for
  %1 = linalg.fill ins(%cst_2 : f16) outs(%0 : tensor<1x56x56x256xf16>) -> tensor<1x56x56x256xf16>
  %2 = linalg.elemwise_unary {__revisited__} ins(%arg0 : tensor<1x56x56x256xf16>) outs(%1 : tensor<1x56x56x256xf16>) -> tensor<1x56x56x256xf16>
  // CHECK: %[[REVISITED:.*]] = linalg.elemwise_unary {__revisited__}
  // CHECK-NOT: __revisited__
  // CHECK-DAG: %[[SUB_SLICE_0:.*]] = tensor.extract_slice %[[REVISITED]]{{.*}} to tensor<1x8x56x32xf16>
  // CHECK-DAG: %[[SUB_SLICE_1:.*]] = tensor.extract_slice %[[REVISITED]]{{.*}} to tensor<1x?x56x256xf16>
  %3 = tensor.empty() : tensor<1x56x56x64xf16>
  %4 = linalg.fill ins(%cst_2 : f16) outs(%3 : tensor<1x56x56x64xf16>) -> tensor<1x56x56x64xf16>
  %5 = linalg.conv_2d_nhwc_fhwc {__conv_0__, dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%2, %cst_1 : tensor<1x56x56x256xf16>, tensor<64x1x1x256xf32>) outs(%4 : tensor<1x56x56x64xf16>) -> tensor<1x56x56x64xf16>
  // CHECK-DAG: %[[CONV_0:.*]] = {{.*}}__conv_0__{{.*}}%[[SUB_SLICE_1]]
  %padded = tensor.pad %5 nofold low[0, 1, 1, 0] high[0, 1, 1, 0] {
  ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
    tensor.yield %cst_2 : f16
  } : tensor<1x56x56x64xf16> to tensor<1x58x58x64xf16>
  // CHECK-DAG: %[[PAD:.*]] = tensor.pad{{.*}}%[[CONV_0:.*]]
  %6 = tensor.empty() : tensor<1x56x56x64xf16>
  %7 = linalg.fill ins(%cst_2 : f16) outs(%6 : tensor<1x56x56x64xf16>) -> tensor<1x56x56x64xf16>
  %8 = linalg.conv_2d_nhwc_fhwc {__conv_1__, dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%padded, %cst_0 : tensor<1x58x58x64xf16>, tensor<64x3x3x64xf32>) outs(%7 : tensor<1x56x56x64xf16>) -> tensor<1x56x56x64xf16>
  // CHECK-DAG: %[[CONV_1:.*]] = {{.*}}__conv_1__{{.*}}%[[PAD]]
  %9 = tensor.empty() : tensor<1x56x56x256xf16>
  %10 = linalg.fill ins(%cst_2 : f16) outs(%9 : tensor<1x56x56x256xf16>) -> tensor<1x56x56x256xf16>
  %11 = linalg.conv_2d_nhwc_fhwc {__conv_2__, dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%8, %cst : tensor<1x56x56x64xf16>, tensor<256x1x1x64xf32>) outs(%10 : tensor<1x56x56x256xf16>) -> tensor<1x56x56x256xf16>
  // CHECK-DAG: %[[CONV_2:.*]] = {{.*}}__conv_2__{{.*}}%[[CONV_1]]
  %12 = tensor.empty() : tensor<1x56x56x256xf16>
  %13 = linalg.generic {__root__, indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2, %11 : tensor<1x56x56x256xf16>, tensor<1x56x56x256xf16>) outs(%12 : tensor<1x56x56x256xf16>) {
  ^bb0(%in: f16, %in_3: f16, %out: f16):
    %14 = arith.addf %in, %in_3 : f16
    linalg.yield %14 : f16
  } -> tensor<1x56x56x256xf16>
  // CHECK-DAG: linalg.generic{{.*}}%[[SUB_SLICE_0]], %[[CONV_2]]
  // CHECK:   scf.yield
  // CHECK: scf.yield
  return %13 : tensor<1x56x56x256xf16>
}
transform.sequence  failures(propagate) {
^bb0(%arg0: !pdl.operation):
  %0 = transform.structured.match attributes {__root__} in %arg0 : (!pdl.operation) -> !pdl.operation
  %transformed, %loops:2 = transform.structured.fuse_ext %0 {tile_interchange = [], tile_sizes = [0, 8, 0, 32]}
  cleanup
}

// -----

module {
  func.func @donot_merge_slices_on_different_dims(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
// CHECK: scf.for
// CHECK:   scf.for
// CHECK:     linalg.elemwise_unary
// CHECK:     linalg.elemwise_unary
// CHECK:     linalg.matmul
// CHECK:     scf.yield
// CHECK:   scf.yield
    %0 = tensor.empty() : tensor<32x32xf32>
    %1 = tensor.empty() : tensor<32x32xf32>
    %2 = linalg.elemwise_unary ins(%arg0 : tensor<32x32xf32>) outs(%0 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %3 = linalg.matmul {__root__} ins(%2, %2 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%1 : tensor<32x32xf32>) -> tensor<32x32xf32>
    return %3 : tensor<32x32xf32>
  }
  transform.sequence  failures(propagate) {
  ^bb0(%arg0: !pdl.operation):
    %0 = transform.structured.match attributes {__root__} in %arg0 : (!pdl.operation) -> !pdl.operation
    %transformed, %loops:2 = transform.structured.fuse_ext %0 {tile_interchange = [], tile_sizes = [4, 8]}
  }
}
