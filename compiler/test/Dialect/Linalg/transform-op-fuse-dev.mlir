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
