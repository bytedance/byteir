// RUN: byteir-opt --transform-dialect-interpreter --split-input-file %s | FileCheck %s

// CHECK-LABEL: func @tile_linalg_matmul
func.func @tile_linalg_matmul(
  %arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>, %arg2: tensor<128x128xf32>)
    -> tensor<128x128xf32> {
// CHECK: scf.for
// CHECK:   scf.for
// CHECK:     scf.for
// CHECK:       linalg.matmul
// CHECK:       scf.yield
// CHECK:     } {__byteir_parallel__}
// CHECK:     scf.yield
// CHECK:   } {__byteir_parallel__}
// CHECK:   scf.yield
// CHECK: }
  %0 = linalg.matmul ins(%arg0, %arg1: tensor<128x128xf32>, tensor<128x128xf32>)
                     outs(%arg2: tensor<128x128xf32>)
    -> tensor<128x128xf32>
  return %0 : tensor<128x128xf32>
}

transform.sequence failures(propagate) {
^bb0(%arg0: !pdl.operation):
  %0 = transform.structured.match ops{["linalg.matmul"]} in %arg0 : (!pdl.operation) -> !pdl.operation
  %1, %loops:3 = transform.structured.tile_ext %0 [2, 4, 8] {interchange = [2, 1, 0]}
  transform.structured.tile_loop_hint %1 : !pdl.operation
}

// -----

// CHECK-LABEL: func @softmax_tensor
func.func @softmax_tensor(%arg0: tensor<1024x64xf32>) -> (tensor<1024x64xf32>) {
// CHECK: scf.for
// CHECK:   linalg_ext.softmax
// CHECK:   scf.yield
// CHECK: } {__byteir_parallel__}
  %0 = tensor.empty() : tensor<1024x64xf32>
  %1 = tensor.empty() : tensor<1024xf32>
  %2 = tensor.empty() : tensor<1024xf32>
  %3 = tensor.empty() : tensor<1024xf32>
  %4:4 = linalg_ext.softmax
    dimension(1)
    ins(%arg0 : tensor<1024x64xf32>) outs(%0, %1, %2, %3 : tensor<1024x64xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>) : tensor<1024x64xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>
  return %4#0 : tensor<1024x64xf32>
}


transform.sequence failures(propagate) {
^bb0(%arg0: !pdl.operation):
  %0 = transform.structured.match ops{["linalg_ext.softmax"]} in %arg0 : (!pdl.operation) -> !pdl.operation
  %1, %loop = transform.structured.tile_ext %0 [4]
  transform.structured.tile_loop_hint %1 : !pdl.operation
}

// -----

//CHECK-LABEL: func.func @softmax_memref
func.func @softmax_memref(%arg0: memref<1024x64xf32>) -> (memref<1024x64xf32>) {
//CHECK: scf.for
//CHECK:   linalg_ext.softmax
//CHECK: } {__byteir_parallel__}
//CHECK: return
  %0 = memref.alloc() : memref<1024x64xf32>
  %1 = memref.alloc() : memref<1024xf32>
  %2 = memref.alloc() : memref<1024xf32>
  %3 = memref.alloc() : memref<1024xf32>
  linalg_ext.softmax {__root__}
    dimension(1)
    ins(%arg0 : memref<1024x64xf32>) outs(%0, %1, %2, %3 : memref<1024x64xf32>, memref<1024xf32>, memref<1024xf32>, memref<1024xf32>)
  return %0 : memref<1024x64xf32>
}

transform.sequence failures(propagate) {
^bb0(%arg0: !pdl.operation):
  %0 = transform.structured.match ops{["linalg_ext.softmax"]} in %arg0 : (!pdl.operation) -> !pdl.operation
  %1, %loops = transform.structured.tile_ext %0 [4]
  transform.structured.tile_loop_hint %1 : !pdl.operation
}

