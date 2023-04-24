// RUN: byteir-opt %s -transform-dialect-interpreter="erase-after" --canonicalize-ext -byteir-bufferize-opt \
// RUN:      -convert-linalg-ext-to-loops -convert-linalg-to-loops --canonicalize-ext -lower-affine -to-llvm \
// RUN: | byteir-cpu-runner -e main -entry-point-result=void \
// RUN:     --shared-libs=%mlir_lib_dir/libmlir_c_runner_utils%shlibext,%mlir_lib_dir/libmlir_runner_utils%shlibext \
// RUN: | FileCheck %s

module attributes {byteir.llvm_module} {
transform.sequence failures(propagate) {
^bb0(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["linalg_ext.scatter"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %1, %loops:2 = transform.structured.tile_ext %0 [1, 2] {interchange = [1, 0]}
}

func.func @scatter(%src: tensor<2x2x3xf32>, %indices: tensor<2x2xi64>, %update: tensor<2x3xf32>) -> (tensor<2x2x3xf32>) {
  %res = linalg_ext.scatter
    ins(%indices, %update: tensor<2x2xi64>, tensor<2x3xf32>)
    outs(%src: tensor<2x2x3xf32>) {
      ^bb0(%arg0: f32, %arg1: f32):
        %0 = arith.addf %arg0, %arg1 : f32
        linalg_ext.yield %0 : f32
      } -> tensor<2x2x3xf32>
  return %res : tensor<2x2x3xf32>
}

func.func @main() {
  %update = arith.constant dense<[
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0]
  ]> : tensor<2x3xf32>

  %indice = arith.constant dense<[
    [0, 1],
    [1, 0]
  ]> : tensor<2x2xi64>

  %cst_0 = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<2x2x3xf32>
  %src = linalg.fill ins(%cst_0 : f32) outs(%empty : tensor<2x2x3xf32>) -> tensor<2x2x3xf32>

  %result = call @scatter(%src, %indice, %update) : (tensor<2x2x3xf32>, tensor<2x2xi64>, tensor<2x3xf32>) -> tensor<2x2x3xf32>

  %result_unranked = tensor.cast %result : tensor<2x2x3xf32> to tensor<*xf32>
  call @printMemrefF32(%result_unranked) : (tensor<*xf32>) -> ()
  // CHECK: rank = 3 offset = 0 sizes = [2, 2, 3]
  //   CHECK-NEXT: [0, 0, 0]
  //   CHECK-NEXT: [1, 2, 3]
  //   CHECK-NEXT: [4, 5, 6]
  //   CHECK-NEXT: [0, 0, 0]
  return
}

func.func private @printMemrefF32(%ptr : tensor<*xf32>)
}