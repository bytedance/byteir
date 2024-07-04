// RUN: byteir-opt %s --allow-unregistered-dialect| FileCheck %s

func.func @custom_tensor(%arg0: tensor<1024x64xf32>) -> (tensor<1024x64xf32>) {
  %0 = linalg_ext.custom {target_name = "foo"} outs(%arg0 : tensor<1024x64xf32>) {
    ^bb0(%arg1 : tensor<1024x64xf32>):  // no predecessors
      %1 = "mhlo.custom_call"(%arg1) {call_target_name = "bar", has_side_effect = false} : (tensor<1024x64xf32>) -> tensor<1024x64xf32>
      linalg_ext.yield %1 : tensor<1024x64xf32>
  } -> tensor<1024x64xf32>
  return %0 : tensor<1024x64xf32>
}
//CHECK-LABEL: func.func @custom_tensor
//CHECK: linalg_ext.custom
//CHECK: mhlo.custom_call

func.func @scan_1d_tensor(%0: tensor<128xi32>) -> tensor<128xi32> {
  %c0 = tensor.empty() : tensor<i32>
  %1 = tensor.empty() : tensor<128xi32>
  %2:2 = linalg_ext.scan
    dimension(0) inclusive(true)
    ins(%0 : tensor<128xi32>) outs(%1, %c0 : tensor<128xi32>, tensor<i32>) {
    ^bb0(%arg0 : i32, %arg1 : i32):
      %sum = arith.addi %arg0, %arg1 : i32
      linalg_ext.yield %sum : i32
  } -> tensor<128xi32>, tensor<i32>
  return %2#0 : tensor<128xi32>
}
//CHECK-LABEL: func.func @scan_1d_tensor
//CHECK: linalg_ext.scan

func.func @scan_2d_tensor(%0: tensor<16x32xi32>) -> tensor<16x32xi32> {
  %c0 = tensor.empty() : tensor<32xi32>
  %1 = tensor.empty() : tensor<16x32xi32>
  %2:2 = linalg_ext.scan
    dimension(0) inclusive(true)
    ins(%0 : tensor<16x32xi32>) outs(%1, %c0 : tensor<16x32xi32>, tensor<32xi32>) {
    ^bb0(%arg0 : i32, %arg1 : i32):
      %sum = arith.addi %arg0, %arg1 : i32
      linalg_ext.yield %sum : i32
  } -> tensor<16x32xi32>, tensor<32xi32>
  return %2#0 : tensor<16x32xi32>
}
//CHECK-LABEL: func.func @scan_2d_tensor
//CHECK: linalg_ext.scan

func.func @scan_2d_memref(%0: memref<16x32xi32>, %1: memref<16x32xi32>) {
  %c0 = memref.alloc() : memref<32xi32>
  linalg_ext.scan
    dimension(0) inclusive(true)
    ins(%0 : memref<16x32xi32>) outs(%1, %c0 : memref<16x32xi32>, memref<32xi32>) {
    ^bb0(%arg0 : i32, %arg1 : i32):
      %sum = arith.addi %arg0, %arg1 : i32
      linalg_ext.yield %sum : i32
  }
  return
}
//CHECK-LABEL: func.func @scan_2d_memref
//CHECK: linalg_ext.scan

func.func @softmax_tensor(%arg0: tensor<1024x64xf32>) -> (tensor<1024x64xf32>) {
  %0 = tensor.empty() : tensor<1024x64xf32>
  %1 = tensor.empty() : tensor<64xf32>
  %2 = tensor.empty() : tensor<64xf32>
  %3 = tensor.empty() : tensor<64xf32>
  %4:4 = linalg_ext.softmax 
    dimension(0) 
    ins(%arg0 : tensor<1024x64xf32>) outs(%0, %1, %2, %3 : tensor<1024x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) : tensor<1024x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>
  return %4#0 : tensor<1024x64xf32>
}
//CHECK-LABEL: func.func @softmax_tensor
//CHECK: linalg_ext.softmax

func.func @softmax_memref(%arg0: memref<1024x64xf32>) -> (memref<1024x64xf32>) {
  %0 = memref.alloc() : memref<1024x64xf32>
  %1 = memref.alloc() : memref<64xf32>
  %2 = memref.alloc() : memref<64xf32>
  %3 = memref.alloc() : memref<64xf32>
  linalg_ext.softmax 
    dimension(0)
    ins(%arg0 : memref<1024x64xf32>) outs(%0, %1, %2, %3 : memref<1024x64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>)
  return %0 : memref<1024x64xf32>
}
//CHECK-LABEL: func.func @softmax_memref
//CHECK: linalg_ext.softmax

func.func @unnorm_softmax_tensor(%arg0: tensor<1024x64xf32>) -> (tensor<1024x64xf32>) {
  %0 = tensor.empty() : tensor<1024x64xf32>
  %1 = tensor.empty() : tensor<64xf32>
  %2 = tensor.empty() : tensor<64xf32>
  %3 = tensor.empty() : tensor<64xf32>
  %4:4 = linalg_ext.unnorm_softmax 
    dimension(0) 
    ins(%arg0 : tensor<1024x64xf32>) outs(%0, %1, %2, %3 : tensor<1024x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) : tensor<1024x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>
  return %4#0 : tensor<1024x64xf32>
}
//CHECK-LABEL: func.func @unnorm_softmax_tensor
//CHECK: linalg_ext.unnorm_softmax

func.func @unnorm_softmax_memref(%arg0: memref<1024x64xf32>) -> (memref<1024x64xf32>) {
  %0 = memref.alloc() : memref<1024x64xf32>
  %1 = memref.alloc() : memref<64xf32>
  %2 = memref.alloc() : memref<64xf32>
  %3 = memref.alloc() : memref<64xf32>
  linalg_ext.unnorm_softmax 
    dimension(0)
    ins(%arg0 : memref<1024x64xf32>) outs(%0, %1, %2, %3 : memref<1024x64xf32>, memref<64xf32>, memref<64xf32>, memref<64xf32>)
  return %0 : memref<1024x64xf32>
}
//CHECK-LABEL: func.func @unnorm_softmax_memref
//CHECK: linalg_ext.unnorm_softmax

func.func @diag_tensor(%arg0: tensor<1024xf32>) -> (tensor<1024x1024xf32>) {
  %0 = tensor.empty() : tensor<1024x1024xf32>
  %1 = linalg_ext.diag 
    ins(%arg0 : tensor<1024xf32>) outs(%0 : tensor<1024x1024xf32>) : tensor<1024x1024xf32>
  return %1 : tensor<1024x1024xf32>
}
//CHECK-LABEL: func.func @diag_tensor
//CHECK: linalg_ext.diag

func.func @diag_tensor_2d(%arg0: tensor<512x1024xf32>) -> (tensor<512x1024x1024xf32>) {
  %0 = tensor.empty() : tensor<512x1024x1024xf32>
  %1 = linalg_ext.diag 
    ins(%arg0 : tensor<512x1024xf32>) outs(%0 : tensor<512x1024x1024xf32>) : tensor<512x1024x1024xf32>
  return %1 : tensor<512x1024x1024xf32>
}
//CHECK-LABEL: func.func @diag_tensor_2d
//CHECK: linalg_ext.diag

func.func @diag_memref(%arg0: memref<1024xf32>) -> (memref<1024x1024xf32>) {
  %0 = memref.alloc() : memref<1024x1024xf32>
  linalg_ext.diag 
    ins(%arg0 : memref<1024xf32>) outs(%0 : memref<1024x1024xf32>)
  return %0 : memref<1024x1024xf32>
}
//CHECK-LABEL: func.func @diag_memref
//CHECK: linalg_ext.diag

func.func @diag_memref_2d(%arg0: memref<512x1024xf32>) -> (memref<512x1024x1024xf32>) {
  %0 = memref.alloc() : memref<512x1024x1024xf32>
  linalg_ext.diag 
    ins(%arg0 : memref<512x1024xf32>) outs(%0 : memref<512x1024x1024xf32>)
  return %0 : memref<512x1024x1024xf32>
}
//CHECK-LABEL: func.func @diag_memref_2d
//CHECK: linalg_ext.diag

func.func @alias(%arg0: f32) -> (f32) {
  %0 = linalg_ext.alias(%arg0 : f32) : f32
  return %0 : f32
}
//CHECK-LABEL: func.func @alias
//CHECK: linalg_ext.alias

#map0 = affine_map<(d0, d1) -> (d0, d1)>
func.func @alias_in_generic(%arg0: tensor<?x?xf32>) -> (tensor<?x?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %2 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %3 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel"]}
      ins(%arg0 : tensor<?x?xf32>)
      outs(%2 : tensor<?x?xf32>) {
    ^bb0(%arg3: f32, %arg4: f32): 
      %4 = linalg_ext.alias(%arg3 : f32) : f32    
      linalg.yield %4 : f32
  } -> tensor<?x?xf32>
  return %3 : tensor<?x?xf32>
}
//CHECK-LABEL: func.func @alias_in_generic
//CHECK: linalg.generic 
//CHECK: linalg_ext.alias

func.func @topk_tensor(%input_values: tensor<1024x64xf32>, %input_indices: tensor<1024x64xi32>) -> (tensor<1024x3xf32>, tensor<1024x3xi32>) {
  %out_values = tensor.empty() : tensor<1024x3xf32>
  %out_indices = tensor.empty() : tensor<1024x3xi32>
  %0:2 = linalg_ext.topk
        dimension(1)
        ins(%input_values, %input_indices : tensor<1024x64xf32> , tensor<1024x64xi32>)
        outs(%out_values, %out_indices : tensor<1024x3xf32>, tensor<1024x3xi32>) {
        ^bb0(%arg0: f32, %arg1: f32):  // no predecessors
          %0 = arith.cmpf ogt, %arg0, %arg1 : f32
          linalg_ext.yield %0 : i1
        } -> tensor<1024x3xf32>, tensor<1024x3xi32>
  return %0#0, %0#1 : tensor<1024x3xf32>, tensor<1024x3xi32>
}
//CHECK-LABEL: func.func @topk_tensor
//CHECK: linalg_ext.topk

func.func @topk_memref(%input_values: memref<1024x64xf32>, %input_indices: memref<1024x64xi32>) -> (memref<1024x3xf32>, memref<1024x3xi32>) {
  %out_values = memref.alloc() : memref<1024x3xf32>
  %out_indices = memref.alloc() : memref<1024x3xi32>
  linalg_ext.topk
    dimension(1)
    ins(%input_values, %input_indices : memref<1024x64xf32> , memref<1024x64xi32>)
    outs(%out_values, %out_indices : memref<1024x3xf32>, memref<1024x3xi32>) {
    ^bb0(%arg0: f32, %arg1: f32):  // no predecessors
      %0 = arith.cmpf ogt, %arg0, %arg1 : f32
      linalg_ext.yield %0 : i1
    }
  return %out_values, %out_indices : memref<1024x3xf32>, memref<1024x3xi32>
}
//CHECK-LABEL: func.func @topk_memref
//CHECK: linalg_ext.topk

func.func @topk_tensor_optional(%input_values: tensor<1024x64xf32>) -> (tensor<1024x3xf32>, tensor<1024x3xi32>) {
  %out_values = tensor.empty() : tensor<1024x3xf32>
  %out_indices = tensor.empty() : tensor<1024x3xi32>
  %0:2 = linalg_ext.topk
        dimension(1)
        ins(%input_values : tensor<1024x64xf32>)
        outs(%out_values, %out_indices : tensor<1024x3xf32>, tensor<1024x3xi32>) {
        ^bb0(%arg0: f32, %arg1: f32):  // no predecessors
          %0 = arith.cmpf ogt, %arg0, %arg1 : f32
          linalg_ext.yield %0 : i1
        } -> tensor<1024x3xf32>, tensor<1024x3xi32>
  return %0#0, %0#1 : tensor<1024x3xf32>, tensor<1024x3xi32>
}
//CHECK-LABEL: func.func @topk_tensor_optional
//CHECK: linalg_ext.topk


func.func @topk_tensor_dynamic(%input_values: tensor<?x?xf32>, %input_indices: tensor<?x?xi32>, %out_values: tensor<?x?xf32>, %out_indices: tensor<?x?xi32>) -> (tensor<?x?xf32>, tensor<?x?xi32>) {
  %0:2 = linalg_ext.topk
        dimension(1)
        ins(%input_values, %input_indices : tensor<?x?xf32> , tensor<?x?xi32>)
        outs(%out_values, %out_indices : tensor<?x?xf32>, tensor<?x?xi32>) {
        ^bb0(%arg0: f32, %arg1: f32):  // no predecessors
          %0 = arith.cmpf ogt, %arg0, %arg1 : f32
          linalg_ext.yield %0 : i1
        } -> tensor<?x?xf32>, tensor<?x?xi32>
  return %0#0, %0#1 : tensor<?x?xf32>, tensor<?x?xi32>
}
//CHECK-LABEL: func.func @topk_tensor_dynamic
//CHECK: linalg_ext.topk

func.func @batch_matmul_3d(%ta3: tensor<8x32x128xf32>, %tb3: tensor<8x128x64xf32>, %tc3: tensor<8x32x64xf32>) -> (tensor<8x32x64xf32>)
{
  %res = linalg_ext.batch_matmul
                    ins(%ta3, %tb3: tensor<8x32x128xf32>, tensor<8x128x64xf32>)
                    outs(%tc3: tensor<8x32x64xf32>)
                    layout = "nn"
  return %res : tensor<8x32x64xf32>
}
//CHECK-LABEL: func.func @batch_matmul_3d

func.func @batch_matmul_4d(%ta4: tensor<16x8x32x128xf32>, %tb4: tensor<16x8x128x64xf32>, %tc4: tensor<16x8x32x64xf32>) -> (tensor<16x8x32x64xf32>)
{
  %res = linalg_ext.batch_matmul
                    ins(%ta4, %tb4: tensor<16x8x32x128xf32>, tensor<16x8x128x64xf32>)
                    outs(%tc4: tensor<16x8x32x64xf32>)
                    layout = "nn"
  return %res : tensor<16x8x32x64xf32>
}
//CHECK-LABEL: func.func @batch_matmul_4d

func.func @batch_matmul_3d_memref(%ta3: memref<8x32x128xf32>, %tb3: memref<8x128x64xf32>, %tc3: memref<8x32x64xf32>) -> (memref<8x32x64xf32>)
{
  linalg_ext.batch_matmul
                    ins(%ta3, %tb3: memref<8x32x128xf32>, memref<8x128x64xf32>)
                    outs(%tc3: memref<8x32x64xf32>)
                    layout = "nn"
  return %tc3 : memref<8x32x64xf32>
}
//CHECK-LABEL: func.func @batch_matmul_3d_memref

func.func @layer_norm_3d_tensor(%arg0: tensor<8x32x128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<8x32x128xf32>) -> tensor<8x32x128xf32> {
  %res = linalg_ext.layer_norm axis([2]) epsilon(9.9999999747524271E-7) ins(%arg0, %arg1, %arg2 : tensor<8x32x128xf32>, tensor<128xf32>, tensor<128xf32>) outs(%arg3 : tensor<8x32x128xf32>) : tensor<8x32x128xf32>
  return %res : tensor<8x32x128xf32>
}
//CHECK-LABEL: func.func @layer_norm_3d_tensor

func.func @layer_norm_3d_memref(%arg0: memref<8x32x128xf32>, %arg1: memref<128xf32>, %arg2: memref<128xf32>) -> memref<8x32x128xf32> {
  %0 = memref.alloc() : memref<8x32x128xf32>
  linalg_ext.layer_norm axis([2]) epsilon(9.9999999747524271E-7) ins(%arg0, %arg1, %arg2 : memref<8x32x128xf32>, memref<128xf32>, memref<128xf32>) outs(%0 : memref<8x32x128xf32>)
  return %0 : memref<8x32x128xf32>
}
//CHECK-LABEL: func.func @layer_norm_3d_memref
