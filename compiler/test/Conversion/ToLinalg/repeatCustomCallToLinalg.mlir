// RUN: byteir-opt -hlo-fusion-to-linalg %s | FileCheck %s

func.func @convert_repeat_static(%arg0: tensor<16x96xf16>, %arg1: tensor<16xi64>) -> tensor<128x96xf16> attributes {__byteir_hlo_aggressive_fusion__} {
  %0 = mhlo.custom_call @byteir.repeat(%arg0, %arg1) {backend_config = "", byteir_attrs = {Trepeats = i64}, device = "host"} : (tensor<16x96xf16>, tensor<16xi64>) -> tensor<128x96xf16>
  return %0 : tensor<128x96xf16>
}
// CHECK-LABEL: func.func @convert_repeat_static(%arg0: tensor<16x96xf16>, %arg1: tensor<16xi64>) -> tensor<128x96xf16> attributes {__byteir_hlo_aggressive_fusion__} {
// CHECK-LABEL:   %c0 = arith.constant 0 : index
// CHECK-LABEL:   %c1 = arith.constant 1 : index
// CHECK-LABEL:   %dim = tensor.dim %arg0, %c0 : tensor<16x96xf16>
// CHECK-LABEL:   %0 = tensor.empty() : tensor<128x96xf16>
// CHECK-LABEL:   %1:2 = scf.for %arg2 = %c0 to %dim step %c1 iter_args(%arg3 = %0, %arg4 = %c0) -> (tensor<128x96xf16>, index) {
// CHECK-LABEL:     %extracted_slice = tensor.extract_slice %arg0[%arg2, 0] [1, 96] [1, 1] : tensor<16x96xf16> to tensor<1x96xf16>
// CHECK-LABEL:     %extracted = tensor.extract %arg1[%arg2] : tensor<16xi64>
// CHECK-LABEL:     %2 = arith.index_cast %extracted : i64 to index
// CHECK-LABEL:     %3 = tensor.empty(%2) : tensor<?x96xf16>
// CHECK-LABEL:     %4 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%extracted_slice : tensor<1x96xf16>) outs(%3 : tensor<?x96xf16>) attrs =  {byteir_attrs = {Trepeats = i64}, device = "host"} {
// CHECK-LABEL:     ^bb0(%in: f16, %out: f16):
// CHECK-LABEL:       linalg.yield %in : f16
// CHECK-LABEL:     } -> tensor<?x96xf16>
// CHECK-LABEL:     %inserted_slice = tensor.insert_slice %4 into %arg3[%arg4, 0] [%2, 96] [1, 1] : tensor<?x96xf16> into tensor<128x96xf16>
// CHECK-LABEL:     %5 = arith.addi %arg4, %2 : index
// CHECK-LABEL:     scf.yield %inserted_slice, %5 : tensor<128x96xf16>, index
// CHECK-LABEL:   }
// CHECK-LABEL:   return %1#0 : tensor<128x96xf16>
// CHECK-LABEL: }

func.func @convert_repeat_output_dynamic(%arg0: tensor<16x96xf16>, %arg1: tensor<16xi64>) -> tensor<?x96xf16> attributes {__byteir_hlo_aggressive_fusion__} {
  %0 = mhlo.custom_call @byteir.repeat(%arg0, %arg1) {backend_config = "", byteir_attrs = {Trepeats = i64}, device = "host"} : (tensor<16x96xf16>, tensor<16xi64>) -> tensor<?x96xf16>
  return %0 : tensor<?x96xf16>
}
// CHECK-LABEL: func.func @convert_repeat_output_dynamic(%arg0: tensor<16x96xf16>, %arg1: tensor<16xi64>) -> tensor<?x96xf16> attributes {__byteir_hlo_aggressive_fusion__} {
// CHECK-LABEL:   %c0 = arith.constant 0 : index
// CHECK-LABEL:   %c1 = arith.constant 1 : index
// CHECK-LABEL:   %dim = tensor.dim %arg0, %c0 : tensor<16x96xf16>
// CHECK-LABEL:   %c0_i64 = arith.constant 0 : i64
// CHECK-LABEL:   %0 = tensor.empty() : tensor<i64>
// CHECK-LABEL:   %1 = linalg.fill ins(%c0_i64 : i64) outs(%0 : tensor<i64>) -> tensor<i64>
// CHECK-LABEL:   %reduced = linalg.reduce ins(%arg1 : tensor<16xi64>) outs(%1 : tensor<i64>) dimensions = [0]
// CHECK-LABEL:     (%in: i64, %init: i64) {
// CHECK-LABEL:       %5 = arith.addi %in, %init : i64
// CHECK-LABEL:       linalg.yield %5 : i64
// CHECK-LABEL:     }
// CHECK-LABEL:   %extracted = tensor.extract %reduced[] : tensor<i64>
// CHECK-LABEL:   %2 = arith.index_cast %extracted : i64 to index
// CHECK-LABEL:   %3 = tensor.empty(%2) : tensor<?x96xf16>
// CHECK-LABEL:   %4:2 = scf.for %arg2 = %c0 to %dim step %c1 iter_args(%arg3 = %3, %arg4 = %c0) -> (tensor<?x96xf16>, index) {
// CHECK-LABEL:     %extracted_slice = tensor.extract_slice %arg0[%arg2, 0] [1, 96] [1, 1] : tensor<16x96xf16> to tensor<1x96xf16>
// CHECK-LABEL:     %extracted_0 = tensor.extract %arg1[%arg2] : tensor<16xi64>
// CHECK-LABEL:     %5 = arith.index_cast %extracted_0 : i64 to index
// CHECK-LABEL:     %6 = tensor.empty(%5) : tensor<?x96xf16>
// CHECK-LABEL:     %7 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%extracted_slice : tensor<1x96xf16>) outs(%6 : tensor<?x96xf16>) attrs =  {byteir_attrs = {Trepeats = i64}, device = "host"} {
// CHECK-LABEL:     ^bb0(%in: f16, %out: f16):
// CHECK-LABEL:       linalg.yield %in : f16
// CHECK-LABEL:     } -> tensor<?x96xf16>
// CHECK-LABEL:     %inserted_slice = tensor.insert_slice %7 into %arg3[%arg4, 0] [%5, 96] [1, 1] : tensor<?x96xf16> into tensor<?x96xf16>
// CHECK-LABEL:     %8 = arith.addi %arg4, %5 : index
// CHECK-LABEL:     scf.yield %inserted_slice, %8 : tensor<?x96xf16>, index
// CHECK-LABEL:   }
// CHECK-LABEL:   return %4#0 : tensor<?x96xf16>
// CHECK-LABEL: }

func.func @convert_repeat_input_dynamic(%arg0: tensor<?x96xf16>, %arg1: tensor<?xi64>) -> tensor<128x96xf16> attributes {__byteir_hlo_aggressive_fusion__} {
  %0 = mhlo.custom_call @byteir.repeat(%arg0, %arg1) {backend_config = "", byteir_attrs = {Trepeats = i64}, device = "host"} : (tensor<?x96xf16>, tensor<?xi64>) -> tensor<128x96xf16>
  return %0 : tensor<128x96xf16>
}
// CHECK-LABEL: func.func @convert_repeat_input_dynamic(%arg0: tensor<?x96xf16>, %arg1: tensor<?xi64>) -> tensor<128x96xf16> attributes {__byteir_hlo_aggressive_fusion__} {
// CHECK-LABEL:   %c0 = arith.constant 0 : index
// CHECK-LABEL:   %c1 = arith.constant 1 : index
// CHECK-LABEL:   %dim = tensor.dim %arg0, %c0 : tensor<?x96xf16>
// CHECK-LABEL:   %0 = tensor.empty() : tensor<128x96xf16>
// CHECK-LABEL:   %1:2 = scf.for %arg2 = %c0 to %dim step %c1 iter_args(%arg3 = %0, %arg4 = %c0) -> (tensor<128x96xf16>, index) {
// CHECK-LABEL:     %extracted_slice = tensor.extract_slice %arg0[%arg2, 0] [1, 96] [1, 1] : tensor<?x96xf16> to tensor<1x96xf16>
// CHECK-LABEL:     %extracted = tensor.extract %arg1[%arg2] : tensor<?xi64>
// CHECK-LABEL:     %2 = arith.index_cast %extracted : i64 to index
// CHECK-LABEL:     %3 = tensor.empty(%2) : tensor<?x96xf16>
// CHECK-LABEL:     %4 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%extracted_slice : tensor<1x96xf16>) outs(%3 : tensor<?x96xf16>) attrs =  {byteir_attrs = {Trepeats = i64}, device = "host"} {
// CHECK-LABEL:     ^bb0(%in: f16, %out: f16):
// CHECK-LABEL:       linalg.yield %in : f16
// CHECK-LABEL:     } -> tensor<?x96xf16>
// CHECK-LABEL:     %inserted_slice = tensor.insert_slice %4 into %arg3[%arg4, 0] [%2, 96] [1, 1] : tensor<?x96xf16> into tensor<128x96xf16>
// CHECK-LABEL:     %5 = arith.addi %arg4, %2 : index
// CHECK-LABEL:     scf.yield %inserted_slice, %5 : tensor<128x96xf16>, index
// CHECK-LABEL:   }
// CHECK-LABEL:   return %1#0 : tensor<128x96xf16>
// CHECK-LABEL: }

func.func @convert_repeat_dynamic(%arg0: tensor<?x96xf16>, %arg1: tensor<?xi64>) -> tensor<?x96xf16> attributes {__byteir_hlo_aggressive_fusion__} {
  %0 = mhlo.custom_call @byteir.repeat(%arg0, %arg1) {byteir_attrs = {Trepeats = i64}, device = "host"} : (tensor<?x96xf16>, tensor<?xi64>) -> tensor<?x96xf16>
  return %0 : tensor<?x96xf16>
}
// CHECK-LABEL: func.func @convert_repeat_dynamic(%arg0: tensor<?x96xf16>, %arg1: tensor<?xi64>) -> tensor<?x96xf16> attributes {__byteir_hlo_aggressive_fusion__} {
// CHECK-LABEL:   %c0 = arith.constant 0 : index
// CHECK-LABEL:   %c1 = arith.constant 1 : index
// CHECK-LABEL:   %dim = tensor.dim %arg0, %c0 : tensor<?x96xf16>
// CHECK-LABEL:   %c0_i64 = arith.constant 0 : i64
// CHECK-LABEL:   %0 = tensor.empty() : tensor<i64>
// CHECK-LABEL:   %1 = linalg.fill ins(%c0_i64 : i64) outs(%0 : tensor<i64>) -> tensor<i64>
// CHECK-LABEL:   %reduced = linalg.reduce ins(%arg1 : tensor<?xi64>) outs(%1 : tensor<i64>) dimensions = [0]
// CHECK-LABEL:     (%in: i64, %init: i64) {
// CHECK-LABEL:       %5 = arith.addi %in, %init : i64
// CHECK-LABEL:       linalg.yield %5 : i64
// CHECK-LABEL:     }
// CHECK-LABEL:   %extracted = tensor.extract %reduced[] : tensor<i64>
// CHECK-LABEL:   %2 = arith.index_cast %extracted : i64 to index
// CHECK-LABEL:   %3 = tensor.empty(%2) : tensor<?x96xf16>
// CHECK-LABEL:   %4:2 = scf.for %arg2 = %c0 to %dim step %c1 iter_args(%arg3 = %3, %arg4 = %c0) -> (tensor<?x96xf16>, index) {
// CHECK-LABEL:     %extracted_slice = tensor.extract_slice %arg0[%arg2, 0] [1, 96] [1, 1] : tensor<?x96xf16> to tensor<1x96xf16>
// CHECK-LABEL:     %extracted_0 = tensor.extract %arg1[%arg2] : tensor<?xi64>
// CHECK-LABEL:     %5 = arith.index_cast %extracted_0 : i64 to index
// CHECK-LABEL:     %6 = tensor.empty(%5) : tensor<?x96xf16>
// CHECK-LABEL:     %7 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%extracted_slice : tensor<1x96xf16>) outs(%6 : tensor<?x96xf16>) attrs =  {byteir_attrs = {Trepeats = i64}, device = "host"} {
// CHECK-LABEL:     ^bb0(%in: f16, %out: f16):
// CHECK-LABEL:       linalg.yield %in : f16
// CHECK-LABEL:     } -> tensor<?x96xf16>
// CHECK-LABEL:     %inserted_slice = tensor.insert_slice %7 into %arg3[%arg4, 0] [%5, 96] [1, 1] : tensor<?x96xf16> into tensor<?x96xf16>
// CHECK-LABEL:     %8 = arith.addi %arg4, %5 : index
// CHECK-LABEL:     scf.yield %inserted_slice, %8 : tensor<?x96xf16>, index
// CHECK-LABEL:   }
// CHECK-LABEL:   return %4#0 : tensor<?x96xf16>
// CHECK-LABEL: }
