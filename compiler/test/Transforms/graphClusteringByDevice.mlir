// RUN: byteir-opt %s -allow-unregistered-dialect -graph-clustering-by-device -canonicalize | FileCheck %s
// RUN: byteir-opt %s -allow-unregistered-dialect -graph-clustering-by-device="dup-outputs" -canonicalize  | FileCheck %s --check-prefix DUPOUTPUTS

module {
  func.func @contain_string(%arg0: tensor<1x1x!tf_type.string>, %arg1: tensor<1x10xf32>, %arg2: tensor<1x10xf32>) -> tensor<1x10xf32> {
    %cst = "tf.Const"() {device = "host", value = dense<"string_to_compare"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
    %0 = "tf.Equal"(%arg0, %cst) {device = "host", incompatible_shape_error = true} : (tensor<1x1x!tf_type.string>, tensor<!tf_type.string>) -> tensor<1x1xi1>
    %1 = "mhlo.reshape"(%0) : (tensor<1x1xi1>) -> tensor<i1>
    %2 = "mhlo.select"(%1, %arg1, %arg2) : (tensor<i1>, tensor<1x10xf32>, tensor<1x10xf32>) -> tensor<1x10xf32>
    %3 = mhlo.add %2, %arg1 : tensor<1x10xf32>
    return %3 : tensor<1x10xf32>
  }
// CHECK-LABEL: func.func @contain_string(%arg0: tensor<1x1x!tf_type.string>, %arg1: tensor<1x10xf32>, %arg2: tensor<1x10xf32>) -> tensor<1x10xf32> {
// CHECK-NEXT: %[[RES0:.*]] = call @contain_string_host(%arg0) : (tensor<1x1x!tf_type.string>) -> tensor<1x1xi1>
// CHECK-NEXT: %[[RES1:.*]] = call @contain_string_test(%0, %arg1, %arg2) : (tensor<1x1xi1>, tensor<1x10xf32>, tensor<1x10xf32>) -> tensor<1x10xf32>

// CHECK: func.func @contain_string_host(%arg0: tensor<1x1x!tf_type.string>) -> tensor<1x1xi1> attributes {__byteir_host_device__, device = "host"} {
// CHECK-NEXT: %[[RES0:.*]] = "tf.Const"() {device = "host", value = dense<"string_to_compare"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
// CHECK-NEXT: %[[RES1:.*]] = "tf.Equal"(%arg0, %0) {device = "host", incompatible_shape_error = true} : (tensor<1x1x!tf_type.string>, tensor<!tf_type.string>) -> tensor<1x1xi1>

// CHECK: func.func @contain_string_test(%arg0: tensor<1x1xi1>, %arg1: tensor<1x10xf32>, %arg2: tensor<1x10xf32>) -> tensor<1x10xf32> attributes {__byteir_test_device__, device = "test"} {
// CHECK-NEXT: %[[RES0:.*]] = mhlo.reshape %arg0 : (tensor<1x1xi1>) -> tensor<i1>
// CHECK-NEXT: %[[RES1:.*]] = mhlo.select %0, %arg1, %arg2 : tensor<i1>, tensor<1x10xf32>
// CHECK-NEXT: %[[RES2:.*]] = mhlo.add %1, %arg1 : tensor<1x10xf32>

  func.func @no_host_ops(%arg0: tensor<1x1x!tf_type.string>, %arg1: tensor<1x10xf32>, %arg2: tensor<1x10xf32>) -> tensor<1x10xf32> {
    %cst = "tf.Const"() {value = dense<"string_to_compare"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
    %0 = "tf.Equal"(%arg0, %cst) {incompatible_shape_error = true} : (tensor<1x1x!tf_type.string>, tensor<!tf_type.string>) -> tensor<1x1xi1>
    %1 = "mhlo.reshape"(%0) : (tensor<1x1xi1>) -> tensor<i1>
    %2 = "mhlo.select"(%1, %arg1, %arg2) : (tensor<i1>, tensor<1x10xf32>, tensor<1x10xf32>) -> tensor<1x10xf32>
    %3 = mhlo.add %2, %arg1 : tensor<1x10xf32>
    return %3 : tensor<1x10xf32>
  }
// CHECK-LABEL:  func.func @no_host_ops(%arg0: tensor<1x1x!tf_type.string>, %arg1: tensor<1x10xf32>, %arg2: tensor<1x10xf32>) -> tensor<1x10xf32> {
// CHECK-NEXT: %0 = call @no_host_ops_test(%arg0, %arg1, %arg2) : (tensor<1x1x!tf_type.string>, tensor<1x10xf32>, tensor<1x10xf32>) -> tensor<1x10xf32>
// CHECK-NEXT: return %0 : tensor<1x10xf32>

  func.func @duplicate_splat_mhlo_const(%arg0: tensor<!tf_type.string>, %arg1: tensor<i1>, %arg2: tensor<i1>) -> tensor<i1> {
    %cst = "tf.Const"() {device = "host", value = dense<"string_to_compare"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
    %0 = "tf.Equal"(%arg0, %cst) {device = "host", incompatible_shape_error = true} : (tensor<!tf_type.string>, tensor<!tf_type.string>) -> tensor<i1>
    %1 = mhlo.constant dense<true> : tensor<i1>
    %2 = "mhlo.add"(%0, %1) {device = "host"} : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %3 = "mhlo.select"(%2, %arg1, %arg2) : (tensor<i1>, tensor<i1>, tensor<i1>) -> tensor<i1>
    %4 = mhlo.add %3, %1 : tensor<i1>
    return %4 : tensor<i1>
  }
// CHECK-LABEL: func.func @duplicate_splat_mhlo_const(%arg0: tensor<!tf_type.string>, %arg1: tensor<i1>, %arg2: tensor<i1>) -> tensor<i1> {
// CHECK-NEXT:    %0 = call @duplicate_splat_mhlo_const_host(%arg0) : (tensor<!tf_type.string>) -> tensor<i1>
// CHECK-NEXT:    %1 = call @duplicate_splat_mhlo_const_test(%0, %arg1, %arg2) : (tensor<i1>, tensor<i1>, tensor<i1>) -> tensor<i1>
// CHECK-NEXT:    return %1 : tensor<i1>
// CHECK-NEXT:  }

// CHECK-LABEL: func.func @duplicate_splat_mhlo_const_host(%arg0: tensor<!tf_type.string>) -> tensor<i1> attributes {__byteir_host_device__, device = "host"} {
// CHECK-NEXT:   %0 = mhlo.constant dense<true> : tensor<i1>
// CHECK-NEXT:   %1 = "tf.Const"() {device = "host", value = dense<"string_to_compare"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
// CHECK-NEXT:   %2 = "tf.Equal"(%arg0, %1) {device = "host", incompatible_shape_error = true} : (tensor<!tf_type.string>, tensor<!tf_type.string>) -> tensor<i1>
// CHECK-NEXT:   %3 = mhlo.add %2, %0 {device = "host"} : tensor<i1>
// CHECK-NEXT:   return %3 : tensor<i1>
// CHECK-NEXT: }

// CHECK-LABEL: func.func @duplicate_splat_mhlo_const_test(%arg0: tensor<i1>, %arg1: tensor<i1>, %arg2: tensor<i1>) -> tensor<i1> attributes {__byteir_test_device__, device = "test"} {
// CHECK-NEXT:   %0 = mhlo.constant dense<true> : tensor<i1>
// CHECK-NEXT:   %1 = mhlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<i1>
// CHECK-NEXT:   %2 = mhlo.add %1, %0 : tensor<i1>
// CHECK-NEXT:   return %2 : tensor<i1>
// CHECK-NEXT: }

  func.func @ops_used_by_internal_region(%arg0: !shape.witness, %arg1: tensor<?xf32>) -> tensor<1xindex> {
    %0 = shape.const_shape [32] : tensor<1xindex>
    %1 = shape.shape_of %arg1 : tensor<?xf32> -> tensor<1xindex>
    %2 = shape.assuming %arg0 -> (tensor<1xindex>) {
      %3 = shape.broadcast %1, %0 : tensor<1xindex>, tensor<1xindex> -> tensor<1xindex>
      shape.assuming_yield %3 : tensor<1xindex>
    }
    return %2 : tensor<1xindex>
  }
// CHECK-LABEL:   func.func @ops_used_by_internal_region
// CHECK-NEXT:      %0 = call @ops_used_by_internal_region_test(%arg1, %arg0) : (tensor<?xf32>, !shape.witness) -> tensor<1xindex>
// CHECK-NEXT:      return %0 : tensor<1xindex>
  
// CHECK-LABEL:   func.func @ops_used_by_internal_region_test
// CHECK:     return %2 : tensor<1xindex>

  func.func @duplicate_outputs() -> (tensor<1xi64>, tensor<1xi64>) {
    %0 = mhlo.constant dense<1> : tensor<1xi64>
    return %0, %0 : tensor<1xi64>, tensor<1xi64>
  }
// CHECK-LABEL:  func.func @duplicate_outputs
// CHECK-NEXT:     %0 = call @duplicate_outputs_test() : () -> tensor<1xi64>
// CHECK-NEXT:     return %0, %0 : tensor<1xi64>, tensor<1xi64>
  
// CHECK-LABEL:  func.func @duplicate_outputs_test
// CHECK-NEXT:     %0 = mhlo.constant dense<1> : tensor<1xi64>
// CHECK-NEXT:     return %0 : tensor<1xi64>

// DUPOUTPUTS-LABEL:  func.func @duplicate_outputs
// DUPOUTPUTS-NEXT:     %0:2 = call @duplicate_outputs_test() : () -> (tensor<1xi64>, tensor<1xi64>)
// DUPOUTPUTS-NEXT:     return %0#0, %0#1 : tensor<1xi64>, tensor<1xi64>

// DUPOUTPUTS-LABEL:  func.func @duplicate_outputs_test
// DUPOUTPUTS-NEXT:     %0 = mhlo.constant dense<1> : tensor<1xi64>
// DUPOUTPUTS-NEXT:     return %0, %0 : tensor<1xi64>, tensor<1xi64>

  func.func @return_host_constant(%arg0: tensor<f32>) -> (tensor<f32>, tensor<f32>) {
    %0 = "mhlo.constant"() {device = "host", value = dense<0.0000> : tensor<f32> } : () -> tensor<f32>
    %1 = mhlo.add %arg0, %arg0 : tensor<f32>
    return %0, %1 : tensor<f32>, tensor<f32>
  }
// CHECK-LABEL:  func.func @return_host_constant
// CHECK-NEXT:     %0 = call @return_host_constant_host() : () -> tensor<f32>
// CHECK-NEXT:     %1 = call @return_host_constant_test(%arg0) : (tensor<f32>) -> tensor<f32>
// CHECK-NEXT:     return %0, %1 : tensor<f32>, tensor<f32>

// CHECK-LABEL:  func.func @return_host_constant_host
// CHECK-NEXT:     %0 = mhlo.constant {device = "host"} dense<0.000000e+00> : tensor<f32>

// CHECK-LABEL:  func.func @return_host_constant_test
// CHECK-NEXT:     %0 = mhlo.add %arg0, %arg0 : tensor<f32>

}

