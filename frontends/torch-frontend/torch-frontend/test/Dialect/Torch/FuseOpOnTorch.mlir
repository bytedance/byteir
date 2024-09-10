// RUN: torch-frontend-opt %s --cse --fuse-op-on-torch="valid-custom-call-ops=byteir.l2_norm" | FileCheck %s

func.func @torch.gelu.tanh(%785: !torch.tensor) -> (!torch.tensor) {
    %int1 = torch.constant.int 1
    %int3 = torch.constant.int 3
    %68 = torch.tensor.literal(dense<5.000000e-01> : tensor<f64>) : !torch.tensor<[],f64>
    %69 = torch.tensor.literal(dense<4.471500e-02> : tensor<f64>) : !torch.tensor<[],f64>
    %70 = torch.tensor.literal(dense<0.79788456080286541> : tensor<f64>) : !torch.tensor<[],f64>
    %71 = torch.tensor.literal(dense<1> : tensor<si64>) : !torch.tensor<[],si64>
    %786 = torch.aten.mul.Tensor %785, %68 : !torch.tensor, !torch.tensor<[],f64> -> !torch.tensor
    %787 = torch.aten.pow.Tensor_Scalar %785, %int3 : !torch.tensor, !torch.int -> !torch.tensor
    %788 = torch.aten.mul.Tensor %787, %69 : !torch.tensor, !torch.tensor<[],f64> -> !torch.tensor
    %789 = torch.aten.add.Tensor %785, %788, %int1 : !torch.tensor, !torch.tensor, !torch.int -> !torch.tensor
    %790 = torch.aten.mul.Tensor %789, %70 : !torch.tensor, !torch.tensor<[],f64> -> !torch.tensor
    %791 = torch.aten.tanh %790 : !torch.tensor -> !torch.tensor
    %792 = torch.aten.add.Tensor %791, %71, %int1 : !torch.tensor, !torch.tensor<[],si64>, !torch.int -> !torch.tensor
    %793 = torch.aten.mul.Tensor %786, %792 : !torch.tensor, !torch.tensor -> !torch.tensor
    return %793 : !torch.tensor
}
// CHECK-LABEL: @torch.gelu.tanh
// CHECK:  torch.constant.str "tanh"
// CHECK:  torch.aten.gelu

func.func @torch.gelu.erf(%880: !torch.tensor) -> !torch.tensor {
    %int1 = torch.constant.int 1
    %876 = torch.tensor.literal(dense<1.000000e+00> : tensor<f64>) : !torch.tensor<[],f64>
    %877 = torch.tensor.literal(dense<1.4142135623730951> : tensor<f64>) : !torch.tensor<[],f64>
    %878 = torch.tensor.literal(dense<5.000000e-01> : tensor<f64>) : !torch.tensor<[],f64>
    %881 = torch.aten.mul.Tensor %880, %878 : !torch.tensor, !torch.tensor<[],f64> -> !torch.tensor
    %882 = torch.aten.div.Tensor %880, %877 : !torch.tensor, !torch.tensor<[],f64> -> !torch.tensor
    %883 = torch.aten.erf %882 : !torch.tensor -> !torch.tensor
    %884 = torch.aten.add.Tensor %883, %876, %int1 : !torch.tensor, !torch.tensor<[],f64>, !torch.int -> !torch.tensor
    %885 = torch.aten.mul.Tensor %881, %884 : !torch.tensor, !torch.tensor -> !torch.tensor
    return %885 : !torch.tensor
}
// CHECK-LABEL: @torch.gelu.erf
// CHECK:  torch.constant.str "none"
// CHECK:  torch.aten.gelu

func.func @torch.layer_norm(%861: !torch.tensor) -> (!torch.tensor) {
    %none = torch.constant.none
    %true = torch.constant.bool true
    %false = torch.constant.bool false
    %int-1 = torch.constant.int -1
    %int1 = torch.constant.int 1
    %463 = torch.tensor.literal(dense_resource<__elided__> : tensor<768xf32>) : !torch.tensor<[768],f32>
    %464 = torch.tensor.literal(dense_resource<__elided__> : tensor<768xf32>) : !torch.tensor<[768],f32>
    %524 = torch.tensor.literal(dense<9.9999999999999998E-13> : tensor<f64>) : !torch.tensor<[],f64>
    %862 = torch.prim.ListConstruct %int-1 : (!torch.int) -> !torch.list<int>
    %863 = torch.aten.mean.dim %861, %862, %true, %none : !torch.tensor, !torch.list<int>, !torch.bool, !torch.none -> !torch.tensor
    %864 = torch.prim.ListConstruct %int-1 : (!torch.int) -> !torch.list<int>
    %865 = torch.aten.std.dim %861, %864, %false, %true : !torch.tensor, !torch.list<int>, !torch.bool, !torch.bool -> !torch.tensor
    %866 = torch.aten.sub.Tensor %861, %863, %int1 : !torch.tensor, !torch.tensor, !torch.int -> !torch.tensor
    %867 = torch.aten.mul.Tensor %464, %866 : !torch.tensor<[768],f32>, !torch.tensor -> !torch.tensor
    %868 = torch.aten.add.Tensor %865, %524, %int1 : !torch.tensor, !torch.tensor<[],f64>, !torch.int -> !torch.tensor
    %869 = torch.aten.div.Tensor %867, %868 : !torch.tensor, !torch.tensor -> !torch.tensor
    %870 = torch.aten.add.Tensor %869, %463, %int1 : !torch.tensor, !torch.tensor<[768],f32>, !torch.int -> !torch.tensor
    return %870 : !torch.tensor
}
// CHECK-LABEL: @torch.layer_norm
// CHECK:  torch.aten.layer_norm
// CHECK-SAME: eps_outside_sqrt = true

func.func @byteir.l2_norm(%arg0: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,4],f32> {
    %none = torch.constant.none
    %float2.000000e00 = torch.constant.float 2.000000e+00
    %int1 = torch.constant.int 1
    %float9.999990e-13 = torch.constant.float 9.9999999999999998E-13
    %true = torch.constant.bool true
    %0 = torch.prim.ListConstruct %int1 : (!torch.int) -> !torch.list<int>
    %1 = torch.aten.linalg_vector_norm %arg0, %float2.000000e00, %0, %true, %none : !torch.vtensor<[3,4],f32>, !torch.float, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[3,1],f32>
    %2 = torch.aten.clamp %1, %float9.999990e-13, %none : !torch.vtensor<[3,1],f32>, !torch.float, !torch.none -> !torch.vtensor<[3,1],f32>
    %3 = torch.aten.expand_as %2, %arg0 : !torch.vtensor<[3,1],f32>, !torch.vtensor<[3,4],f32> -> !torch.vtensor<[3,4],f32>
    %4 = torch.aten.div.Tensor %arg0, %3 : !torch.vtensor<[3,4],f32>, !torch.vtensor<[3,4],f32> -> !torch.vtensor<[3,4],f32>
    return %4 : !torch.vtensor<[3,4],f32>
}
// CHECK-LBAEL: @byteir.l2_norm
// CHECK:  torch.operator "byteir.l2_norm"

func.func @byteir.l2_norm.p_int(%arg0: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,4],f32> {
    %none = torch.constant.none
    %int2 = torch.constant.int 2
    %int1 = torch.constant.int 1
    %float9.999990e-13 = torch.constant.float 9.9999999999999998E-13
    %true = torch.constant.bool true
    %0 = torch.prim.ListConstruct %int1 : (!torch.int) -> !torch.list<int>
    %1 = torch.aten.linalg_vector_norm %arg0, %int2, %0, %true, %none : !torch.vtensor<[3,4],f32>, !torch.int, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[3,1],f32>
    %2 = torch.aten.clamp %1, %float9.999990e-13, %none : !torch.vtensor<[3,1],f32>, !torch.float, !torch.none -> !torch.vtensor<[3,1],f32>
    %3 = torch.aten.expand_as %2, %arg0 : !torch.vtensor<[3,1],f32>, !torch.vtensor<[3,4],f32> -> !torch.vtensor<[3,4],f32>
    %4 = torch.aten.div.Tensor %arg0, %3 : !torch.vtensor<[3,4],f32>, !torch.vtensor<[3,4],f32> -> !torch.vtensor<[3,4],f32>
    return %4 : !torch.vtensor<[3,4],f32>
}
// CHECK-LBAEL: @byteir.l2_norm.p_int
// CHECK:  torch.operator "byteir.l2_norm"
