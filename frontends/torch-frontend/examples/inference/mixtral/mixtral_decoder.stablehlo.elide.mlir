module {
  func.func @main(%arg0: tensor<131072x2xf32>, %arg1: tensor<131072x2xf32>, %arg2: tensor<5x7x32xf32>) -> tensor<5x7x32xf32> {
    %cst = stablehlo.constant dense<3.200000e+01> : tensor<f64>
    %cst_0 = stablehlo.constant dense<8.000000e+00> : tensor<f64>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<35x32xf32>
    %cst_3 = stablehlo.constant dense<1.000000e+00> : tensor<1xf32>
    %cst_4 = stablehlo.constant dense<9.99999974E-6> : tensor<1xf32>
    %cst_5 = stablehlo.constant dense<3.200000e+01> : tensor<1xf32>
    %c = stablehlo.constant dense<0> : tensor<1xi64>
    %c_6 = stablehlo.constant dense<1> : tensor<1xi64>
    %c_7 = stablehlo.constant dense<0> : tensor<32xi64>
    %c_8 = stablehlo.constant dense<1> : tensor<32xi64>
    %c_9 = stablehlo.constant dense<0> : tensor<8xi64>
    %c_10 = stablehlo.constant dense<1> : tensor<8xi64>
    %cst_11 = stablehlo.constant dense_resource<torch_tensor_32_14336_torch.float32_7> : tensor<32x14336xf32>
    %cst_12 = stablehlo.constant dense_resource<torch_tensor_14336_32_torch.float32_15> : tensor<14336x32xf32>
    %cst_13 = stablehlo.constant dense_resource<torch_tensor_14336_32_torch.float32_14> : tensor<14336x32xf32>
    %cst_14 = stablehlo.constant dense_resource<torch_tensor_32_14336_torch.float32_6> : tensor<32x14336xf32>
    %cst_15 = stablehlo.constant dense_resource<torch_tensor_14336_32_torch.float32_13> : tensor<14336x32xf32>
    %cst_16 = stablehlo.constant dense_resource<torch_tensor_14336_32_torch.float32_12> : tensor<14336x32xf32>
    %cst_17 = stablehlo.constant dense_resource<torch_tensor_32_14336_torch.float32_5> : tensor<32x14336xf32>
    %cst_18 = stablehlo.constant dense_resource<torch_tensor_14336_32_torch.float32_11> : tensor<14336x32xf32>
    %cst_19 = stablehlo.constant dense_resource<torch_tensor_14336_32_torch.float32_10> : tensor<14336x32xf32>
    %cst_20 = stablehlo.constant dense_resource<torch_tensor_32_14336_torch.float32_4> : tensor<32x14336xf32>
    %cst_21 = stablehlo.constant dense_resource<torch_tensor_14336_32_torch.float32_9> : tensor<14336x32xf32>
    %cst_22 = stablehlo.constant dense_resource<torch_tensor_14336_32_torch.float32_8> : tensor<14336x32xf32>
    %cst_23 = stablehlo.constant dense_resource<torch_tensor_32_14336_torch.float32_3> : tensor<32x14336xf32>
    %cst_24 = stablehlo.constant dense_resource<torch_tensor_14336_32_torch.float32_7> : tensor<14336x32xf32>
    %cst_25 = stablehlo.constant dense_resource<torch_tensor_14336_32_torch.float32_6> : tensor<14336x32xf32>
    %cst_26 = stablehlo.constant dense_resource<torch_tensor_32_14336_torch.float32_2> : tensor<32x14336xf32>
    %cst_27 = stablehlo.constant dense_resource<torch_tensor_14336_32_torch.float32_5> : tensor<14336x32xf32>
    %cst_28 = stablehlo.constant dense_resource<torch_tensor_14336_32_torch.float32_4> : tensor<14336x32xf32>
    %cst_29 = stablehlo.constant dense_resource<torch_tensor_32_14336_torch.float32_1> : tensor<32x14336xf32>
    %cst_30 = stablehlo.constant dense_resource<torch_tensor_14336_32_torch.float32_3> : tensor<14336x32xf32>
    %cst_31 = stablehlo.constant dense_resource<torch_tensor_14336_32_torch.float32_2> : tensor<14336x32xf32>
    %cst_32 = stablehlo.constant dense_resource<torch_tensor_32_14336_torch.float32> : tensor<32x14336xf32>
    %cst_33 = stablehlo.constant dense_resource<torch_tensor_14336_32_torch.float32_1> : tensor<14336x32xf32>
    %cst_34 = stablehlo.constant dense_resource<torch_tensor_14336_32_torch.float32> : tensor<14336x32xf32>
    %cst_35 = stablehlo.constant dense_resource<torch_tensor_8_32_torch.float32_2> : tensor<8x32xf32>
    %cst_36 = stablehlo.constant dense_resource<torch_tensor_32_torch.float32_1> : tensor<32xf32>
    %cst_37 = stablehlo.constant dense_resource<torch_tensor_32_32_torch.float32_1> : tensor<32x32xf32>
    %cst_38 = stablehlo.constant dense_resource<torch_tensor_8_32_torch.float32_1> : tensor<8x32xf32>
    %cst_39 = stablehlo.constant dense_resource<torch_tensor_8_32_torch.float32> : tensor<8x32xf32>
    %cst_40 = stablehlo.constant dense_resource<torch_tensor_32_32_torch.float32> : tensor<32x32xf32>
    %cst_41 = stablehlo.constant dense_resource<torch_tensor_32_torch.float32> : tensor<32xf32>
    %c1_i64 = arith.constant 1 : i64
    %cst_42 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c_43 = stablehlo.constant dense<0> : tensor<2xi64>
    %c_44 = stablehlo.constant dense<1> : tensor<2xi64>
    %c_45 = stablehlo.constant dense<[-1, 32]> : tensor<2xi64>
    %c2_i64 = arith.constant 2 : i64
    %c_46 = stablehlo.constant dense<[0, 1]> : tensor<2xi64>
    %c32_i64 = arith.constant 32 : i64
    %c32 = arith.constant 32 : index
    %cst_47 = stablehlo.constant dense<2.000000e+00> : tensor<1xf32>
    %0 = stablehlo.reshape %cst_47 : (tensor<1xf32>) -> tensor<f32>
    %1 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1, 2] : (tensor<5x7x32xf32>) -> tensor<5x7x32xf32>
    %2 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<5x7x32xf32>
    %3 = stablehlo.power %1, %2 : tensor<5x7x32xf32>
    %4 = stablehlo.reduce(%3 init: %cst_42) applies stablehlo.add across dimensions = [2] : (tensor<5x7x32xf32>, tensor<f32>) -> tensor<5x7xf32>
    %5 = stablehlo.reshape %4 : (tensor<5x7xf32>) -> tensor<5x7x1xf32>
    %6 = stablehlo.reshape %cst_5 : (tensor<1xf32>) -> tensor<f32>
    %7 = stablehlo.broadcast_in_dim %5, dims = [0, 1, 2] : (tensor<5x7x1xf32>) -> tensor<5x7x1xf32>
    %8 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<f32>) -> tensor<5x7x1xf32>
    %9 = stablehlo.divide %7, %8 : tensor<5x7x1xf32>
    %10 = stablehlo.reshape %cst_4 : (tensor<1xf32>) -> tensor<f32>
    %11 = stablehlo.broadcast_in_dim %9, dims = [0, 1, 2] : (tensor<5x7x1xf32>) -> tensor<5x7x1xf32>
    %12 = stablehlo.broadcast_in_dim %10, dims = [] : (tensor<f32>) -> tensor<5x7x1xf32>
    %13 = stablehlo.add %11, %12 : tensor<5x7x1xf32>
    %14 = stablehlo.rsqrt %13 : tensor<5x7x1xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [0, 1, 2] : (tensor<5x7x1xf32>) -> tensor<5x7x32xf32>
    %16 = stablehlo.multiply %1, %15 : tensor<5x7x32xf32>
    %17 = stablehlo.broadcast_in_dim %16, dims = [0, 1, 2] : (tensor<5x7x32xf32>) -> tensor<5x7x32xf32>
    %18 = stablehlo.broadcast_in_dim %cst_41, dims = [2] : (tensor<32xf32>) -> tensor<5x7x32xf32>
    %19 = stablehlo.multiply %17, %18 : tensor<5x7x32xf32>
    %20 = stablehlo.transpose %cst_40, dims = [1, 0] : (tensor<32x32xf32>) -> tensor<32x32xf32>
    %21 = stablehlo.reshape %19 : (tensor<5x7x32xf32>) -> tensor<35x32xf32>
    %22 = stablehlo.dot %21, %20 : (tensor<35x32xf32>, tensor<32x32xf32>) -> tensor<35x32xf32>
    %23 = stablehlo.reshape %22 : (tensor<35x32xf32>) -> tensor<5x7x32xf32>
    %24 = stablehlo.transpose %cst_39, dims = [1, 0] : (tensor<8x32xf32>) -> tensor<32x8xf32>
    %25 = stablehlo.dot %21, %24 : (tensor<35x32xf32>, tensor<32x8xf32>) -> tensor<35x8xf32>
    %26 = stablehlo.reshape %25 : (tensor<35x8xf32>) -> tensor<5x7x8xf32>
    %27 = stablehlo.transpose %cst_38, dims = [1, 0] : (tensor<8x32xf32>) -> tensor<32x8xf32>
    %28 = stablehlo.dot %21, %27 : (tensor<35x32xf32>, tensor<32x8xf32>) -> tensor<35x8xf32>
    %29 = stablehlo.reshape %28 : (tensor<35x8xf32>) -> tensor<5x7x8xf32>
    %30 = stablehlo.reshape %23 : (tensor<5x7x32xf32>) -> tensor<5x7x32x1xf32>
    %31 = stablehlo.transpose %30, dims = [0, 2, 1, 3] : (tensor<5x7x32x1xf32>) -> tensor<5x32x7x1xf32>
    %32 = stablehlo.reshape %26 : (tensor<5x7x8xf32>) -> tensor<5x7x8x1xf32>
    %33 = stablehlo.transpose %32, dims = [0, 2, 1, 3] : (tensor<5x7x8x1xf32>) -> tensor<5x8x7x1xf32>
    %34 = stablehlo.reshape %29 : (tensor<5x7x8xf32>) -> tensor<5x7x8x1xf32>
    %35 = stablehlo.transpose %34, dims = [0, 2, 1, 3] : (tensor<5x7x8x1xf32>) -> tensor<5x8x7x1xf32>
    %36 = stablehlo.slice %arg0 [0:7, 0:2] : (tensor<131072x2xf32>) -> tensor<7x2xf32>
    %37 = stablehlo.slice %arg1 [0:7, 0:2] : (tensor<131072x2xf32>) -> tensor<7x2xf32>
    %38 = stablehlo.reshape %36 : (tensor<7x2xf32>) -> tensor<1x7x2xf32>
    %39 = stablehlo.reshape %38 : (tensor<1x7x2xf32>) -> tensor<1x1x7x2xf32>
    %40 = stablehlo.reshape %37 : (tensor<7x2xf32>) -> tensor<1x7x2xf32>
    %41 = stablehlo.reshape %40 : (tensor<1x7x2xf32>) -> tensor<1x1x7x2xf32>
    %42 = stablehlo.broadcast_in_dim %31, dims = [0, 1, 2, 3] : (tensor<5x32x7x1xf32>) -> tensor<5x32x7x2xf32>
    %43 = stablehlo.broadcast_in_dim %39, dims = [0, 1, 2, 3] : (tensor<1x1x7x2xf32>) -> tensor<5x32x7x2xf32>
    %44 = stablehlo.multiply %42, %43 : tensor<5x32x7x2xf32>
    %45 = stablehlo.negate %31 : tensor<5x32x7x1xf32>
    %46 = stablehlo.broadcast_in_dim %45, dims = [0, 1, 2, 3] : (tensor<5x32x7x1xf32>) -> tensor<5x32x7x2xf32>
    %47 = stablehlo.broadcast_in_dim %41, dims = [0, 1, 2, 3] : (tensor<1x1x7x2xf32>) -> tensor<5x32x7x2xf32>
    %48 = stablehlo.multiply %46, %47 : tensor<5x32x7x2xf32>
    %49 = stablehlo.add %44, %48 : tensor<5x32x7x2xf32>
    %50 = stablehlo.broadcast_in_dim %33, dims = [0, 1, 2, 3] : (tensor<5x8x7x1xf32>) -> tensor<5x8x7x2xf32>
    %51 = stablehlo.broadcast_in_dim %39, dims = [0, 1, 2, 3] : (tensor<1x1x7x2xf32>) -> tensor<5x8x7x2xf32>
    %52 = stablehlo.multiply %50, %51 : tensor<5x8x7x2xf32>
    %53 = stablehlo.negate %33 : tensor<5x8x7x1xf32>
    %54 = stablehlo.broadcast_in_dim %53, dims = [0, 1, 2, 3] : (tensor<5x8x7x1xf32>) -> tensor<5x8x7x2xf32>
    %55 = stablehlo.broadcast_in_dim %41, dims = [0, 1, 2, 3] : (tensor<1x1x7x2xf32>) -> tensor<5x8x7x2xf32>
    %56 = stablehlo.multiply %54, %55 : tensor<5x8x7x2xf32>
    %57 = stablehlo.add %52, %56 : tensor<5x8x7x2xf32>
    %58 = stablehlo.reshape %57 : (tensor<5x8x7x2xf32>) -> tensor<5x8x1x7x2xf32>
    %59 = stablehlo.broadcast_in_dim %58, dims = [0, 1, 2, 3, 4] : (tensor<5x8x1x7x2xf32>) -> tensor<5x8x4x7x2xf32>
    %60 = stablehlo.reshape %59 : (tensor<5x8x4x7x2xf32>) -> tensor<5x32x7x2xf32>
    %61 = stablehlo.reshape %35 : (tensor<5x8x7x1xf32>) -> tensor<5x8x1x7x1xf32>
    %62 = stablehlo.broadcast_in_dim %61, dims = [0, 1, 2, 3, 4] : (tensor<5x8x1x7x1xf32>) -> tensor<5x8x4x7x1xf32>
    %63 = stablehlo.reshape %62 : (tensor<5x8x4x7x1xf32>) -> tensor<5x32x7x1xf32>
    %64 = stablehlo.transpose %60, dims = [0, 1, 3, 2] : (tensor<5x32x7x2xf32>) -> tensor<5x32x2x7xf32>
    %65 = stablehlo.reshape %49 : (tensor<5x32x7x2xf32>) -> tensor<160x7x2xf32>
    %66 = stablehlo.reshape %64 : (tensor<5x32x2x7xf32>) -> tensor<160x2x7xf32>
    %67 = stablehlo.broadcast_in_dim %66, dims = [0, 1, 2] : (tensor<160x2x7xf32>) -> tensor<160x2x7xf32>
    %68 = stablehlo.dot_general %65, %67, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<160x7x2xf32>, tensor<160x2x7xf32>) -> tensor<160x7x7xf32>
    %69 = stablehlo.reshape %68 : (tensor<160x7x7xf32>) -> tensor<5x32x7x7xf32>
    %70 = stablehlo.reshape %cst_3 : (tensor<1xf32>) -> tensor<f32>
    %71 = stablehlo.broadcast_in_dim %69, dims = [0, 1, 2, 3] : (tensor<5x32x7x7xf32>) -> tensor<5x32x7x7xf32>
    %72 = stablehlo.broadcast_in_dim %70, dims = [] : (tensor<f32>) -> tensor<5x32x7x7xf32>
    %73 = stablehlo.divide %71, %72 : tensor<5x32x7x7xf32>
    %74 = stablehlo.custom_call @byteir.softmax(%73) {byteir_attrs = {axis = 3 : i64}} : (tensor<5x32x7x7xf32>) -> tensor<5x32x7x7xf32>
    %75 = stablehlo.reshape %74 : (tensor<5x32x7x7xf32>) -> tensor<160x7x7xf32>
    %76 = stablehlo.reshape %63 : (tensor<5x32x7x1xf32>) -> tensor<160x7x1xf32>
    %77 = stablehlo.broadcast_in_dim %76, dims = [0, 1, 2] : (tensor<160x7x1xf32>) -> tensor<160x7x1xf32>
    %78 = stablehlo.dot_general %75, %77, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<160x7x7xf32>, tensor<160x7x1xf32>) -> tensor<160x7x1xf32>
    %79 = stablehlo.reshape %78 : (tensor<160x7x1xf32>) -> tensor<5x32x7x1xf32>
    %80 = stablehlo.transpose %79, dims = [0, 2, 1, 3] : (tensor<5x32x7x1xf32>) -> tensor<5x7x32x1xf32>
    %81 = stablehlo.reshape %80 : (tensor<5x7x32x1xf32>) -> tensor<5x7x32xf32>
    %82 = stablehlo.transpose %cst_37, dims = [1, 0] : (tensor<32x32xf32>) -> tensor<32x32xf32>
    %83 = stablehlo.reshape %81 : (tensor<5x7x32xf32>) -> tensor<35x32xf32>
    %84 = stablehlo.dot %83, %82 : (tensor<35x32xf32>, tensor<32x32xf32>) -> tensor<35x32xf32>
    %85 = stablehlo.reshape %84 : (tensor<35x32xf32>) -> tensor<5x7x32xf32>
    %86 = stablehlo.add %arg2, %85 : tensor<5x7x32xf32>
    %87 = stablehlo.broadcast_in_dim %86, dims = [0, 1, 2] : (tensor<5x7x32xf32>) -> tensor<5x7x32xf32>
    %88 = stablehlo.power %87, %2 : tensor<5x7x32xf32>
    %89 = stablehlo.reduce(%88 init: %cst_42) applies stablehlo.add across dimensions = [2] : (tensor<5x7x32xf32>, tensor<f32>) -> tensor<5x7xf32>
    %90 = stablehlo.reshape %89 : (tensor<5x7xf32>) -> tensor<5x7x1xf32>
    %91 = stablehlo.broadcast_in_dim %90, dims = [0, 1, 2] : (tensor<5x7x1xf32>) -> tensor<5x7x1xf32>
    %92 = stablehlo.divide %91, %8 : tensor<5x7x1xf32>
    %93 = stablehlo.broadcast_in_dim %92, dims = [0, 1, 2] : (tensor<5x7x1xf32>) -> tensor<5x7x1xf32>
    %94 = stablehlo.add %93, %12 : tensor<5x7x1xf32>
    %95 = stablehlo.rsqrt %94 : tensor<5x7x1xf32>
    %96 = stablehlo.broadcast_in_dim %95, dims = [0, 1, 2] : (tensor<5x7x1xf32>) -> tensor<5x7x32xf32>
    %97 = stablehlo.multiply %87, %96 : tensor<5x7x32xf32>
    %98 = stablehlo.broadcast_in_dim %97, dims = [0, 1, 2] : (tensor<5x7x32xf32>) -> tensor<5x7x32xf32>
    %99 = stablehlo.broadcast_in_dim %cst_36, dims = [2] : (tensor<32xf32>) -> tensor<5x7x32xf32>
    %100 = stablehlo.multiply %98, %99 : tensor<5x7x32xf32>
    %101 = stablehlo.reshape %100 : (tensor<5x7x32xf32>) -> tensor<35x32xf32>
    %102 = stablehlo.transpose %cst_35, dims = [1, 0] : (tensor<8x32xf32>) -> tensor<32x8xf32>
    %103 = stablehlo.dot %101, %102 : (tensor<35x32xf32>, tensor<32x8xf32>) -> tensor<35x8xf32>
    %104 = stablehlo.custom_call @byteir.softmax(%103) {byteir_attrs = {axis = 1 : i64}} : (tensor<35x8xf32>) -> tensor<35x8xf32>
    %105:2 = stablehlo.custom_call @byteir.top_k(%104) {byteir_attrs = {axis = [1], k = 2 : i64, sorted = true}} : (tensor<35x8xf32>) -> (tensor<35x2xf32>, tensor<35x2xi64>)
    %106 = stablehlo.reduce(%105#0 init: %cst_42) applies stablehlo.add across dimensions = [1] : (tensor<35x2xf32>, tensor<f32>) -> tensor<35xf32>
    %107 = stablehlo.reshape %106 : (tensor<35xf32>) -> tensor<35x1xf32>
    %108 = stablehlo.broadcast_in_dim %105#0, dims = [0, 1] : (tensor<35x2xf32>) -> tensor<35x2xf32>
    %109 = stablehlo.broadcast_in_dim %107, dims = [0, 1] : (tensor<35x1xf32>) -> tensor<35x2xf32>
    %110 = stablehlo.divide %108, %109 : tensor<35x2xf32>
    %111 = stablehlo.divide %cst_0, %cst_1 : tensor<f64>
    %112 = stablehlo.ceil %111 : tensor<f64>
    %113 = stablehlo.convert %112 : (tensor<f64>) -> tensor<i64>
    %114 = stablehlo.reshape %113 : (tensor<i64>) -> tensor<1xi64>
    %115 = stablehlo.dynamic_iota %114, dim = 0 : (tensor<1xi64>) -> tensor<8xi64>
    %116 = stablehlo.broadcast_in_dim %115, dims = [0] : (tensor<8xi64>) -> tensor<8xi64>
    %117 = stablehlo.multiply %116, %c_10 : tensor<8xi64>
    %118 = stablehlo.broadcast_in_dim %117, dims = [0] : (tensor<8xi64>) -> tensor<8xi64>
    %119 = stablehlo.add %118, %c_9 : tensor<8xi64>
    %120 = stablehlo.reshape %105#1 : (tensor<35x2xi64>) -> tensor<35x2x1xi64>
    %121 = stablehlo.broadcast_in_dim %120, dims = [0, 1, 2] : (tensor<35x2x1xi64>) -> tensor<35x2x8xi64>
    %122 = stablehlo.broadcast_in_dim %119, dims = [2] : (tensor<8xi64>) -> tensor<35x2x8xi64>
    %123 = stablehlo.compare  EQ, %121, %122,  SIGNED : (tensor<35x2x8xi64>, tensor<35x2x8xi64>) -> tensor<35x2x8xi1>
    %124 = stablehlo.convert %123 : (tensor<35x2x8xi1>) -> tensor<35x2x8xi64>
    %125 = stablehlo.transpose %124, dims = [2, 1, 0] : (tensor<35x2x8xi64>) -> tensor<8x2x35xi64>
    %126 = stablehlo.slice %125 [0:1, 0:2, 0:35] : (tensor<8x2x35xi64>) -> tensor<1x2x35xi64>
    %127 = stablehlo.reshape %126 : (tensor<1x2x35xi64>) -> tensor<2x35xi64>
    %128 = stablehlo.custom_call @byteir.non_zero(%127) {byteir_attrs = {}} : (tensor<2x35xi64>) -> tensor<?x2xi64>
    %dim = tensor.dim %128, %c0 : tensor<?x2xi64>
    %129 = arith.index_cast %dim : index to i64
    %from_elements = tensor.from_elements %129, %c1_i64 : tensor<2xi64>
    %130 = stablehlo.real_dynamic_slice %128, %c_43, %from_elements, %c_44 : (tensor<?x2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<?x1xi64>
    %dim_48 = tensor.dim %130, %c0 : tensor<?x1xi64>
    %131 = arith.index_cast %dim_48 : index to i64
    %from_elements_49 = tensor.from_elements %131 : tensor<1xi64>
    %132 = stablehlo.dynamic_reshape %130, %from_elements_49 : (tensor<?x1xi64>, tensor<1xi64>) -> tensor<?xi64>
    %from_elements_50 = tensor.from_elements %129, %c2_i64 : tensor<2xi64>
    %133 = stablehlo.real_dynamic_slice %128, %c_46, %from_elements_50, %c_44 : (tensor<?x2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<?x1xi64>
    %dim_51 = tensor.dim %133, %c0 : tensor<?x1xi64>
    %134 = arith.index_cast %dim_51 : index to i64
    %from_elements_52 = tensor.from_elements %134 : tensor<1xi64>
    %135 = stablehlo.dynamic_reshape %133, %from_elements_52 : (tensor<?x1xi64>, tensor<1xi64>) -> tensor<?xi64>
    %136 = stablehlo.reshape %101 : (tensor<35x32xf32>) -> tensor<1x35x32xf32>
    %137 = stablehlo.divide %cst, %cst_1 : tensor<f64>
    %138 = stablehlo.ceil %137 : tensor<f64>
    %139 = stablehlo.convert %138 : (tensor<f64>) -> tensor<i64>
    %140 = stablehlo.reshape %139 : (tensor<i64>) -> tensor<1xi64>
    %141 = stablehlo.dynamic_iota %140, dim = 0 : (tensor<1xi64>) -> tensor<32xi64>
    %142 = stablehlo.broadcast_in_dim %141, dims = [0] : (tensor<32xi64>) -> tensor<32xi64>
    %143 = stablehlo.multiply %142, %c_8 : tensor<32xi64>
    %144 = stablehlo.broadcast_in_dim %143, dims = [0] : (tensor<32xi64>) -> tensor<32xi64>
    %145 = stablehlo.add %144, %c_7 : tensor<32xi64>
    %dim_53 = tensor.dim %135, %c0 : tensor<?xi64>
    %146 = arith.index_cast %dim_53 : index to i64
    %from_elements_54 = tensor.from_elements %146, %c1_i64 : tensor<2xi64>
    %147 = stablehlo.dynamic_reshape %135, %from_elements_54 : (tensor<?xi64>, tensor<2xi64>) -> tensor<?x1xi64>
    %148 = stablehlo.divide %cst_1, %cst_1 : tensor<f64>
    %149 = stablehlo.ceil %148 : tensor<f64>
    %150 = stablehlo.convert %149 : (tensor<f64>) -> tensor<i64>
    %151 = stablehlo.reshape %150 : (tensor<i64>) -> tensor<1xi64>
    %152 = stablehlo.dynamic_iota %151, dim = 0 : (tensor<1xi64>) -> tensor<1xi64>
    %153 = stablehlo.broadcast_in_dim %152, dims = [0] : (tensor<1xi64>) -> tensor<1xi64>
    %154 = stablehlo.multiply %153, %c_6 : tensor<1xi64>
    %155 = stablehlo.broadcast_in_dim %154, dims = [0] : (tensor<1xi64>) -> tensor<1xi64>
    %156 = stablehlo.add %155, %c : tensor<1xi64>
    %157 = stablehlo.reshape %156 : (tensor<1xi64>) -> tensor<1x1xi64>
    %158 = stablehlo.reshape %157 : (tensor<1x1xi64>) -> tensor<1x1x1xi64>
    %dim_55 = tensor.dim %147, %c0 : tensor<?x1xi64>
    %159 = arith.index_cast %dim_55 : index to i64
    %from_elements_56 = tensor.from_elements %c1_i64, %159, %c32_i64 : tensor<3xi64>
    %160 = stablehlo.dynamic_broadcast_in_dim %158, %from_elements_56, dims = [0, 1, 2] : (tensor<1x1x1xi64>, tensor<3xi64>) -> tensor<1x?x32xi64>
    %dim_57 = tensor.dim %160, %c1 : tensor<1x?x32xi64>
    %161 = arith.index_cast %dim_57 : index to i64
    %from_elements_58 = tensor.from_elements %c1_i64, %161, %c32_i64, %c1_i64 : tensor<4xi64>
    %162 = stablehlo.dynamic_reshape %160, %from_elements_58 : (tensor<1x?x32xi64>, tensor<4xi64>) -> tensor<1x?x32x1xi64>
    %163 = stablehlo.dynamic_broadcast_in_dim %147, %from_elements_56, dims = [1, 2] : (tensor<?x1xi64>, tensor<3xi64>) -> tensor<1x?x32xi64>
    %dim_59 = tensor.dim %163, %c1 : tensor<1x?x32xi64>
    %164 = arith.index_cast %dim_59 : index to i64
    %from_elements_60 = tensor.from_elements %c1_i64, %164, %c32_i64, %c1_i64 : tensor<4xi64>
    %165 = stablehlo.dynamic_reshape %163, %from_elements_60 : (tensor<1x?x32xi64>, tensor<4xi64>) -> tensor<1x?x32x1xi64>
    %166 = stablehlo.dynamic_broadcast_in_dim %145, %from_elements_56, dims = [2] : (tensor<32xi64>, tensor<3xi64>) -> tensor<1x?x32xi64>
    %dim_61 = tensor.dim %166, %c1 : tensor<1x?x32xi64>
    %167 = arith.index_cast %dim_61 : index to i64
    %from_elements_62 = tensor.from_elements %c1_i64, %167, %c32_i64, %c1_i64 : tensor<4xi64>
    %168 = stablehlo.dynamic_reshape %166, %from_elements_62 : (tensor<1x?x32xi64>, tensor<4xi64>) -> tensor<1x?x32x1xi64>
    %169 = stablehlo.concatenate %162, %165, %168, dim = 3 : (tensor<1x?x32x1xi64>, tensor<1x?x32x1xi64>, tensor<1x?x32x1xi64>) -> tensor<1x?x32x3xi64>
    %170 = "stablehlo.gather"(%136, %169) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 3>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<1x35x32xf32>, tensor<1x?x32x3xi64>) -> tensor<1x?x32xf32>
    %171 = shape.shape_of %170 : tensor<1x?x32xf32> -> tensor<3xindex>
    %172 = shape.num_elements %171 : tensor<3xindex> -> index
    %173 = stablehlo.compute_reshape_shape %172, %c_45 : (index, tensor<2xi64>) -> tensor<2xi64>
    %174 = stablehlo.dynamic_reshape %170, %173 : (tensor<1x?x32xf32>, tensor<2xi64>) -> tensor<?x32xf32>
    %175 = stablehlo.transpose %cst_34, dims = [1, 0] : (tensor<14336x32xf32>) -> tensor<32x14336xf32>
    %176 = stablehlo.dot %174, %175 : (tensor<?x32xf32>, tensor<32x14336xf32>) -> tensor<?x14336xf32>
    %177 = stablehlo.logistic %176 : tensor<?x14336xf32>
    %178 = shape.shape_of %177 : tensor<?x14336xf32> -> tensor<2xindex>
    %179 = shape.shape_of %176 : tensor<?x14336xf32> -> tensor<2xindex>
    %180 = shape.cstr_broadcastable %178, %179 : tensor<2xindex>, tensor<2xindex>
    %181 = shape.assuming %180 -> (tensor<?x14336xf32>) {
      %695 = shape.broadcast %178, %179 : tensor<2xindex>, tensor<2xindex> -> tensor<2xindex>
      %696 = stablehlo.dynamic_broadcast_in_dim %177, %695, dims = [0, 1] : (tensor<?x14336xf32>, tensor<2xindex>) -> tensor<?x14336xf32>
      %697 = stablehlo.dynamic_broadcast_in_dim %176, %695, dims = [0, 1] : (tensor<?x14336xf32>, tensor<2xindex>) -> tensor<?x14336xf32>
      %698 = stablehlo.multiply %696, %697 : tensor<?x14336xf32>
      shape.assuming_yield %698 : tensor<?x14336xf32>
    }
    %182 = stablehlo.transpose %cst_33, dims = [1, 0] : (tensor<14336x32xf32>) -> tensor<32x14336xf32>
    %183 = stablehlo.dot %174, %182 : (tensor<?x32xf32>, tensor<32x14336xf32>) -> tensor<?x14336xf32>
    %184 = shape.shape_of %181 : tensor<?x14336xf32> -> tensor<2xindex>
    %185 = shape.shape_of %183 : tensor<?x14336xf32> -> tensor<2xindex>
    %186 = shape.cstr_broadcastable %184, %185 : tensor<2xindex>, tensor<2xindex>
    %187 = shape.assuming %186 -> (tensor<?x14336xf32>) {
      %695 = shape.broadcast %184, %185 : tensor<2xindex>, tensor<2xindex> -> tensor<2xindex>
      %696 = stablehlo.dynamic_broadcast_in_dim %181, %695, dims = [0, 1] : (tensor<?x14336xf32>, tensor<2xindex>) -> tensor<?x14336xf32>
      %697 = stablehlo.dynamic_broadcast_in_dim %183, %695, dims = [0, 1] : (tensor<?x14336xf32>, tensor<2xindex>) -> tensor<?x14336xf32>
      %698 = stablehlo.multiply %696, %697 : tensor<?x14336xf32>
      shape.assuming_yield %698 : tensor<?x14336xf32>
    }
    %188 = stablehlo.transpose %cst_32, dims = [1, 0] : (tensor<32x14336xf32>) -> tensor<14336x32xf32>
    %189 = stablehlo.dot %187, %188 : (tensor<?x14336xf32>, tensor<14336x32xf32>) -> tensor<?x32xf32>
    %190 = stablehlo.reshape %110 : (tensor<35x2xf32>) -> tensor<35x2x1xf32>
    %dim_63 = tensor.dim %135, %c0 : tensor<?xi64>
    %191 = arith.index_cast %dim_63 : index to i64
    %from_elements_64 = tensor.from_elements %191, %c1_i64 : tensor<2xi64>
    %192 = stablehlo.dynamic_reshape %135, %from_elements_64 : (tensor<?xi64>, tensor<2xi64>) -> tensor<?x1xi64>
    %dim_65 = tensor.dim %132, %c0 : tensor<?xi64>
    %193 = arith.index_cast %dim_65 : index to i64
    %from_elements_66 = tensor.from_elements %193, %c1_i64 : tensor<2xi64>
    %194 = stablehlo.dynamic_reshape %132, %from_elements_66 : (tensor<?xi64>, tensor<2xi64>) -> tensor<?x1xi64>
    %195 = stablehlo.concatenate %192, %194, dim = 1 : (tensor<?x1xi64>, tensor<?x1xi64>) -> tensor<?x2xi64>
    %196 = "stablehlo.gather"(%190, %195) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<35x2x1xf32>, tensor<?x2xi64>) -> tensor<?x1xf32>
    %197 = shape.shape_of %189 : tensor<?x32xf32> -> tensor<2xindex>
    %198 = shape.shape_of %196 : tensor<?x1xf32> -> tensor<2xindex>
    %199 = shape.cstr_broadcastable %197, %198 : tensor<2xindex>, tensor<2xindex>
    %200 = shape.assuming %199 -> (tensor<?x32xf32>) {
      %695 = shape.broadcast %197, %198 : tensor<2xindex>, tensor<2xindex> -> tensor<2xindex>
      %696 = stablehlo.dynamic_broadcast_in_dim %189, %695, dims = [0, 1] : (tensor<?x32xf32>, tensor<2xindex>) -> tensor<?x32xf32>
      %697 = stablehlo.dynamic_broadcast_in_dim %196, %695, dims = [0, 1] : (tensor<?x1xf32>, tensor<2xindex>) -> tensor<?x32xf32>
      %698 = stablehlo.multiply %696, %697 : tensor<?x32xf32>
      shape.assuming_yield %698 : tensor<?x32xf32>
    }
    %201 = stablehlo.reshape %cst_3 : (tensor<1xf32>) -> tensor<f32>
    %202 = shape.shape_of %200 : tensor<?x32xf32> -> tensor<2xindex>
    %203 = stablehlo.dynamic_broadcast_in_dim %200, %202, dims = [0, 1] : (tensor<?x32xf32>, tensor<2xindex>) -> tensor<?x32xf32>
    %204 = stablehlo.dynamic_broadcast_in_dim %201, %202, dims = [] : (tensor<f32>, tensor<2xindex>) -> tensor<?x32xf32>
    %205 = stablehlo.multiply %203, %204 : tensor<?x32xf32>
    %dim_67 = tensor.dim %147, %c0 : tensor<?x1xi64>
    %206 = arith.index_cast %dim_67 : index to i64
    %dim_68 = tensor.dim %200, %c0 : tensor<?x32xf32>
    %207 = arith.index_cast %dim_68 : index to i64
    %208 = arith.maxsi %206, %207 : i64
    %209 = arith.index_cast %208 : i64 to index
    %from_elements_69 = tensor.from_elements %209, %c32 : tensor<2xindex>
    %210 = stablehlo.dynamic_broadcast_in_dim %147, %from_elements_69, dims = [0, 1] : (tensor<?x1xi64>, tensor<2xindex>) -> tensor<?x32xi64>
    %dim_70 = tensor.dim %210, %c0 : tensor<?x32xi64>
    %211 = arith.index_cast %dim_70 : index to i64
    %from_elements_71 = tensor.from_elements %211, %c32_i64 : tensor<2xi64>
    %212 = stablehlo.real_dynamic_slice %205, %c_43, %from_elements_71, %c_44 : (tensor<?x32xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<?x32xf32>
    %from_elements_72 = tensor.from_elements %211, %c32_i64, %c1_i64 : tensor<3xi64>
    %213 = stablehlo.dynamic_reshape %210, %from_elements_72 : (tensor<?x32xi64>, tensor<3xi64>) -> tensor<?x32x1xi64>
    %214 = stablehlo.dynamic_iota %from_elements_72, dim = 1 : (tensor<3xi64>) -> tensor<?x32x1xi64>
    %215 = stablehlo.concatenate %213, %214, dim = 2 : (tensor<?x32x1xi64>, tensor<?x32x1xi64>) -> tensor<?x32x2xi64>
    %216 = "stablehlo.scatter"(%cst_2, %215, %212) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 2>, unique_indices = false}> ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %695 = stablehlo.add %arg3, %arg4 : tensor<f32>
      stablehlo.return %695 : tensor<f32>
    }) : (tensor<35x32xf32>, tensor<?x32x2xi64>, tensor<?x32xf32>) -> tensor<35x32xf32>
    %217 = stablehlo.slice %125 [1:2, 0:2, 0:35] : (tensor<8x2x35xi64>) -> tensor<1x2x35xi64>
    %218 = stablehlo.reshape %217 : (tensor<1x2x35xi64>) -> tensor<2x35xi64>
    %219 = stablehlo.custom_call @byteir.non_zero(%218) {byteir_attrs = {}} : (tensor<2x35xi64>) -> tensor<?x2xi64>
    %dim_73 = tensor.dim %219, %c0 : tensor<?x2xi64>
    %220 = arith.index_cast %dim_73 : index to i64
    %from_elements_74 = tensor.from_elements %220, %c1_i64 : tensor<2xi64>
    %221 = stablehlo.real_dynamic_slice %219, %c_43, %from_elements_74, %c_44 : (tensor<?x2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<?x1xi64>
    %dim_75 = tensor.dim %221, %c0 : tensor<?x1xi64>
    %222 = arith.index_cast %dim_75 : index to i64
    %from_elements_76 = tensor.from_elements %222 : tensor<1xi64>
    %223 = stablehlo.dynamic_reshape %221, %from_elements_76 : (tensor<?x1xi64>, tensor<1xi64>) -> tensor<?xi64>
    %from_elements_77 = tensor.from_elements %220, %c2_i64 : tensor<2xi64>
    %224 = stablehlo.real_dynamic_slice %219, %c_46, %from_elements_77, %c_44 : (tensor<?x2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<?x1xi64>
    %dim_78 = tensor.dim %224, %c0 : tensor<?x1xi64>
    %225 = arith.index_cast %dim_78 : index to i64
    %from_elements_79 = tensor.from_elements %225 : tensor<1xi64>
    %226 = stablehlo.dynamic_reshape %224, %from_elements_79 : (tensor<?x1xi64>, tensor<1xi64>) -> tensor<?xi64>
    %dim_80 = tensor.dim %226, %c0 : tensor<?xi64>
    %227 = arith.index_cast %dim_80 : index to i64
    %from_elements_81 = tensor.from_elements %227, %c1_i64 : tensor<2xi64>
    %228 = stablehlo.dynamic_reshape %226, %from_elements_81 : (tensor<?xi64>, tensor<2xi64>) -> tensor<?x1xi64>
    %dim_82 = tensor.dim %228, %c0 : tensor<?x1xi64>
    %229 = arith.index_cast %dim_82 : index to i64
    %from_elements_83 = tensor.from_elements %c1_i64, %229, %c32_i64 : tensor<3xi64>
    %230 = stablehlo.dynamic_broadcast_in_dim %158, %from_elements_83, dims = [0, 1, 2] : (tensor<1x1x1xi64>, tensor<3xi64>) -> tensor<1x?x32xi64>
    %dim_84 = tensor.dim %230, %c1 : tensor<1x?x32xi64>
    %231 = arith.index_cast %dim_84 : index to i64
    %from_elements_85 = tensor.from_elements %c1_i64, %231, %c32_i64, %c1_i64 : tensor<4xi64>
    %232 = stablehlo.dynamic_reshape %230, %from_elements_85 : (tensor<1x?x32xi64>, tensor<4xi64>) -> tensor<1x?x32x1xi64>
    %233 = stablehlo.dynamic_broadcast_in_dim %228, %from_elements_83, dims = [1, 2] : (tensor<?x1xi64>, tensor<3xi64>) -> tensor<1x?x32xi64>
    %dim_86 = tensor.dim %233, %c1 : tensor<1x?x32xi64>
    %234 = arith.index_cast %dim_86 : index to i64
    %from_elements_87 = tensor.from_elements %c1_i64, %234, %c32_i64, %c1_i64 : tensor<4xi64>
    %235 = stablehlo.dynamic_reshape %233, %from_elements_87 : (tensor<1x?x32xi64>, tensor<4xi64>) -> tensor<1x?x32x1xi64>
    %236 = stablehlo.dynamic_broadcast_in_dim %145, %from_elements_83, dims = [2] : (tensor<32xi64>, tensor<3xi64>) -> tensor<1x?x32xi64>
    %dim_88 = tensor.dim %236, %c1 : tensor<1x?x32xi64>
    %237 = arith.index_cast %dim_88 : index to i64
    %from_elements_89 = tensor.from_elements %c1_i64, %237, %c32_i64, %c1_i64 : tensor<4xi64>
    %238 = stablehlo.dynamic_reshape %236, %from_elements_89 : (tensor<1x?x32xi64>, tensor<4xi64>) -> tensor<1x?x32x1xi64>
    %239 = stablehlo.concatenate %232, %235, %238, dim = 3 : (tensor<1x?x32x1xi64>, tensor<1x?x32x1xi64>, tensor<1x?x32x1xi64>) -> tensor<1x?x32x3xi64>
    %240 = "stablehlo.gather"(%136, %239) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 3>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<1x35x32xf32>, tensor<1x?x32x3xi64>) -> tensor<1x?x32xf32>
    %241 = shape.shape_of %240 : tensor<1x?x32xf32> -> tensor<3xindex>
    %242 = shape.num_elements %241 : tensor<3xindex> -> index
    %243 = stablehlo.compute_reshape_shape %242, %c_45 : (index, tensor<2xi64>) -> tensor<2xi64>
    %244 = stablehlo.dynamic_reshape %240, %243 : (tensor<1x?x32xf32>, tensor<2xi64>) -> tensor<?x32xf32>
    %245 = stablehlo.transpose %cst_31, dims = [1, 0] : (tensor<14336x32xf32>) -> tensor<32x14336xf32>
    %246 = stablehlo.dot %244, %245 : (tensor<?x32xf32>, tensor<32x14336xf32>) -> tensor<?x14336xf32>
    %247 = stablehlo.logistic %246 : tensor<?x14336xf32>
    %248 = shape.shape_of %247 : tensor<?x14336xf32> -> tensor<2xindex>
    %249 = shape.shape_of %246 : tensor<?x14336xf32> -> tensor<2xindex>
    %250 = shape.cstr_broadcastable %248, %249 : tensor<2xindex>, tensor<2xindex>
    %251 = shape.assuming %250 -> (tensor<?x14336xf32>) {
      %695 = shape.broadcast %248, %249 : tensor<2xindex>, tensor<2xindex> -> tensor<2xindex>
      %696 = stablehlo.dynamic_broadcast_in_dim %247, %695, dims = [0, 1] : (tensor<?x14336xf32>, tensor<2xindex>) -> tensor<?x14336xf32>
      %697 = stablehlo.dynamic_broadcast_in_dim %246, %695, dims = [0, 1] : (tensor<?x14336xf32>, tensor<2xindex>) -> tensor<?x14336xf32>
      %698 = stablehlo.multiply %696, %697 : tensor<?x14336xf32>
      shape.assuming_yield %698 : tensor<?x14336xf32>
    }
    %252 = stablehlo.transpose %cst_30, dims = [1, 0] : (tensor<14336x32xf32>) -> tensor<32x14336xf32>
    %253 = stablehlo.dot %244, %252 : (tensor<?x32xf32>, tensor<32x14336xf32>) -> tensor<?x14336xf32>
    %254 = shape.shape_of %251 : tensor<?x14336xf32> -> tensor<2xindex>
    %255 = shape.shape_of %253 : tensor<?x14336xf32> -> tensor<2xindex>
    %256 = shape.cstr_broadcastable %254, %255 : tensor<2xindex>, tensor<2xindex>
    %257 = shape.assuming %256 -> (tensor<?x14336xf32>) {
      %695 = shape.broadcast %254, %255 : tensor<2xindex>, tensor<2xindex> -> tensor<2xindex>
      %696 = stablehlo.dynamic_broadcast_in_dim %251, %695, dims = [0, 1] : (tensor<?x14336xf32>, tensor<2xindex>) -> tensor<?x14336xf32>
      %697 = stablehlo.dynamic_broadcast_in_dim %253, %695, dims = [0, 1] : (tensor<?x14336xf32>, tensor<2xindex>) -> tensor<?x14336xf32>
      %698 = stablehlo.multiply %696, %697 : tensor<?x14336xf32>
      shape.assuming_yield %698 : tensor<?x14336xf32>
    }
    %258 = stablehlo.transpose %cst_29, dims = [1, 0] : (tensor<32x14336xf32>) -> tensor<14336x32xf32>
    %259 = stablehlo.dot %257, %258 : (tensor<?x14336xf32>, tensor<14336x32xf32>) -> tensor<?x32xf32>
    %dim_90 = tensor.dim %226, %c0 : tensor<?xi64>
    %260 = arith.index_cast %dim_90 : index to i64
    %from_elements_91 = tensor.from_elements %260, %c1_i64 : tensor<2xi64>
    %261 = stablehlo.dynamic_reshape %226, %from_elements_91 : (tensor<?xi64>, tensor<2xi64>) -> tensor<?x1xi64>
    %dim_92 = tensor.dim %223, %c0 : tensor<?xi64>
    %262 = arith.index_cast %dim_92 : index to i64
    %from_elements_93 = tensor.from_elements %262, %c1_i64 : tensor<2xi64>
    %263 = stablehlo.dynamic_reshape %223, %from_elements_93 : (tensor<?xi64>, tensor<2xi64>) -> tensor<?x1xi64>
    %264 = stablehlo.concatenate %261, %263, dim = 1 : (tensor<?x1xi64>, tensor<?x1xi64>) -> tensor<?x2xi64>
    %265 = "stablehlo.gather"(%190, %264) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<35x2x1xf32>, tensor<?x2xi64>) -> tensor<?x1xf32>
    %266 = shape.shape_of %259 : tensor<?x32xf32> -> tensor<2xindex>
    %267 = shape.shape_of %265 : tensor<?x1xf32> -> tensor<2xindex>
    %268 = shape.cstr_broadcastable %266, %267 : tensor<2xindex>, tensor<2xindex>
    %269 = shape.assuming %268 -> (tensor<?x32xf32>) {
      %695 = shape.broadcast %266, %267 : tensor<2xindex>, tensor<2xindex> -> tensor<2xindex>
      %696 = stablehlo.dynamic_broadcast_in_dim %259, %695, dims = [0, 1] : (tensor<?x32xf32>, tensor<2xindex>) -> tensor<?x32xf32>
      %697 = stablehlo.dynamic_broadcast_in_dim %265, %695, dims = [0, 1] : (tensor<?x1xf32>, tensor<2xindex>) -> tensor<?x32xf32>
      %698 = stablehlo.multiply %696, %697 : tensor<?x32xf32>
      shape.assuming_yield %698 : tensor<?x32xf32>
    }
    %270 = shape.shape_of %269 : tensor<?x32xf32> -> tensor<2xindex>
    %271 = stablehlo.dynamic_broadcast_in_dim %269, %270, dims = [0, 1] : (tensor<?x32xf32>, tensor<2xindex>) -> tensor<?x32xf32>
    %272 = stablehlo.dynamic_broadcast_in_dim %201, %270, dims = [] : (tensor<f32>, tensor<2xindex>) -> tensor<?x32xf32>
    %273 = stablehlo.multiply %271, %272 : tensor<?x32xf32>
    %dim_94 = tensor.dim %228, %c0 : tensor<?x1xi64>
    %274 = arith.index_cast %dim_94 : index to i64
    %dim_95 = tensor.dim %269, %c0 : tensor<?x32xf32>
    %275 = arith.index_cast %dim_95 : index to i64
    %276 = arith.maxsi %274, %275 : i64
    %277 = arith.index_cast %276 : i64 to index
    %from_elements_96 = tensor.from_elements %277, %c32 : tensor<2xindex>
    %278 = stablehlo.dynamic_broadcast_in_dim %228, %from_elements_96, dims = [0, 1] : (tensor<?x1xi64>, tensor<2xindex>) -> tensor<?x32xi64>
    %dim_97 = tensor.dim %278, %c0 : tensor<?x32xi64>
    %279 = arith.index_cast %dim_97 : index to i64
    %from_elements_98 = tensor.from_elements %279, %c32_i64 : tensor<2xi64>
    %280 = stablehlo.real_dynamic_slice %273, %c_43, %from_elements_98, %c_44 : (tensor<?x32xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<?x32xf32>
    %from_elements_99 = tensor.from_elements %279, %c32_i64, %c1_i64 : tensor<3xi64>
    %281 = stablehlo.dynamic_reshape %278, %from_elements_99 : (tensor<?x32xi64>, tensor<3xi64>) -> tensor<?x32x1xi64>
    %282 = stablehlo.dynamic_iota %from_elements_99, dim = 1 : (tensor<3xi64>) -> tensor<?x32x1xi64>
    %283 = stablehlo.concatenate %281, %282, dim = 2 : (tensor<?x32x1xi64>, tensor<?x32x1xi64>) -> tensor<?x32x2xi64>
    %284 = "stablehlo.scatter"(%216, %283, %280) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 2>, unique_indices = false}> ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %695 = stablehlo.add %arg3, %arg4 : tensor<f32>
      stablehlo.return %695 : tensor<f32>
    }) : (tensor<35x32xf32>, tensor<?x32x2xi64>, tensor<?x32xf32>) -> tensor<35x32xf32>
    %285 = stablehlo.slice %125 [2:3, 0:2, 0:35] : (tensor<8x2x35xi64>) -> tensor<1x2x35xi64>
    %286 = stablehlo.reshape %285 : (tensor<1x2x35xi64>) -> tensor<2x35xi64>
    %287 = stablehlo.custom_call @byteir.non_zero(%286) {byteir_attrs = {}} : (tensor<2x35xi64>) -> tensor<?x2xi64>
    %dim_100 = tensor.dim %287, %c0 : tensor<?x2xi64>
    %288 = arith.index_cast %dim_100 : index to i64
    %from_elements_101 = tensor.from_elements %288, %c1_i64 : tensor<2xi64>
    %289 = stablehlo.real_dynamic_slice %287, %c_43, %from_elements_101, %c_44 : (tensor<?x2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<?x1xi64>
    %dim_102 = tensor.dim %289, %c0 : tensor<?x1xi64>
    %290 = arith.index_cast %dim_102 : index to i64
    %from_elements_103 = tensor.from_elements %290 : tensor<1xi64>
    %291 = stablehlo.dynamic_reshape %289, %from_elements_103 : (tensor<?x1xi64>, tensor<1xi64>) -> tensor<?xi64>
    %from_elements_104 = tensor.from_elements %288, %c2_i64 : tensor<2xi64>
    %292 = stablehlo.real_dynamic_slice %287, %c_46, %from_elements_104, %c_44 : (tensor<?x2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<?x1xi64>
    %dim_105 = tensor.dim %292, %c0 : tensor<?x1xi64>
    %293 = arith.index_cast %dim_105 : index to i64
    %from_elements_106 = tensor.from_elements %293 : tensor<1xi64>
    %294 = stablehlo.dynamic_reshape %292, %from_elements_106 : (tensor<?x1xi64>, tensor<1xi64>) -> tensor<?xi64>
    %dim_107 = tensor.dim %294, %c0 : tensor<?xi64>
    %295 = arith.index_cast %dim_107 : index to i64
    %from_elements_108 = tensor.from_elements %295, %c1_i64 : tensor<2xi64>
    %296 = stablehlo.dynamic_reshape %294, %from_elements_108 : (tensor<?xi64>, tensor<2xi64>) -> tensor<?x1xi64>
    %dim_109 = tensor.dim %296, %c0 : tensor<?x1xi64>
    %297 = arith.index_cast %dim_109 : index to i64
    %from_elements_110 = tensor.from_elements %c1_i64, %297, %c32_i64 : tensor<3xi64>
    %298 = stablehlo.dynamic_broadcast_in_dim %158, %from_elements_110, dims = [0, 1, 2] : (tensor<1x1x1xi64>, tensor<3xi64>) -> tensor<1x?x32xi64>
    %dim_111 = tensor.dim %298, %c1 : tensor<1x?x32xi64>
    %299 = arith.index_cast %dim_111 : index to i64
    %from_elements_112 = tensor.from_elements %c1_i64, %299, %c32_i64, %c1_i64 : tensor<4xi64>
    %300 = stablehlo.dynamic_reshape %298, %from_elements_112 : (tensor<1x?x32xi64>, tensor<4xi64>) -> tensor<1x?x32x1xi64>
    %301 = stablehlo.dynamic_broadcast_in_dim %296, %from_elements_110, dims = [1, 2] : (tensor<?x1xi64>, tensor<3xi64>) -> tensor<1x?x32xi64>
    %dim_113 = tensor.dim %301, %c1 : tensor<1x?x32xi64>
    %302 = arith.index_cast %dim_113 : index to i64
    %from_elements_114 = tensor.from_elements %c1_i64, %302, %c32_i64, %c1_i64 : tensor<4xi64>
    %303 = stablehlo.dynamic_reshape %301, %from_elements_114 : (tensor<1x?x32xi64>, tensor<4xi64>) -> tensor<1x?x32x1xi64>
    %304 = stablehlo.dynamic_broadcast_in_dim %145, %from_elements_110, dims = [2] : (tensor<32xi64>, tensor<3xi64>) -> tensor<1x?x32xi64>
    %dim_115 = tensor.dim %304, %c1 : tensor<1x?x32xi64>
    %305 = arith.index_cast %dim_115 : index to i64
    %from_elements_116 = tensor.from_elements %c1_i64, %305, %c32_i64, %c1_i64 : tensor<4xi64>
    %306 = stablehlo.dynamic_reshape %304, %from_elements_116 : (tensor<1x?x32xi64>, tensor<4xi64>) -> tensor<1x?x32x1xi64>
    %307 = stablehlo.concatenate %300, %303, %306, dim = 3 : (tensor<1x?x32x1xi64>, tensor<1x?x32x1xi64>, tensor<1x?x32x1xi64>) -> tensor<1x?x32x3xi64>
    %308 = "stablehlo.gather"(%136, %307) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 3>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<1x35x32xf32>, tensor<1x?x32x3xi64>) -> tensor<1x?x32xf32>
    %309 = shape.shape_of %308 : tensor<1x?x32xf32> -> tensor<3xindex>
    %310 = shape.num_elements %309 : tensor<3xindex> -> index
    %311 = stablehlo.compute_reshape_shape %310, %c_45 : (index, tensor<2xi64>) -> tensor<2xi64>
    %312 = stablehlo.dynamic_reshape %308, %311 : (tensor<1x?x32xf32>, tensor<2xi64>) -> tensor<?x32xf32>
    %313 = stablehlo.transpose %cst_28, dims = [1, 0] : (tensor<14336x32xf32>) -> tensor<32x14336xf32>
    %314 = stablehlo.dot %312, %313 : (tensor<?x32xf32>, tensor<32x14336xf32>) -> tensor<?x14336xf32>
    %315 = stablehlo.logistic %314 : tensor<?x14336xf32>
    %316 = shape.shape_of %315 : tensor<?x14336xf32> -> tensor<2xindex>
    %317 = shape.shape_of %314 : tensor<?x14336xf32> -> tensor<2xindex>
    %318 = shape.cstr_broadcastable %316, %317 : tensor<2xindex>, tensor<2xindex>
    %319 = shape.assuming %318 -> (tensor<?x14336xf32>) {
      %695 = shape.broadcast %316, %317 : tensor<2xindex>, tensor<2xindex> -> tensor<2xindex>
      %696 = stablehlo.dynamic_broadcast_in_dim %315, %695, dims = [0, 1] : (tensor<?x14336xf32>, tensor<2xindex>) -> tensor<?x14336xf32>
      %697 = stablehlo.dynamic_broadcast_in_dim %314, %695, dims = [0, 1] : (tensor<?x14336xf32>, tensor<2xindex>) -> tensor<?x14336xf32>
      %698 = stablehlo.multiply %696, %697 : tensor<?x14336xf32>
      shape.assuming_yield %698 : tensor<?x14336xf32>
    }
    %320 = stablehlo.transpose %cst_27, dims = [1, 0] : (tensor<14336x32xf32>) -> tensor<32x14336xf32>
    %321 = stablehlo.dot %312, %320 : (tensor<?x32xf32>, tensor<32x14336xf32>) -> tensor<?x14336xf32>
    %322 = shape.shape_of %319 : tensor<?x14336xf32> -> tensor<2xindex>
    %323 = shape.shape_of %321 : tensor<?x14336xf32> -> tensor<2xindex>
    %324 = shape.cstr_broadcastable %322, %323 : tensor<2xindex>, tensor<2xindex>
    %325 = shape.assuming %324 -> (tensor<?x14336xf32>) {
      %695 = shape.broadcast %322, %323 : tensor<2xindex>, tensor<2xindex> -> tensor<2xindex>
      %696 = stablehlo.dynamic_broadcast_in_dim %319, %695, dims = [0, 1] : (tensor<?x14336xf32>, tensor<2xindex>) -> tensor<?x14336xf32>
      %697 = stablehlo.dynamic_broadcast_in_dim %321, %695, dims = [0, 1] : (tensor<?x14336xf32>, tensor<2xindex>) -> tensor<?x14336xf32>
      %698 = stablehlo.multiply %696, %697 : tensor<?x14336xf32>
      shape.assuming_yield %698 : tensor<?x14336xf32>
    }
    %326 = stablehlo.transpose %cst_26, dims = [1, 0] : (tensor<32x14336xf32>) -> tensor<14336x32xf32>
    %327 = stablehlo.dot %325, %326 : (tensor<?x14336xf32>, tensor<14336x32xf32>) -> tensor<?x32xf32>
    %dim_117 = tensor.dim %294, %c0 : tensor<?xi64>
    %328 = arith.index_cast %dim_117 : index to i64
    %from_elements_118 = tensor.from_elements %328, %c1_i64 : tensor<2xi64>
    %329 = stablehlo.dynamic_reshape %294, %from_elements_118 : (tensor<?xi64>, tensor<2xi64>) -> tensor<?x1xi64>
    %dim_119 = tensor.dim %291, %c0 : tensor<?xi64>
    %330 = arith.index_cast %dim_119 : index to i64
    %from_elements_120 = tensor.from_elements %330, %c1_i64 : tensor<2xi64>
    %331 = stablehlo.dynamic_reshape %291, %from_elements_120 : (tensor<?xi64>, tensor<2xi64>) -> tensor<?x1xi64>
    %332 = stablehlo.concatenate %329, %331, dim = 1 : (tensor<?x1xi64>, tensor<?x1xi64>) -> tensor<?x2xi64>
    %333 = "stablehlo.gather"(%190, %332) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<35x2x1xf32>, tensor<?x2xi64>) -> tensor<?x1xf32>
    %334 = shape.shape_of %327 : tensor<?x32xf32> -> tensor<2xindex>
    %335 = shape.shape_of %333 : tensor<?x1xf32> -> tensor<2xindex>
    %336 = shape.cstr_broadcastable %334, %335 : tensor<2xindex>, tensor<2xindex>
    %337 = shape.assuming %336 -> (tensor<?x32xf32>) {
      %695 = shape.broadcast %334, %335 : tensor<2xindex>, tensor<2xindex> -> tensor<2xindex>
      %696 = stablehlo.dynamic_broadcast_in_dim %327, %695, dims = [0, 1] : (tensor<?x32xf32>, tensor<2xindex>) -> tensor<?x32xf32>
      %697 = stablehlo.dynamic_broadcast_in_dim %333, %695, dims = [0, 1] : (tensor<?x1xf32>, tensor<2xindex>) -> tensor<?x32xf32>
      %698 = stablehlo.multiply %696, %697 : tensor<?x32xf32>
      shape.assuming_yield %698 : tensor<?x32xf32>
    }
    %338 = shape.shape_of %337 : tensor<?x32xf32> -> tensor<2xindex>
    %339 = stablehlo.dynamic_broadcast_in_dim %337, %338, dims = [0, 1] : (tensor<?x32xf32>, tensor<2xindex>) -> tensor<?x32xf32>
    %340 = stablehlo.dynamic_broadcast_in_dim %201, %338, dims = [] : (tensor<f32>, tensor<2xindex>) -> tensor<?x32xf32>
    %341 = stablehlo.multiply %339, %340 : tensor<?x32xf32>
    %dim_121 = tensor.dim %296, %c0 : tensor<?x1xi64>
    %342 = arith.index_cast %dim_121 : index to i64
    %dim_122 = tensor.dim %337, %c0 : tensor<?x32xf32>
    %343 = arith.index_cast %dim_122 : index to i64
    %344 = arith.maxsi %342, %343 : i64
    %345 = arith.index_cast %344 : i64 to index
    %from_elements_123 = tensor.from_elements %345, %c32 : tensor<2xindex>
    %346 = stablehlo.dynamic_broadcast_in_dim %296, %from_elements_123, dims = [0, 1] : (tensor<?x1xi64>, tensor<2xindex>) -> tensor<?x32xi64>
    %dim_124 = tensor.dim %346, %c0 : tensor<?x32xi64>
    %347 = arith.index_cast %dim_124 : index to i64
    %from_elements_125 = tensor.from_elements %347, %c32_i64 : tensor<2xi64>
    %348 = stablehlo.real_dynamic_slice %341, %c_43, %from_elements_125, %c_44 : (tensor<?x32xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<?x32xf32>
    %from_elements_126 = tensor.from_elements %347, %c32_i64, %c1_i64 : tensor<3xi64>
    %349 = stablehlo.dynamic_reshape %346, %from_elements_126 : (tensor<?x32xi64>, tensor<3xi64>) -> tensor<?x32x1xi64>
    %350 = stablehlo.dynamic_iota %from_elements_126, dim = 1 : (tensor<3xi64>) -> tensor<?x32x1xi64>
    %351 = stablehlo.concatenate %349, %350, dim = 2 : (tensor<?x32x1xi64>, tensor<?x32x1xi64>) -> tensor<?x32x2xi64>
    %352 = "stablehlo.scatter"(%284, %351, %348) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 2>, unique_indices = false}> ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %695 = stablehlo.add %arg3, %arg4 : tensor<f32>
      stablehlo.return %695 : tensor<f32>
    }) : (tensor<35x32xf32>, tensor<?x32x2xi64>, tensor<?x32xf32>) -> tensor<35x32xf32>
    %353 = stablehlo.slice %125 [3:4, 0:2, 0:35] : (tensor<8x2x35xi64>) -> tensor<1x2x35xi64>
    %354 = stablehlo.reshape %353 : (tensor<1x2x35xi64>) -> tensor<2x35xi64>
    %355 = stablehlo.custom_call @byteir.non_zero(%354) {byteir_attrs = {}} : (tensor<2x35xi64>) -> tensor<?x2xi64>
    %dim_127 = tensor.dim %355, %c0 : tensor<?x2xi64>
    %356 = arith.index_cast %dim_127 : index to i64
    %from_elements_128 = tensor.from_elements %356, %c1_i64 : tensor<2xi64>
    %357 = stablehlo.real_dynamic_slice %355, %c_43, %from_elements_128, %c_44 : (tensor<?x2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<?x1xi64>
    %dim_129 = tensor.dim %357, %c0 : tensor<?x1xi64>
    %358 = arith.index_cast %dim_129 : index to i64
    %from_elements_130 = tensor.from_elements %358 : tensor<1xi64>
    %359 = stablehlo.dynamic_reshape %357, %from_elements_130 : (tensor<?x1xi64>, tensor<1xi64>) -> tensor<?xi64>
    %from_elements_131 = tensor.from_elements %356, %c2_i64 : tensor<2xi64>
    %360 = stablehlo.real_dynamic_slice %355, %c_46, %from_elements_131, %c_44 : (tensor<?x2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<?x1xi64>
    %dim_132 = tensor.dim %360, %c0 : tensor<?x1xi64>
    %361 = arith.index_cast %dim_132 : index to i64
    %from_elements_133 = tensor.from_elements %361 : tensor<1xi64>
    %362 = stablehlo.dynamic_reshape %360, %from_elements_133 : (tensor<?x1xi64>, tensor<1xi64>) -> tensor<?xi64>
    %dim_134 = tensor.dim %362, %c0 : tensor<?xi64>
    %363 = arith.index_cast %dim_134 : index to i64
    %from_elements_135 = tensor.from_elements %363, %c1_i64 : tensor<2xi64>
    %364 = stablehlo.dynamic_reshape %362, %from_elements_135 : (tensor<?xi64>, tensor<2xi64>) -> tensor<?x1xi64>
    %dim_136 = tensor.dim %364, %c0 : tensor<?x1xi64>
    %365 = arith.index_cast %dim_136 : index to i64
    %from_elements_137 = tensor.from_elements %c1_i64, %365, %c32_i64 : tensor<3xi64>
    %366 = stablehlo.dynamic_broadcast_in_dim %158, %from_elements_137, dims = [0, 1, 2] : (tensor<1x1x1xi64>, tensor<3xi64>) -> tensor<1x?x32xi64>
    %dim_138 = tensor.dim %366, %c1 : tensor<1x?x32xi64>
    %367 = arith.index_cast %dim_138 : index to i64
    %from_elements_139 = tensor.from_elements %c1_i64, %367, %c32_i64, %c1_i64 : tensor<4xi64>
    %368 = stablehlo.dynamic_reshape %366, %from_elements_139 : (tensor<1x?x32xi64>, tensor<4xi64>) -> tensor<1x?x32x1xi64>
    %369 = stablehlo.dynamic_broadcast_in_dim %364, %from_elements_137, dims = [1, 2] : (tensor<?x1xi64>, tensor<3xi64>) -> tensor<1x?x32xi64>
    %dim_140 = tensor.dim %369, %c1 : tensor<1x?x32xi64>
    %370 = arith.index_cast %dim_140 : index to i64
    %from_elements_141 = tensor.from_elements %c1_i64, %370, %c32_i64, %c1_i64 : tensor<4xi64>
    %371 = stablehlo.dynamic_reshape %369, %from_elements_141 : (tensor<1x?x32xi64>, tensor<4xi64>) -> tensor<1x?x32x1xi64>
    %372 = stablehlo.dynamic_broadcast_in_dim %145, %from_elements_137, dims = [2] : (tensor<32xi64>, tensor<3xi64>) -> tensor<1x?x32xi64>
    %dim_142 = tensor.dim %372, %c1 : tensor<1x?x32xi64>
    %373 = arith.index_cast %dim_142 : index to i64
    %from_elements_143 = tensor.from_elements %c1_i64, %373, %c32_i64, %c1_i64 : tensor<4xi64>
    %374 = stablehlo.dynamic_reshape %372, %from_elements_143 : (tensor<1x?x32xi64>, tensor<4xi64>) -> tensor<1x?x32x1xi64>
    %375 = stablehlo.concatenate %368, %371, %374, dim = 3 : (tensor<1x?x32x1xi64>, tensor<1x?x32x1xi64>, tensor<1x?x32x1xi64>) -> tensor<1x?x32x3xi64>
    %376 = "stablehlo.gather"(%136, %375) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 3>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<1x35x32xf32>, tensor<1x?x32x3xi64>) -> tensor<1x?x32xf32>
    %377 = shape.shape_of %376 : tensor<1x?x32xf32> -> tensor<3xindex>
    %378 = shape.num_elements %377 : tensor<3xindex> -> index
    %379 = stablehlo.compute_reshape_shape %378, %c_45 : (index, tensor<2xi64>) -> tensor<2xi64>
    %380 = stablehlo.dynamic_reshape %376, %379 : (tensor<1x?x32xf32>, tensor<2xi64>) -> tensor<?x32xf32>
    %381 = stablehlo.transpose %cst_25, dims = [1, 0] : (tensor<14336x32xf32>) -> tensor<32x14336xf32>
    %382 = stablehlo.dot %380, %381 : (tensor<?x32xf32>, tensor<32x14336xf32>) -> tensor<?x14336xf32>
    %383 = stablehlo.logistic %382 : tensor<?x14336xf32>
    %384 = shape.shape_of %383 : tensor<?x14336xf32> -> tensor<2xindex>
    %385 = shape.shape_of %382 : tensor<?x14336xf32> -> tensor<2xindex>
    %386 = shape.cstr_broadcastable %384, %385 : tensor<2xindex>, tensor<2xindex>
    %387 = shape.assuming %386 -> (tensor<?x14336xf32>) {
      %695 = shape.broadcast %384, %385 : tensor<2xindex>, tensor<2xindex> -> tensor<2xindex>
      %696 = stablehlo.dynamic_broadcast_in_dim %383, %695, dims = [0, 1] : (tensor<?x14336xf32>, tensor<2xindex>) -> tensor<?x14336xf32>
      %697 = stablehlo.dynamic_broadcast_in_dim %382, %695, dims = [0, 1] : (tensor<?x14336xf32>, tensor<2xindex>) -> tensor<?x14336xf32>
      %698 = stablehlo.multiply %696, %697 : tensor<?x14336xf32>
      shape.assuming_yield %698 : tensor<?x14336xf32>
    }
    %388 = stablehlo.transpose %cst_24, dims = [1, 0] : (tensor<14336x32xf32>) -> tensor<32x14336xf32>
    %389 = stablehlo.dot %380, %388 : (tensor<?x32xf32>, tensor<32x14336xf32>) -> tensor<?x14336xf32>
    %390 = shape.shape_of %387 : tensor<?x14336xf32> -> tensor<2xindex>
    %391 = shape.shape_of %389 : tensor<?x14336xf32> -> tensor<2xindex>
    %392 = shape.cstr_broadcastable %390, %391 : tensor<2xindex>, tensor<2xindex>
    %393 = shape.assuming %392 -> (tensor<?x14336xf32>) {
      %695 = shape.broadcast %390, %391 : tensor<2xindex>, tensor<2xindex> -> tensor<2xindex>
      %696 = stablehlo.dynamic_broadcast_in_dim %387, %695, dims = [0, 1] : (tensor<?x14336xf32>, tensor<2xindex>) -> tensor<?x14336xf32>
      %697 = stablehlo.dynamic_broadcast_in_dim %389, %695, dims = [0, 1] : (tensor<?x14336xf32>, tensor<2xindex>) -> tensor<?x14336xf32>
      %698 = stablehlo.multiply %696, %697 : tensor<?x14336xf32>
      shape.assuming_yield %698 : tensor<?x14336xf32>
    }
    %394 = stablehlo.transpose %cst_23, dims = [1, 0] : (tensor<32x14336xf32>) -> tensor<14336x32xf32>
    %395 = stablehlo.dot %393, %394 : (tensor<?x14336xf32>, tensor<14336x32xf32>) -> tensor<?x32xf32>
    %dim_144 = tensor.dim %362, %c0 : tensor<?xi64>
    %396 = arith.index_cast %dim_144 : index to i64
    %from_elements_145 = tensor.from_elements %396, %c1_i64 : tensor<2xi64>
    %397 = stablehlo.dynamic_reshape %362, %from_elements_145 : (tensor<?xi64>, tensor<2xi64>) -> tensor<?x1xi64>
    %dim_146 = tensor.dim %359, %c0 : tensor<?xi64>
    %398 = arith.index_cast %dim_146 : index to i64
    %from_elements_147 = tensor.from_elements %398, %c1_i64 : tensor<2xi64>
    %399 = stablehlo.dynamic_reshape %359, %from_elements_147 : (tensor<?xi64>, tensor<2xi64>) -> tensor<?x1xi64>
    %400 = stablehlo.concatenate %397, %399, dim = 1 : (tensor<?x1xi64>, tensor<?x1xi64>) -> tensor<?x2xi64>
    %401 = "stablehlo.gather"(%190, %400) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<35x2x1xf32>, tensor<?x2xi64>) -> tensor<?x1xf32>
    %402 = shape.shape_of %395 : tensor<?x32xf32> -> tensor<2xindex>
    %403 = shape.shape_of %401 : tensor<?x1xf32> -> tensor<2xindex>
    %404 = shape.cstr_broadcastable %402, %403 : tensor<2xindex>, tensor<2xindex>
    %405 = shape.assuming %404 -> (tensor<?x32xf32>) {
      %695 = shape.broadcast %402, %403 : tensor<2xindex>, tensor<2xindex> -> tensor<2xindex>
      %696 = stablehlo.dynamic_broadcast_in_dim %395, %695, dims = [0, 1] : (tensor<?x32xf32>, tensor<2xindex>) -> tensor<?x32xf32>
      %697 = stablehlo.dynamic_broadcast_in_dim %401, %695, dims = [0, 1] : (tensor<?x1xf32>, tensor<2xindex>) -> tensor<?x32xf32>
      %698 = stablehlo.multiply %696, %697 : tensor<?x32xf32>
      shape.assuming_yield %698 : tensor<?x32xf32>
    }
    %406 = shape.shape_of %405 : tensor<?x32xf32> -> tensor<2xindex>
    %407 = stablehlo.dynamic_broadcast_in_dim %405, %406, dims = [0, 1] : (tensor<?x32xf32>, tensor<2xindex>) -> tensor<?x32xf32>
    %408 = stablehlo.dynamic_broadcast_in_dim %201, %406, dims = [] : (tensor<f32>, tensor<2xindex>) -> tensor<?x32xf32>
    %409 = stablehlo.multiply %407, %408 : tensor<?x32xf32>
    %dim_148 = tensor.dim %364, %c0 : tensor<?x1xi64>
    %410 = arith.index_cast %dim_148 : index to i64
    %dim_149 = tensor.dim %405, %c0 : tensor<?x32xf32>
    %411 = arith.index_cast %dim_149 : index to i64
    %412 = arith.maxsi %410, %411 : i64
    %413 = arith.index_cast %412 : i64 to index
    %from_elements_150 = tensor.from_elements %413, %c32 : tensor<2xindex>
    %414 = stablehlo.dynamic_broadcast_in_dim %364, %from_elements_150, dims = [0, 1] : (tensor<?x1xi64>, tensor<2xindex>) -> tensor<?x32xi64>
    %dim_151 = tensor.dim %414, %c0 : tensor<?x32xi64>
    %415 = arith.index_cast %dim_151 : index to i64
    %from_elements_152 = tensor.from_elements %415, %c32_i64 : tensor<2xi64>
    %416 = stablehlo.real_dynamic_slice %409, %c_43, %from_elements_152, %c_44 : (tensor<?x32xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<?x32xf32>
    %from_elements_153 = tensor.from_elements %415, %c32_i64, %c1_i64 : tensor<3xi64>
    %417 = stablehlo.dynamic_reshape %414, %from_elements_153 : (tensor<?x32xi64>, tensor<3xi64>) -> tensor<?x32x1xi64>
    %418 = stablehlo.dynamic_iota %from_elements_153, dim = 1 : (tensor<3xi64>) -> tensor<?x32x1xi64>
    %419 = stablehlo.concatenate %417, %418, dim = 2 : (tensor<?x32x1xi64>, tensor<?x32x1xi64>) -> tensor<?x32x2xi64>
    %420 = "stablehlo.scatter"(%352, %419, %416) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 2>, unique_indices = false}> ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %695 = stablehlo.add %arg3, %arg4 : tensor<f32>
      stablehlo.return %695 : tensor<f32>
    }) : (tensor<35x32xf32>, tensor<?x32x2xi64>, tensor<?x32xf32>) -> tensor<35x32xf32>
    %421 = stablehlo.slice %125 [4:5, 0:2, 0:35] : (tensor<8x2x35xi64>) -> tensor<1x2x35xi64>
    %422 = stablehlo.reshape %421 : (tensor<1x2x35xi64>) -> tensor<2x35xi64>
    %423 = stablehlo.custom_call @byteir.non_zero(%422) {byteir_attrs = {}} : (tensor<2x35xi64>) -> tensor<?x2xi64>
    %dim_154 = tensor.dim %423, %c0 : tensor<?x2xi64>
    %424 = arith.index_cast %dim_154 : index to i64
    %from_elements_155 = tensor.from_elements %424, %c1_i64 : tensor<2xi64>
    %425 = stablehlo.real_dynamic_slice %423, %c_43, %from_elements_155, %c_44 : (tensor<?x2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<?x1xi64>
    %dim_156 = tensor.dim %425, %c0 : tensor<?x1xi64>
    %426 = arith.index_cast %dim_156 : index to i64
    %from_elements_157 = tensor.from_elements %426 : tensor<1xi64>
    %427 = stablehlo.dynamic_reshape %425, %from_elements_157 : (tensor<?x1xi64>, tensor<1xi64>) -> tensor<?xi64>
    %from_elements_158 = tensor.from_elements %424, %c2_i64 : tensor<2xi64>
    %428 = stablehlo.real_dynamic_slice %423, %c_46, %from_elements_158, %c_44 : (tensor<?x2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<?x1xi64>
    %dim_159 = tensor.dim %428, %c0 : tensor<?x1xi64>
    %429 = arith.index_cast %dim_159 : index to i64
    %from_elements_160 = tensor.from_elements %429 : tensor<1xi64>
    %430 = stablehlo.dynamic_reshape %428, %from_elements_160 : (tensor<?x1xi64>, tensor<1xi64>) -> tensor<?xi64>
    %dim_161 = tensor.dim %430, %c0 : tensor<?xi64>
    %431 = arith.index_cast %dim_161 : index to i64
    %from_elements_162 = tensor.from_elements %431, %c1_i64 : tensor<2xi64>
    %432 = stablehlo.dynamic_reshape %430, %from_elements_162 : (tensor<?xi64>, tensor<2xi64>) -> tensor<?x1xi64>
    %dim_163 = tensor.dim %432, %c0 : tensor<?x1xi64>
    %433 = arith.index_cast %dim_163 : index to i64
    %from_elements_164 = tensor.from_elements %c1_i64, %433, %c32_i64 : tensor<3xi64>
    %434 = stablehlo.dynamic_broadcast_in_dim %158, %from_elements_164, dims = [0, 1, 2] : (tensor<1x1x1xi64>, tensor<3xi64>) -> tensor<1x?x32xi64>
    %dim_165 = tensor.dim %434, %c1 : tensor<1x?x32xi64>
    %435 = arith.index_cast %dim_165 : index to i64
    %from_elements_166 = tensor.from_elements %c1_i64, %435, %c32_i64, %c1_i64 : tensor<4xi64>
    %436 = stablehlo.dynamic_reshape %434, %from_elements_166 : (tensor<1x?x32xi64>, tensor<4xi64>) -> tensor<1x?x32x1xi64>
    %437 = stablehlo.dynamic_broadcast_in_dim %432, %from_elements_164, dims = [1, 2] : (tensor<?x1xi64>, tensor<3xi64>) -> tensor<1x?x32xi64>
    %dim_167 = tensor.dim %437, %c1 : tensor<1x?x32xi64>
    %438 = arith.index_cast %dim_167 : index to i64
    %from_elements_168 = tensor.from_elements %c1_i64, %438, %c32_i64, %c1_i64 : tensor<4xi64>
    %439 = stablehlo.dynamic_reshape %437, %from_elements_168 : (tensor<1x?x32xi64>, tensor<4xi64>) -> tensor<1x?x32x1xi64>
    %440 = stablehlo.dynamic_broadcast_in_dim %145, %from_elements_164, dims = [2] : (tensor<32xi64>, tensor<3xi64>) -> tensor<1x?x32xi64>
    %dim_169 = tensor.dim %440, %c1 : tensor<1x?x32xi64>
    %441 = arith.index_cast %dim_169 : index to i64
    %from_elements_170 = tensor.from_elements %c1_i64, %441, %c32_i64, %c1_i64 : tensor<4xi64>
    %442 = stablehlo.dynamic_reshape %440, %from_elements_170 : (tensor<1x?x32xi64>, tensor<4xi64>) -> tensor<1x?x32x1xi64>
    %443 = stablehlo.concatenate %436, %439, %442, dim = 3 : (tensor<1x?x32x1xi64>, tensor<1x?x32x1xi64>, tensor<1x?x32x1xi64>) -> tensor<1x?x32x3xi64>
    %444 = "stablehlo.gather"(%136, %443) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 3>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<1x35x32xf32>, tensor<1x?x32x3xi64>) -> tensor<1x?x32xf32>
    %445 = shape.shape_of %444 : tensor<1x?x32xf32> -> tensor<3xindex>
    %446 = shape.num_elements %445 : tensor<3xindex> -> index
    %447 = stablehlo.compute_reshape_shape %446, %c_45 : (index, tensor<2xi64>) -> tensor<2xi64>
    %448 = stablehlo.dynamic_reshape %444, %447 : (tensor<1x?x32xf32>, tensor<2xi64>) -> tensor<?x32xf32>
    %449 = stablehlo.transpose %cst_22, dims = [1, 0] : (tensor<14336x32xf32>) -> tensor<32x14336xf32>
    %450 = stablehlo.dot %448, %449 : (tensor<?x32xf32>, tensor<32x14336xf32>) -> tensor<?x14336xf32>
    %451 = stablehlo.logistic %450 : tensor<?x14336xf32>
    %452 = shape.shape_of %451 : tensor<?x14336xf32> -> tensor<2xindex>
    %453 = shape.shape_of %450 : tensor<?x14336xf32> -> tensor<2xindex>
    %454 = shape.cstr_broadcastable %452, %453 : tensor<2xindex>, tensor<2xindex>
    %455 = shape.assuming %454 -> (tensor<?x14336xf32>) {
      %695 = shape.broadcast %452, %453 : tensor<2xindex>, tensor<2xindex> -> tensor<2xindex>
      %696 = stablehlo.dynamic_broadcast_in_dim %451, %695, dims = [0, 1] : (tensor<?x14336xf32>, tensor<2xindex>) -> tensor<?x14336xf32>
      %697 = stablehlo.dynamic_broadcast_in_dim %450, %695, dims = [0, 1] : (tensor<?x14336xf32>, tensor<2xindex>) -> tensor<?x14336xf32>
      %698 = stablehlo.multiply %696, %697 : tensor<?x14336xf32>
      shape.assuming_yield %698 : tensor<?x14336xf32>
    }
    %456 = stablehlo.transpose %cst_21, dims = [1, 0] : (tensor<14336x32xf32>) -> tensor<32x14336xf32>
    %457 = stablehlo.dot %448, %456 : (tensor<?x32xf32>, tensor<32x14336xf32>) -> tensor<?x14336xf32>
    %458 = shape.shape_of %455 : tensor<?x14336xf32> -> tensor<2xindex>
    %459 = shape.shape_of %457 : tensor<?x14336xf32> -> tensor<2xindex>
    %460 = shape.cstr_broadcastable %458, %459 : tensor<2xindex>, tensor<2xindex>
    %461 = shape.assuming %460 -> (tensor<?x14336xf32>) {
      %695 = shape.broadcast %458, %459 : tensor<2xindex>, tensor<2xindex> -> tensor<2xindex>
      %696 = stablehlo.dynamic_broadcast_in_dim %455, %695, dims = [0, 1] : (tensor<?x14336xf32>, tensor<2xindex>) -> tensor<?x14336xf32>
      %697 = stablehlo.dynamic_broadcast_in_dim %457, %695, dims = [0, 1] : (tensor<?x14336xf32>, tensor<2xindex>) -> tensor<?x14336xf32>
      %698 = stablehlo.multiply %696, %697 : tensor<?x14336xf32>
      shape.assuming_yield %698 : tensor<?x14336xf32>
    }
    %462 = stablehlo.transpose %cst_20, dims = [1, 0] : (tensor<32x14336xf32>) -> tensor<14336x32xf32>
    %463 = stablehlo.dot %461, %462 : (tensor<?x14336xf32>, tensor<14336x32xf32>) -> tensor<?x32xf32>
    %dim_171 = tensor.dim %430, %c0 : tensor<?xi64>
    %464 = arith.index_cast %dim_171 : index to i64
    %from_elements_172 = tensor.from_elements %464, %c1_i64 : tensor<2xi64>
    %465 = stablehlo.dynamic_reshape %430, %from_elements_172 : (tensor<?xi64>, tensor<2xi64>) -> tensor<?x1xi64>
    %dim_173 = tensor.dim %427, %c0 : tensor<?xi64>
    %466 = arith.index_cast %dim_173 : index to i64
    %from_elements_174 = tensor.from_elements %466, %c1_i64 : tensor<2xi64>
    %467 = stablehlo.dynamic_reshape %427, %from_elements_174 : (tensor<?xi64>, tensor<2xi64>) -> tensor<?x1xi64>
    %468 = stablehlo.concatenate %465, %467, dim = 1 : (tensor<?x1xi64>, tensor<?x1xi64>) -> tensor<?x2xi64>
    %469 = "stablehlo.gather"(%190, %468) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<35x2x1xf32>, tensor<?x2xi64>) -> tensor<?x1xf32>
    %470 = shape.shape_of %463 : tensor<?x32xf32> -> tensor<2xindex>
    %471 = shape.shape_of %469 : tensor<?x1xf32> -> tensor<2xindex>
    %472 = shape.cstr_broadcastable %470, %471 : tensor<2xindex>, tensor<2xindex>
    %473 = shape.assuming %472 -> (tensor<?x32xf32>) {
      %695 = shape.broadcast %470, %471 : tensor<2xindex>, tensor<2xindex> -> tensor<2xindex>
      %696 = stablehlo.dynamic_broadcast_in_dim %463, %695, dims = [0, 1] : (tensor<?x32xf32>, tensor<2xindex>) -> tensor<?x32xf32>
      %697 = stablehlo.dynamic_broadcast_in_dim %469, %695, dims = [0, 1] : (tensor<?x1xf32>, tensor<2xindex>) -> tensor<?x32xf32>
      %698 = stablehlo.multiply %696, %697 : tensor<?x32xf32>
      shape.assuming_yield %698 : tensor<?x32xf32>
    }
    %474 = shape.shape_of %473 : tensor<?x32xf32> -> tensor<2xindex>
    %475 = stablehlo.dynamic_broadcast_in_dim %473, %474, dims = [0, 1] : (tensor<?x32xf32>, tensor<2xindex>) -> tensor<?x32xf32>
    %476 = stablehlo.dynamic_broadcast_in_dim %201, %474, dims = [] : (tensor<f32>, tensor<2xindex>) -> tensor<?x32xf32>
    %477 = stablehlo.multiply %475, %476 : tensor<?x32xf32>
    %dim_175 = tensor.dim %432, %c0 : tensor<?x1xi64>
    %478 = arith.index_cast %dim_175 : index to i64
    %dim_176 = tensor.dim %473, %c0 : tensor<?x32xf32>
    %479 = arith.index_cast %dim_176 : index to i64
    %480 = arith.maxsi %478, %479 : i64
    %481 = arith.index_cast %480 : i64 to index
    %from_elements_177 = tensor.from_elements %481, %c32 : tensor<2xindex>
    %482 = stablehlo.dynamic_broadcast_in_dim %432, %from_elements_177, dims = [0, 1] : (tensor<?x1xi64>, tensor<2xindex>) -> tensor<?x32xi64>
    %dim_178 = tensor.dim %482, %c0 : tensor<?x32xi64>
    %483 = arith.index_cast %dim_178 : index to i64
    %from_elements_179 = tensor.from_elements %483, %c32_i64 : tensor<2xi64>
    %484 = stablehlo.real_dynamic_slice %477, %c_43, %from_elements_179, %c_44 : (tensor<?x32xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<?x32xf32>
    %from_elements_180 = tensor.from_elements %483, %c32_i64, %c1_i64 : tensor<3xi64>
    %485 = stablehlo.dynamic_reshape %482, %from_elements_180 : (tensor<?x32xi64>, tensor<3xi64>) -> tensor<?x32x1xi64>
    %486 = stablehlo.dynamic_iota %from_elements_180, dim = 1 : (tensor<3xi64>) -> tensor<?x32x1xi64>
    %487 = stablehlo.concatenate %485, %486, dim = 2 : (tensor<?x32x1xi64>, tensor<?x32x1xi64>) -> tensor<?x32x2xi64>
    %488 = "stablehlo.scatter"(%420, %487, %484) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 2>, unique_indices = false}> ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %695 = stablehlo.add %arg3, %arg4 : tensor<f32>
      stablehlo.return %695 : tensor<f32>
    }) : (tensor<35x32xf32>, tensor<?x32x2xi64>, tensor<?x32xf32>) -> tensor<35x32xf32>
    %489 = stablehlo.slice %125 [5:6, 0:2, 0:35] : (tensor<8x2x35xi64>) -> tensor<1x2x35xi64>
    %490 = stablehlo.reshape %489 : (tensor<1x2x35xi64>) -> tensor<2x35xi64>
    %491 = stablehlo.custom_call @byteir.non_zero(%490) {byteir_attrs = {}} : (tensor<2x35xi64>) -> tensor<?x2xi64>
    %dim_181 = tensor.dim %491, %c0 : tensor<?x2xi64>
    %492 = arith.index_cast %dim_181 : index to i64
    %from_elements_182 = tensor.from_elements %492, %c1_i64 : tensor<2xi64>
    %493 = stablehlo.real_dynamic_slice %491, %c_43, %from_elements_182, %c_44 : (tensor<?x2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<?x1xi64>
    %dim_183 = tensor.dim %493, %c0 : tensor<?x1xi64>
    %494 = arith.index_cast %dim_183 : index to i64
    %from_elements_184 = tensor.from_elements %494 : tensor<1xi64>
    %495 = stablehlo.dynamic_reshape %493, %from_elements_184 : (tensor<?x1xi64>, tensor<1xi64>) -> tensor<?xi64>
    %from_elements_185 = tensor.from_elements %492, %c2_i64 : tensor<2xi64>
    %496 = stablehlo.real_dynamic_slice %491, %c_46, %from_elements_185, %c_44 : (tensor<?x2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<?x1xi64>
    %dim_186 = tensor.dim %496, %c0 : tensor<?x1xi64>
    %497 = arith.index_cast %dim_186 : index to i64
    %from_elements_187 = tensor.from_elements %497 : tensor<1xi64>
    %498 = stablehlo.dynamic_reshape %496, %from_elements_187 : (tensor<?x1xi64>, tensor<1xi64>) -> tensor<?xi64>
    %dim_188 = tensor.dim %498, %c0 : tensor<?xi64>
    %499 = arith.index_cast %dim_188 : index to i64
    %from_elements_189 = tensor.from_elements %499, %c1_i64 : tensor<2xi64>
    %500 = stablehlo.dynamic_reshape %498, %from_elements_189 : (tensor<?xi64>, tensor<2xi64>) -> tensor<?x1xi64>
    %dim_190 = tensor.dim %500, %c0 : tensor<?x1xi64>
    %501 = arith.index_cast %dim_190 : index to i64
    %from_elements_191 = tensor.from_elements %c1_i64, %501, %c32_i64 : tensor<3xi64>
    %502 = stablehlo.dynamic_broadcast_in_dim %158, %from_elements_191, dims = [0, 1, 2] : (tensor<1x1x1xi64>, tensor<3xi64>) -> tensor<1x?x32xi64>
    %dim_192 = tensor.dim %502, %c1 : tensor<1x?x32xi64>
    %503 = arith.index_cast %dim_192 : index to i64
    %from_elements_193 = tensor.from_elements %c1_i64, %503, %c32_i64, %c1_i64 : tensor<4xi64>
    %504 = stablehlo.dynamic_reshape %502, %from_elements_193 : (tensor<1x?x32xi64>, tensor<4xi64>) -> tensor<1x?x32x1xi64>
    %505 = stablehlo.dynamic_broadcast_in_dim %500, %from_elements_191, dims = [1, 2] : (tensor<?x1xi64>, tensor<3xi64>) -> tensor<1x?x32xi64>
    %dim_194 = tensor.dim %505, %c1 : tensor<1x?x32xi64>
    %506 = arith.index_cast %dim_194 : index to i64
    %from_elements_195 = tensor.from_elements %c1_i64, %506, %c32_i64, %c1_i64 : tensor<4xi64>
    %507 = stablehlo.dynamic_reshape %505, %from_elements_195 : (tensor<1x?x32xi64>, tensor<4xi64>) -> tensor<1x?x32x1xi64>
    %508 = stablehlo.dynamic_broadcast_in_dim %145, %from_elements_191, dims = [2] : (tensor<32xi64>, tensor<3xi64>) -> tensor<1x?x32xi64>
    %dim_196 = tensor.dim %508, %c1 : tensor<1x?x32xi64>
    %509 = arith.index_cast %dim_196 : index to i64
    %from_elements_197 = tensor.from_elements %c1_i64, %509, %c32_i64, %c1_i64 : tensor<4xi64>
    %510 = stablehlo.dynamic_reshape %508, %from_elements_197 : (tensor<1x?x32xi64>, tensor<4xi64>) -> tensor<1x?x32x1xi64>
    %511 = stablehlo.concatenate %504, %507, %510, dim = 3 : (tensor<1x?x32x1xi64>, tensor<1x?x32x1xi64>, tensor<1x?x32x1xi64>) -> tensor<1x?x32x3xi64>
    %512 = "stablehlo.gather"(%136, %511) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 3>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<1x35x32xf32>, tensor<1x?x32x3xi64>) -> tensor<1x?x32xf32>
    %513 = shape.shape_of %512 : tensor<1x?x32xf32> -> tensor<3xindex>
    %514 = shape.num_elements %513 : tensor<3xindex> -> index
    %515 = stablehlo.compute_reshape_shape %514, %c_45 : (index, tensor<2xi64>) -> tensor<2xi64>
    %516 = stablehlo.dynamic_reshape %512, %515 : (tensor<1x?x32xf32>, tensor<2xi64>) -> tensor<?x32xf32>
    %517 = stablehlo.transpose %cst_19, dims = [1, 0] : (tensor<14336x32xf32>) -> tensor<32x14336xf32>
    %518 = stablehlo.dot %516, %517 : (tensor<?x32xf32>, tensor<32x14336xf32>) -> tensor<?x14336xf32>
    %519 = stablehlo.logistic %518 : tensor<?x14336xf32>
    %520 = shape.shape_of %519 : tensor<?x14336xf32> -> tensor<2xindex>
    %521 = shape.shape_of %518 : tensor<?x14336xf32> -> tensor<2xindex>
    %522 = shape.cstr_broadcastable %520, %521 : tensor<2xindex>, tensor<2xindex>
    %523 = shape.assuming %522 -> (tensor<?x14336xf32>) {
      %695 = shape.broadcast %520, %521 : tensor<2xindex>, tensor<2xindex> -> tensor<2xindex>
      %696 = stablehlo.dynamic_broadcast_in_dim %519, %695, dims = [0, 1] : (tensor<?x14336xf32>, tensor<2xindex>) -> tensor<?x14336xf32>
      %697 = stablehlo.dynamic_broadcast_in_dim %518, %695, dims = [0, 1] : (tensor<?x14336xf32>, tensor<2xindex>) -> tensor<?x14336xf32>
      %698 = stablehlo.multiply %696, %697 : tensor<?x14336xf32>
      shape.assuming_yield %698 : tensor<?x14336xf32>
    }
    %524 = stablehlo.transpose %cst_18, dims = [1, 0] : (tensor<14336x32xf32>) -> tensor<32x14336xf32>
    %525 = stablehlo.dot %516, %524 : (tensor<?x32xf32>, tensor<32x14336xf32>) -> tensor<?x14336xf32>
    %526 = shape.shape_of %523 : tensor<?x14336xf32> -> tensor<2xindex>
    %527 = shape.shape_of %525 : tensor<?x14336xf32> -> tensor<2xindex>
    %528 = shape.cstr_broadcastable %526, %527 : tensor<2xindex>, tensor<2xindex>
    %529 = shape.assuming %528 -> (tensor<?x14336xf32>) {
      %695 = shape.broadcast %526, %527 : tensor<2xindex>, tensor<2xindex> -> tensor<2xindex>
      %696 = stablehlo.dynamic_broadcast_in_dim %523, %695, dims = [0, 1] : (tensor<?x14336xf32>, tensor<2xindex>) -> tensor<?x14336xf32>
      %697 = stablehlo.dynamic_broadcast_in_dim %525, %695, dims = [0, 1] : (tensor<?x14336xf32>, tensor<2xindex>) -> tensor<?x14336xf32>
      %698 = stablehlo.multiply %696, %697 : tensor<?x14336xf32>
      shape.assuming_yield %698 : tensor<?x14336xf32>
    }
    %530 = stablehlo.transpose %cst_17, dims = [1, 0] : (tensor<32x14336xf32>) -> tensor<14336x32xf32>
    %531 = stablehlo.dot %529, %530 : (tensor<?x14336xf32>, tensor<14336x32xf32>) -> tensor<?x32xf32>
    %dim_198 = tensor.dim %498, %c0 : tensor<?xi64>
    %532 = arith.index_cast %dim_198 : index to i64
    %from_elements_199 = tensor.from_elements %532, %c1_i64 : tensor<2xi64>
    %533 = stablehlo.dynamic_reshape %498, %from_elements_199 : (tensor<?xi64>, tensor<2xi64>) -> tensor<?x1xi64>
    %dim_200 = tensor.dim %495, %c0 : tensor<?xi64>
    %534 = arith.index_cast %dim_200 : index to i64
    %from_elements_201 = tensor.from_elements %534, %c1_i64 : tensor<2xi64>
    %535 = stablehlo.dynamic_reshape %495, %from_elements_201 : (tensor<?xi64>, tensor<2xi64>) -> tensor<?x1xi64>
    %536 = stablehlo.concatenate %533, %535, dim = 1 : (tensor<?x1xi64>, tensor<?x1xi64>) -> tensor<?x2xi64>
    %537 = "stablehlo.gather"(%190, %536) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<35x2x1xf32>, tensor<?x2xi64>) -> tensor<?x1xf32>
    %538 = shape.shape_of %531 : tensor<?x32xf32> -> tensor<2xindex>
    %539 = shape.shape_of %537 : tensor<?x1xf32> -> tensor<2xindex>
    %540 = shape.cstr_broadcastable %538, %539 : tensor<2xindex>, tensor<2xindex>
    %541 = shape.assuming %540 -> (tensor<?x32xf32>) {
      %695 = shape.broadcast %538, %539 : tensor<2xindex>, tensor<2xindex> -> tensor<2xindex>
      %696 = stablehlo.dynamic_broadcast_in_dim %531, %695, dims = [0, 1] : (tensor<?x32xf32>, tensor<2xindex>) -> tensor<?x32xf32>
      %697 = stablehlo.dynamic_broadcast_in_dim %537, %695, dims = [0, 1] : (tensor<?x1xf32>, tensor<2xindex>) -> tensor<?x32xf32>
      %698 = stablehlo.multiply %696, %697 : tensor<?x32xf32>
      shape.assuming_yield %698 : tensor<?x32xf32>
    }
    %542 = shape.shape_of %541 : tensor<?x32xf32> -> tensor<2xindex>
    %543 = stablehlo.dynamic_broadcast_in_dim %541, %542, dims = [0, 1] : (tensor<?x32xf32>, tensor<2xindex>) -> tensor<?x32xf32>
    %544 = stablehlo.dynamic_broadcast_in_dim %201, %542, dims = [] : (tensor<f32>, tensor<2xindex>) -> tensor<?x32xf32>
    %545 = stablehlo.multiply %543, %544 : tensor<?x32xf32>
    %dim_202 = tensor.dim %500, %c0 : tensor<?x1xi64>
    %546 = arith.index_cast %dim_202 : index to i64
    %dim_203 = tensor.dim %541, %c0 : tensor<?x32xf32>
    %547 = arith.index_cast %dim_203 : index to i64
    %548 = arith.maxsi %546, %547 : i64
    %549 = arith.index_cast %548 : i64 to index
    %from_elements_204 = tensor.from_elements %549, %c32 : tensor<2xindex>
    %550 = stablehlo.dynamic_broadcast_in_dim %500, %from_elements_204, dims = [0, 1] : (tensor<?x1xi64>, tensor<2xindex>) -> tensor<?x32xi64>
    %dim_205 = tensor.dim %550, %c0 : tensor<?x32xi64>
    %551 = arith.index_cast %dim_205 : index to i64
    %from_elements_206 = tensor.from_elements %551, %c32_i64 : tensor<2xi64>
    %552 = stablehlo.real_dynamic_slice %545, %c_43, %from_elements_206, %c_44 : (tensor<?x32xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<?x32xf32>
    %from_elements_207 = tensor.from_elements %551, %c32_i64, %c1_i64 : tensor<3xi64>
    %553 = stablehlo.dynamic_reshape %550, %from_elements_207 : (tensor<?x32xi64>, tensor<3xi64>) -> tensor<?x32x1xi64>
    %554 = stablehlo.dynamic_iota %from_elements_207, dim = 1 : (tensor<3xi64>) -> tensor<?x32x1xi64>
    %555 = stablehlo.concatenate %553, %554, dim = 2 : (tensor<?x32x1xi64>, tensor<?x32x1xi64>) -> tensor<?x32x2xi64>
    %556 = "stablehlo.scatter"(%488, %555, %552) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 2>, unique_indices = false}> ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %695 = stablehlo.add %arg3, %arg4 : tensor<f32>
      stablehlo.return %695 : tensor<f32>
    }) : (tensor<35x32xf32>, tensor<?x32x2xi64>, tensor<?x32xf32>) -> tensor<35x32xf32>
    %557 = stablehlo.slice %125 [6:7, 0:2, 0:35] : (tensor<8x2x35xi64>) -> tensor<1x2x35xi64>
    %558 = stablehlo.reshape %557 : (tensor<1x2x35xi64>) -> tensor<2x35xi64>
    %559 = stablehlo.custom_call @byteir.non_zero(%558) {byteir_attrs = {}} : (tensor<2x35xi64>) -> tensor<?x2xi64>
    %dim_208 = tensor.dim %559, %c0 : tensor<?x2xi64>
    %560 = arith.index_cast %dim_208 : index to i64
    %from_elements_209 = tensor.from_elements %560, %c1_i64 : tensor<2xi64>
    %561 = stablehlo.real_dynamic_slice %559, %c_43, %from_elements_209, %c_44 : (tensor<?x2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<?x1xi64>
    %dim_210 = tensor.dim %561, %c0 : tensor<?x1xi64>
    %562 = arith.index_cast %dim_210 : index to i64
    %from_elements_211 = tensor.from_elements %562 : tensor<1xi64>
    %563 = stablehlo.dynamic_reshape %561, %from_elements_211 : (tensor<?x1xi64>, tensor<1xi64>) -> tensor<?xi64>
    %from_elements_212 = tensor.from_elements %560, %c2_i64 : tensor<2xi64>
    %564 = stablehlo.real_dynamic_slice %559, %c_46, %from_elements_212, %c_44 : (tensor<?x2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<?x1xi64>
    %dim_213 = tensor.dim %564, %c0 : tensor<?x1xi64>
    %565 = arith.index_cast %dim_213 : index to i64
    %from_elements_214 = tensor.from_elements %565 : tensor<1xi64>
    %566 = stablehlo.dynamic_reshape %564, %from_elements_214 : (tensor<?x1xi64>, tensor<1xi64>) -> tensor<?xi64>
    %dim_215 = tensor.dim %566, %c0 : tensor<?xi64>
    %567 = arith.index_cast %dim_215 : index to i64
    %from_elements_216 = tensor.from_elements %567, %c1_i64 : tensor<2xi64>
    %568 = stablehlo.dynamic_reshape %566, %from_elements_216 : (tensor<?xi64>, tensor<2xi64>) -> tensor<?x1xi64>
    %dim_217 = tensor.dim %568, %c0 : tensor<?x1xi64>
    %569 = arith.index_cast %dim_217 : index to i64
    %from_elements_218 = tensor.from_elements %c1_i64, %569, %c32_i64 : tensor<3xi64>
    %570 = stablehlo.dynamic_broadcast_in_dim %158, %from_elements_218, dims = [0, 1, 2] : (tensor<1x1x1xi64>, tensor<3xi64>) -> tensor<1x?x32xi64>
    %dim_219 = tensor.dim %570, %c1 : tensor<1x?x32xi64>
    %571 = arith.index_cast %dim_219 : index to i64
    %from_elements_220 = tensor.from_elements %c1_i64, %571, %c32_i64, %c1_i64 : tensor<4xi64>
    %572 = stablehlo.dynamic_reshape %570, %from_elements_220 : (tensor<1x?x32xi64>, tensor<4xi64>) -> tensor<1x?x32x1xi64>
    %573 = stablehlo.dynamic_broadcast_in_dim %568, %from_elements_218, dims = [1, 2] : (tensor<?x1xi64>, tensor<3xi64>) -> tensor<1x?x32xi64>
    %dim_221 = tensor.dim %573, %c1 : tensor<1x?x32xi64>
    %574 = arith.index_cast %dim_221 : index to i64
    %from_elements_222 = tensor.from_elements %c1_i64, %574, %c32_i64, %c1_i64 : tensor<4xi64>
    %575 = stablehlo.dynamic_reshape %573, %from_elements_222 : (tensor<1x?x32xi64>, tensor<4xi64>) -> tensor<1x?x32x1xi64>
    %576 = stablehlo.dynamic_broadcast_in_dim %145, %from_elements_218, dims = [2] : (tensor<32xi64>, tensor<3xi64>) -> tensor<1x?x32xi64>
    %dim_223 = tensor.dim %576, %c1 : tensor<1x?x32xi64>
    %577 = arith.index_cast %dim_223 : index to i64
    %from_elements_224 = tensor.from_elements %c1_i64, %577, %c32_i64, %c1_i64 : tensor<4xi64>
    %578 = stablehlo.dynamic_reshape %576, %from_elements_224 : (tensor<1x?x32xi64>, tensor<4xi64>) -> tensor<1x?x32x1xi64>
    %579 = stablehlo.concatenate %572, %575, %578, dim = 3 : (tensor<1x?x32x1xi64>, tensor<1x?x32x1xi64>, tensor<1x?x32x1xi64>) -> tensor<1x?x32x3xi64>
    %580 = "stablehlo.gather"(%136, %579) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 3>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<1x35x32xf32>, tensor<1x?x32x3xi64>) -> tensor<1x?x32xf32>
    %581 = shape.shape_of %580 : tensor<1x?x32xf32> -> tensor<3xindex>
    %582 = shape.num_elements %581 : tensor<3xindex> -> index
    %583 = stablehlo.compute_reshape_shape %582, %c_45 : (index, tensor<2xi64>) -> tensor<2xi64>
    %584 = stablehlo.dynamic_reshape %580, %583 : (tensor<1x?x32xf32>, tensor<2xi64>) -> tensor<?x32xf32>
    %585 = stablehlo.transpose %cst_16, dims = [1, 0] : (tensor<14336x32xf32>) -> tensor<32x14336xf32>
    %586 = stablehlo.dot %584, %585 : (tensor<?x32xf32>, tensor<32x14336xf32>) -> tensor<?x14336xf32>
    %587 = stablehlo.logistic %586 : tensor<?x14336xf32>
    %588 = shape.shape_of %587 : tensor<?x14336xf32> -> tensor<2xindex>
    %589 = shape.shape_of %586 : tensor<?x14336xf32> -> tensor<2xindex>
    %590 = shape.cstr_broadcastable %588, %589 : tensor<2xindex>, tensor<2xindex>
    %591 = shape.assuming %590 -> (tensor<?x14336xf32>) {
      %695 = shape.broadcast %588, %589 : tensor<2xindex>, tensor<2xindex> -> tensor<2xindex>
      %696 = stablehlo.dynamic_broadcast_in_dim %587, %695, dims = [0, 1] : (tensor<?x14336xf32>, tensor<2xindex>) -> tensor<?x14336xf32>
      %697 = stablehlo.dynamic_broadcast_in_dim %586, %695, dims = [0, 1] : (tensor<?x14336xf32>, tensor<2xindex>) -> tensor<?x14336xf32>
      %698 = stablehlo.multiply %696, %697 : tensor<?x14336xf32>
      shape.assuming_yield %698 : tensor<?x14336xf32>
    }
    %592 = stablehlo.transpose %cst_15, dims = [1, 0] : (tensor<14336x32xf32>) -> tensor<32x14336xf32>
    %593 = stablehlo.dot %584, %592 : (tensor<?x32xf32>, tensor<32x14336xf32>) -> tensor<?x14336xf32>
    %594 = shape.shape_of %591 : tensor<?x14336xf32> -> tensor<2xindex>
    %595 = shape.shape_of %593 : tensor<?x14336xf32> -> tensor<2xindex>
    %596 = shape.cstr_broadcastable %594, %595 : tensor<2xindex>, tensor<2xindex>
    %597 = shape.assuming %596 -> (tensor<?x14336xf32>) {
      %695 = shape.broadcast %594, %595 : tensor<2xindex>, tensor<2xindex> -> tensor<2xindex>
      %696 = stablehlo.dynamic_broadcast_in_dim %591, %695, dims = [0, 1] : (tensor<?x14336xf32>, tensor<2xindex>) -> tensor<?x14336xf32>
      %697 = stablehlo.dynamic_broadcast_in_dim %593, %695, dims = [0, 1] : (tensor<?x14336xf32>, tensor<2xindex>) -> tensor<?x14336xf32>
      %698 = stablehlo.multiply %696, %697 : tensor<?x14336xf32>
      shape.assuming_yield %698 : tensor<?x14336xf32>
    }
    %598 = stablehlo.transpose %cst_14, dims = [1, 0] : (tensor<32x14336xf32>) -> tensor<14336x32xf32>
    %599 = stablehlo.dot %597, %598 : (tensor<?x14336xf32>, tensor<14336x32xf32>) -> tensor<?x32xf32>
    %dim_225 = tensor.dim %566, %c0 : tensor<?xi64>
    %600 = arith.index_cast %dim_225 : index to i64
    %from_elements_226 = tensor.from_elements %600, %c1_i64 : tensor<2xi64>
    %601 = stablehlo.dynamic_reshape %566, %from_elements_226 : (tensor<?xi64>, tensor<2xi64>) -> tensor<?x1xi64>
    %dim_227 = tensor.dim %563, %c0 : tensor<?xi64>
    %602 = arith.index_cast %dim_227 : index to i64
    %from_elements_228 = tensor.from_elements %602, %c1_i64 : tensor<2xi64>
    %603 = stablehlo.dynamic_reshape %563, %from_elements_228 : (tensor<?xi64>, tensor<2xi64>) -> tensor<?x1xi64>
    %604 = stablehlo.concatenate %601, %603, dim = 1 : (tensor<?x1xi64>, tensor<?x1xi64>) -> tensor<?x2xi64>
    %605 = "stablehlo.gather"(%190, %604) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<35x2x1xf32>, tensor<?x2xi64>) -> tensor<?x1xf32>
    %606 = shape.shape_of %599 : tensor<?x32xf32> -> tensor<2xindex>
    %607 = shape.shape_of %605 : tensor<?x1xf32> -> tensor<2xindex>
    %608 = shape.cstr_broadcastable %606, %607 : tensor<2xindex>, tensor<2xindex>
    %609 = shape.assuming %608 -> (tensor<?x32xf32>) {
      %695 = shape.broadcast %606, %607 : tensor<2xindex>, tensor<2xindex> -> tensor<2xindex>
      %696 = stablehlo.dynamic_broadcast_in_dim %599, %695, dims = [0, 1] : (tensor<?x32xf32>, tensor<2xindex>) -> tensor<?x32xf32>
      %697 = stablehlo.dynamic_broadcast_in_dim %605, %695, dims = [0, 1] : (tensor<?x1xf32>, tensor<2xindex>) -> tensor<?x32xf32>
      %698 = stablehlo.multiply %696, %697 : tensor<?x32xf32>
      shape.assuming_yield %698 : tensor<?x32xf32>
    }
    %610 = shape.shape_of %609 : tensor<?x32xf32> -> tensor<2xindex>
    %611 = stablehlo.dynamic_broadcast_in_dim %609, %610, dims = [0, 1] : (tensor<?x32xf32>, tensor<2xindex>) -> tensor<?x32xf32>
    %612 = stablehlo.dynamic_broadcast_in_dim %201, %610, dims = [] : (tensor<f32>, tensor<2xindex>) -> tensor<?x32xf32>
    %613 = stablehlo.multiply %611, %612 : tensor<?x32xf32>
    %dim_229 = tensor.dim %568, %c0 : tensor<?x1xi64>
    %614 = arith.index_cast %dim_229 : index to i64
    %dim_230 = tensor.dim %609, %c0 : tensor<?x32xf32>
    %615 = arith.index_cast %dim_230 : index to i64
    %616 = arith.maxsi %614, %615 : i64
    %617 = arith.index_cast %616 : i64 to index
    %from_elements_231 = tensor.from_elements %617, %c32 : tensor<2xindex>
    %618 = stablehlo.dynamic_broadcast_in_dim %568, %from_elements_231, dims = [0, 1] : (tensor<?x1xi64>, tensor<2xindex>) -> tensor<?x32xi64>
    %dim_232 = tensor.dim %618, %c0 : tensor<?x32xi64>
    %619 = arith.index_cast %dim_232 : index to i64
    %from_elements_233 = tensor.from_elements %619, %c32_i64 : tensor<2xi64>
    %620 = stablehlo.real_dynamic_slice %613, %c_43, %from_elements_233, %c_44 : (tensor<?x32xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<?x32xf32>
    %from_elements_234 = tensor.from_elements %619, %c32_i64, %c1_i64 : tensor<3xi64>
    %621 = stablehlo.dynamic_reshape %618, %from_elements_234 : (tensor<?x32xi64>, tensor<3xi64>) -> tensor<?x32x1xi64>
    %622 = stablehlo.dynamic_iota %from_elements_234, dim = 1 : (tensor<3xi64>) -> tensor<?x32x1xi64>
    %623 = stablehlo.concatenate %621, %622, dim = 2 : (tensor<?x32x1xi64>, tensor<?x32x1xi64>) -> tensor<?x32x2xi64>
    %624 = "stablehlo.scatter"(%556, %623, %620) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 2>, unique_indices = false}> ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %695 = stablehlo.add %arg3, %arg4 : tensor<f32>
      stablehlo.return %695 : tensor<f32>
    }) : (tensor<35x32xf32>, tensor<?x32x2xi64>, tensor<?x32xf32>) -> tensor<35x32xf32>
    %625 = stablehlo.slice %125 [7:8, 0:2, 0:35] : (tensor<8x2x35xi64>) -> tensor<1x2x35xi64>
    %626 = stablehlo.reshape %625 : (tensor<1x2x35xi64>) -> tensor<2x35xi64>
    %627 = stablehlo.custom_call @byteir.non_zero(%626) {byteir_attrs = {}} : (tensor<2x35xi64>) -> tensor<?x2xi64>
    %dim_235 = tensor.dim %627, %c0 : tensor<?x2xi64>
    %628 = arith.index_cast %dim_235 : index to i64
    %from_elements_236 = tensor.from_elements %628, %c1_i64 : tensor<2xi64>
    %629 = stablehlo.real_dynamic_slice %627, %c_43, %from_elements_236, %c_44 : (tensor<?x2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<?x1xi64>
    %dim_237 = tensor.dim %629, %c0 : tensor<?x1xi64>
    %630 = arith.index_cast %dim_237 : index to i64
    %from_elements_238 = tensor.from_elements %630 : tensor<1xi64>
    %631 = stablehlo.dynamic_reshape %629, %from_elements_238 : (tensor<?x1xi64>, tensor<1xi64>) -> tensor<?xi64>
    %from_elements_239 = tensor.from_elements %628, %c2_i64 : tensor<2xi64>
    %632 = stablehlo.real_dynamic_slice %627, %c_46, %from_elements_239, %c_44 : (tensor<?x2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<?x1xi64>
    %dim_240 = tensor.dim %632, %c0 : tensor<?x1xi64>
    %633 = arith.index_cast %dim_240 : index to i64
    %from_elements_241 = tensor.from_elements %633 : tensor<1xi64>
    %634 = stablehlo.dynamic_reshape %632, %from_elements_241 : (tensor<?x1xi64>, tensor<1xi64>) -> tensor<?xi64>
    %dim_242 = tensor.dim %634, %c0 : tensor<?xi64>
    %635 = arith.index_cast %dim_242 : index to i64
    %from_elements_243 = tensor.from_elements %635, %c1_i64 : tensor<2xi64>
    %636 = stablehlo.dynamic_reshape %634, %from_elements_243 : (tensor<?xi64>, tensor<2xi64>) -> tensor<?x1xi64>
    %dim_244 = tensor.dim %636, %c0 : tensor<?x1xi64>
    %637 = arith.index_cast %dim_244 : index to i64
    %from_elements_245 = tensor.from_elements %c1_i64, %637, %c32_i64 : tensor<3xi64>
    %638 = stablehlo.dynamic_broadcast_in_dim %158, %from_elements_245, dims = [0, 1, 2] : (tensor<1x1x1xi64>, tensor<3xi64>) -> tensor<1x?x32xi64>
    %dim_246 = tensor.dim %638, %c1 : tensor<1x?x32xi64>
    %639 = arith.index_cast %dim_246 : index to i64
    %from_elements_247 = tensor.from_elements %c1_i64, %639, %c32_i64, %c1_i64 : tensor<4xi64>
    %640 = stablehlo.dynamic_reshape %638, %from_elements_247 : (tensor<1x?x32xi64>, tensor<4xi64>) -> tensor<1x?x32x1xi64>
    %641 = stablehlo.dynamic_broadcast_in_dim %636, %from_elements_245, dims = [1, 2] : (tensor<?x1xi64>, tensor<3xi64>) -> tensor<1x?x32xi64>
    %dim_248 = tensor.dim %641, %c1 : tensor<1x?x32xi64>
    %642 = arith.index_cast %dim_248 : index to i64
    %from_elements_249 = tensor.from_elements %c1_i64, %642, %c32_i64, %c1_i64 : tensor<4xi64>
    %643 = stablehlo.dynamic_reshape %641, %from_elements_249 : (tensor<1x?x32xi64>, tensor<4xi64>) -> tensor<1x?x32x1xi64>
    %644 = stablehlo.dynamic_broadcast_in_dim %145, %from_elements_245, dims = [2] : (tensor<32xi64>, tensor<3xi64>) -> tensor<1x?x32xi64>
    %dim_250 = tensor.dim %644, %c1 : tensor<1x?x32xi64>
    %645 = arith.index_cast %dim_250 : index to i64
    %from_elements_251 = tensor.from_elements %c1_i64, %645, %c32_i64, %c1_i64 : tensor<4xi64>
    %646 = stablehlo.dynamic_reshape %644, %from_elements_251 : (tensor<1x?x32xi64>, tensor<4xi64>) -> tensor<1x?x32x1xi64>
    %647 = stablehlo.concatenate %640, %643, %646, dim = 3 : (tensor<1x?x32x1xi64>, tensor<1x?x32x1xi64>, tensor<1x?x32x1xi64>) -> tensor<1x?x32x3xi64>
    %648 = "stablehlo.gather"(%136, %647) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2], index_vector_dim = 3>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<1x35x32xf32>, tensor<1x?x32x3xi64>) -> tensor<1x?x32xf32>
    %649 = shape.shape_of %648 : tensor<1x?x32xf32> -> tensor<3xindex>
    %650 = shape.num_elements %649 : tensor<3xindex> -> index
    %651 = stablehlo.compute_reshape_shape %650, %c_45 : (index, tensor<2xi64>) -> tensor<2xi64>
    %652 = stablehlo.dynamic_reshape %648, %651 : (tensor<1x?x32xf32>, tensor<2xi64>) -> tensor<?x32xf32>
    %653 = stablehlo.transpose %cst_13, dims = [1, 0] : (tensor<14336x32xf32>) -> tensor<32x14336xf32>
    %654 = stablehlo.dot %652, %653 : (tensor<?x32xf32>, tensor<32x14336xf32>) -> tensor<?x14336xf32>
    %655 = stablehlo.logistic %654 : tensor<?x14336xf32>
    %656 = shape.shape_of %655 : tensor<?x14336xf32> -> tensor<2xindex>
    %657 = shape.shape_of %654 : tensor<?x14336xf32> -> tensor<2xindex>
    %658 = shape.cstr_broadcastable %656, %657 : tensor<2xindex>, tensor<2xindex>
    %659 = shape.assuming %658 -> (tensor<?x14336xf32>) {
      %695 = shape.broadcast %656, %657 : tensor<2xindex>, tensor<2xindex> -> tensor<2xindex>
      %696 = stablehlo.dynamic_broadcast_in_dim %655, %695, dims = [0, 1] : (tensor<?x14336xf32>, tensor<2xindex>) -> tensor<?x14336xf32>
      %697 = stablehlo.dynamic_broadcast_in_dim %654, %695, dims = [0, 1] : (tensor<?x14336xf32>, tensor<2xindex>) -> tensor<?x14336xf32>
      %698 = stablehlo.multiply %696, %697 : tensor<?x14336xf32>
      shape.assuming_yield %698 : tensor<?x14336xf32>
    }
    %660 = stablehlo.transpose %cst_12, dims = [1, 0] : (tensor<14336x32xf32>) -> tensor<32x14336xf32>
    %661 = stablehlo.dot %652, %660 : (tensor<?x32xf32>, tensor<32x14336xf32>) -> tensor<?x14336xf32>
    %662 = shape.shape_of %659 : tensor<?x14336xf32> -> tensor<2xindex>
    %663 = shape.shape_of %661 : tensor<?x14336xf32> -> tensor<2xindex>
    %664 = shape.cstr_broadcastable %662, %663 : tensor<2xindex>, tensor<2xindex>
    %665 = shape.assuming %664 -> (tensor<?x14336xf32>) {
      %695 = shape.broadcast %662, %663 : tensor<2xindex>, tensor<2xindex> -> tensor<2xindex>
      %696 = stablehlo.dynamic_broadcast_in_dim %659, %695, dims = [0, 1] : (tensor<?x14336xf32>, tensor<2xindex>) -> tensor<?x14336xf32>
      %697 = stablehlo.dynamic_broadcast_in_dim %661, %695, dims = [0, 1] : (tensor<?x14336xf32>, tensor<2xindex>) -> tensor<?x14336xf32>
      %698 = stablehlo.multiply %696, %697 : tensor<?x14336xf32>
      shape.assuming_yield %698 : tensor<?x14336xf32>
    }
    %666 = stablehlo.transpose %cst_11, dims = [1, 0] : (tensor<32x14336xf32>) -> tensor<14336x32xf32>
    %667 = stablehlo.dot %665, %666 : (tensor<?x14336xf32>, tensor<14336x32xf32>) -> tensor<?x32xf32>
    %dim_252 = tensor.dim %634, %c0 : tensor<?xi64>
    %668 = arith.index_cast %dim_252 : index to i64
    %from_elements_253 = tensor.from_elements %668, %c1_i64 : tensor<2xi64>
    %669 = stablehlo.dynamic_reshape %634, %from_elements_253 : (tensor<?xi64>, tensor<2xi64>) -> tensor<?x1xi64>
    %dim_254 = tensor.dim %631, %c0 : tensor<?xi64>
    %670 = arith.index_cast %dim_254 : index to i64
    %from_elements_255 = tensor.from_elements %670, %c1_i64 : tensor<2xi64>
    %671 = stablehlo.dynamic_reshape %631, %from_elements_255 : (tensor<?xi64>, tensor<2xi64>) -> tensor<?x1xi64>
    %672 = stablehlo.concatenate %669, %671, dim = 1 : (tensor<?x1xi64>, tensor<?x1xi64>) -> tensor<?x2xi64>
    %673 = "stablehlo.gather"(%190, %672) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<35x2x1xf32>, tensor<?x2xi64>) -> tensor<?x1xf32>
    %674 = shape.shape_of %667 : tensor<?x32xf32> -> tensor<2xindex>
    %675 = shape.shape_of %673 : tensor<?x1xf32> -> tensor<2xindex>
    %676 = shape.cstr_broadcastable %674, %675 : tensor<2xindex>, tensor<2xindex>
    %677 = shape.assuming %676 -> (tensor<?x32xf32>) {
      %695 = shape.broadcast %674, %675 : tensor<2xindex>, tensor<2xindex> -> tensor<2xindex>
      %696 = stablehlo.dynamic_broadcast_in_dim %667, %695, dims = [0, 1] : (tensor<?x32xf32>, tensor<2xindex>) -> tensor<?x32xf32>
      %697 = stablehlo.dynamic_broadcast_in_dim %673, %695, dims = [0, 1] : (tensor<?x1xf32>, tensor<2xindex>) -> tensor<?x32xf32>
      %698 = stablehlo.multiply %696, %697 : tensor<?x32xf32>
      shape.assuming_yield %698 : tensor<?x32xf32>
    }
    %678 = shape.shape_of %677 : tensor<?x32xf32> -> tensor<2xindex>
    %679 = stablehlo.dynamic_broadcast_in_dim %677, %678, dims = [0, 1] : (tensor<?x32xf32>, tensor<2xindex>) -> tensor<?x32xf32>
    %680 = stablehlo.dynamic_broadcast_in_dim %201, %678, dims = [] : (tensor<f32>, tensor<2xindex>) -> tensor<?x32xf32>
    %681 = stablehlo.multiply %679, %680 : tensor<?x32xf32>
    %dim_256 = tensor.dim %636, %c0 : tensor<?x1xi64>
    %682 = arith.index_cast %dim_256 : index to i64
    %dim_257 = tensor.dim %677, %c0 : tensor<?x32xf32>
    %683 = arith.index_cast %dim_257 : index to i64
    %684 = arith.maxsi %682, %683 : i64
    %685 = arith.index_cast %684 : i64 to index
    %from_elements_258 = tensor.from_elements %685, %c32 : tensor<2xindex>
    %686 = stablehlo.dynamic_broadcast_in_dim %636, %from_elements_258, dims = [0, 1] : (tensor<?x1xi64>, tensor<2xindex>) -> tensor<?x32xi64>
    %dim_259 = tensor.dim %686, %c0 : tensor<?x32xi64>
    %687 = arith.index_cast %dim_259 : index to i64
    %from_elements_260 = tensor.from_elements %687, %c32_i64 : tensor<2xi64>
    %688 = stablehlo.real_dynamic_slice %681, %c_43, %from_elements_260, %c_44 : (tensor<?x32xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<?x32xf32>
    %from_elements_261 = tensor.from_elements %687, %c32_i64, %c1_i64 : tensor<3xi64>
    %689 = stablehlo.dynamic_reshape %686, %from_elements_261 : (tensor<?x32xi64>, tensor<3xi64>) -> tensor<?x32x1xi64>
    %690 = stablehlo.dynamic_iota %from_elements_261, dim = 1 : (tensor<3xi64>) -> tensor<?x32x1xi64>
    %691 = stablehlo.concatenate %689, %690, dim = 2 : (tensor<?x32x1xi64>, tensor<?x32x1xi64>) -> tensor<?x32x2xi64>
    %692 = "stablehlo.scatter"(%624, %691, %688) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 2>, unique_indices = false}> ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %695 = stablehlo.add %arg3, %arg4 : tensor<f32>
      stablehlo.return %695 : tensor<f32>
    }) : (tensor<35x32xf32>, tensor<?x32x2xi64>, tensor<?x32xf32>) -> tensor<35x32xf32>
    %693 = stablehlo.reshape %692 : (tensor<35x32xf32>) -> tensor<5x7x32xf32>
    %694 = stablehlo.add %86, %693 : tensor<5x7x32xf32>
    return %694 : tensor<5x7x32xf32>
  }
}

{-#
  dialect_resources: {
    builtin: {
      torch_tensor_32_torch.float32_1: "0x040000000000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F",
      torch_tensor_32_torch.float32: "0x040000000000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F"
    }
  }
#-}

