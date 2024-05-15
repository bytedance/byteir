// RUN: tf-ext-opt -rewrite-to-custom-call="ops=softmax,log_softmax,gelu,erf,arg_max,arg_min,top_k,layer_norm,l2_norm,addn,one_hot,DynamicMaskStitch,DynamicPartition,DynamicStitch" %s -o %t
// RUN: FileCheck %s < %t
// RUN: python3 numerical_test.py %s %t --config rewrite_to_custom_call

func.func @softmax_case0(%arg0: tensor<100x100xf32>) -> tensor<100x100xf32> {
  %0 = "tf.Softmax"(%arg0) : (tensor<100x100xf32>) -> tensor<100x100xf32>
  func.return %0 : tensor<100x100xf32>
}
// CHECK-LABEL:  func.func @softmax_case0(%arg0: tensor<100x100xf32>) -> tensor<100x100xf32> {
// CHECK:  mhlo.custom_call
// CHECK-SAME: @byteir.softmax
// CHECK-SAME: byteir_attrs = {axis = 1 : i64}

func.func @log_softmax_case0(%arg0: tensor<100x?xf32>) -> tensor<100x?xf32> {
  %0 = "tf.LogSoftmax"(%arg0) : (tensor<100x?xf32>) -> tensor<100x?xf32>
  func.return %0 : tensor<100x?xf32>
}
// CHECK-LABEL:  func.func @log_softmax_case0(%arg0: tensor<100x?xf32>) -> tensor<100x?xf32> {
// CHECK:  mhlo.custom_call
// CHECK-SAME: @byteir.log_softmax
// CHECK-SAME: byteir_attrs = {axis = 1 : i64}

func.func @erf_case0(%1: tensor<100x?x?xf32>) -> tensor<100x?x?xf32> {
  %2 = "tf.Erf"(%1) : (tensor<100x?x?xf32>) -> tensor<100x?x?xf32>
  func.return %2 : tensor<100x?x?xf32>
}
// CHECK-LABEL: func.func @erf_case0
// CHECK:  mhlo.custom_call
// CHECK-SAME: @byteir.erf
// CHECK-SAME: byteir_attrs = {}

func.func @gelu_erf_case0(%arg0: tensor<100x?x?xf32>) -> tensor<100x?x?xf32> {
  %cst = "tf.Const"() {value = dense<0.707106769> : tensor<f32>} : () -> tensor<f32>
  %cst_0 = "tf.Const"() {value = dense<1.000000e+00> : tensor<f32>} : () -> tensor<f32>
  %cst_1 = "tf.Const"() {value = dense<5.000000e-01> : tensor<f32>} : () -> tensor<f32>
  %0 = "tf.Mul"(%cst_1, %arg0) : (tensor<f32>, tensor<100x?x?xf32>) -> tensor<100x?x?xf32>
  %1 = "tf.Mul"(%cst, %arg0) : (tensor<f32>, tensor<100x?x?xf32>) -> tensor<100x?x?xf32>
  %2 = "tf.Erf"(%1) : (tensor<100x?x?xf32>) -> tensor<100x?x?xf32>
  %3 = "tf.AddV2"(%2, %cst_0) : (tensor<100x?x?xf32>, tensor<f32>) -> tensor<100x?x?xf32>
  %4 = "tf.Mul"(%3, %0) : (tensor<100x?x?xf32>, tensor<100x?x?xf32>) -> tensor<100x?x?xf32>
  func.return %4 : tensor<100x?x?xf32>
}
// CHECK-LABEL:  func.func @gelu_erf_case0(%arg0: tensor<100x?x?xf32>) -> tensor<100x?x?xf32> {
// CHECK:  mhlo.custom_call
// CHECK-SAME: @byteir.gelu
// CHECK-SAME: byteir_attrs = {approximate = "erf"}

func.func @gelu_erf_case1(%arg0: tensor<100x?x?xf32>) -> tensor<100x?x?xf32> {
  %cst = "tf.Const"() {value = dense<0.707106769> : tensor<f32>} : () -> tensor<f32>
  %cst_0 = "tf.Const"() {value = dense<1.000000e+00> : tensor<f32>} : () -> tensor<f32>
  %cst_1 = "tf.Const"() {value = dense<5.000000e-01> : tensor<f32>} : () -> tensor<f32>
  %1 = "tf.Mul"(%arg0, %cst) : (tensor<100x?x?xf32>, tensor<f32>) -> tensor<100x?x?xf32>
  %2 = "tf.Erf"(%1) : (tensor<100x?x?xf32>) -> tensor<100x?x?xf32>
  %3 = "tf.AddV2"(%2, %cst_0) : (tensor<100x?x?xf32>, tensor<f32>) -> tensor<100x?x?xf32>
  %4 = "tf.Mul"(%3, %cst_1) : (tensor<100x?x?xf32>, tensor<f32>) -> tensor<100x?x?xf32>
  %5 = "tf.Mul"(%arg0, %4) : (tensor<100x?x?xf32>, tensor<100x?x?xf32>) -> tensor<100x?x?xf32>
  func.return %5 : tensor<100x?x?xf32>
}
// CHECK-LABEL:  func.func @gelu_erf_case1(%arg0: tensor<100x?x?xf32>) -> tensor<100x?x?xf32> {
// CHECK:  mhlo.custom_call
// CHECK-SAME: @byteir.gelu
// CHECK-SAME: byteir_attrs = {approximate = "erf"}

func.func @gelu_tanh_case0(%arg0: tensor<100x?x?xf32>) -> tensor<100x?x?xf32> {
  %cst = "tf.Const"() {value = dense<4.471500e-02> : tensor<f32>} : () -> tensor<f32>
  %cst_0 = "tf.Const"() {value = dense<3.000000e+00> : tensor<f32>} : () -> tensor<f32>
  %cst_1 = "tf.Const"() {value = dense<1.000000e+00> : tensor<f32>} : () -> tensor<f32>
  %cst_2 = "tf.Const"() {value = dense<5.000000e-01> : tensor<f32>} : () -> tensor<f32>
  %cst_3 = "tf.Const"() {value = dense<0.797884583> : tensor<f32>} : () -> tensor<f32>
  %0 = "tf.Pow"(%arg0, %cst_0) : (tensor<100x?x?xf32>, tensor<f32>) -> tensor<100x?x?xf32>
  %1 = "tf.Mul"(%cst, %0) : (tensor<f32>, tensor<100x?x?xf32>) -> tensor<100x?x?xf32>
  %2 = "tf.AddV2"(%1, %arg0) : (tensor<100x?x?xf32>, tensor<100x?x?xf32>) -> tensor<100x?x?xf32>
  %3 = "tf.Mul"(%2, %cst_3) : (tensor<100x?x?xf32>, tensor<f32>) -> tensor<100x?x?xf32>
  %4 = "tf.Tanh"(%3) : (tensor<100x?x?xf32>) -> tensor<100x?x?xf32>
  %5 = "tf.AddV2"(%4, %cst_1) : (tensor<100x?x?xf32>, tensor<f32>) -> tensor<100x?x?xf32>
  %6 = "tf.Mul"(%cst_2, %arg0) : (tensor<f32>, tensor<100x?x?xf32>) -> tensor<100x?x?xf32>
  %7 = "tf.Mul"(%5, %6) : (tensor<100x?x?xf32>, tensor<100x?x?xf32>) -> tensor<100x?x?xf32>
  func.return %7 : tensor<100x?x?xf32>
}
// CHECK-LABEL:  func.func @gelu_tanh_case0(%arg0: tensor<100x?x?xf32>) -> tensor<100x?x?xf32> {
// CHECK:  mhlo.custom_call
// CHECK-SAME: @byteir.gelu
// CHECK-SAME: byteir_attrs = {approximate = "tanh"}

func.func @gelu_tanh_case1(%arg0: tensor<128x512xf16>) -> tensor<128x512xf16> {
  %cst_212 = "tf.Const"() {value = dense<3.000000e+00> : tensor<f16>} : () -> tensor<f16>
  %cst_99 = "tf.Const"() {value = dense<4.470830e-02> : tensor<f16>} : () -> tensor<f16>
  %cst_95 = "tf.Const"() {value = dense<7.978520e-01> : tensor<f16>} : () -> tensor<f16>
  %cst_209 = "tf.Const"() {value = dense<1.000000e+00> : tensor<f16>} : () -> tensor<f16>
  %cst_101 = "tf.Const"() {value = dense<5.000000e-01> : tensor<f16>} : () -> tensor<f16>
  %0 = "tf.Pow"(%arg0, %cst_212) {device = ""} : (tensor<128x512xf16>, tensor<f16>) -> tensor<128x512xf16>
  %1 = "tf.Mul"(%0, %cst_99) {device = ""} : (tensor<128x512xf16>, tensor<f16>) -> tensor<128x512xf16>
  %2 = "tf.AddV2"(%arg0, %1) {device = ""} : (tensor<128x512xf16>, tensor<128x512xf16>) -> tensor<128x512xf16>
  %3 = "tf.Mul"(%2, %cst_95) {device = ""} : (tensor<128x512xf16>, tensor<f16>) -> tensor<128x512xf16>
  %4 = "tf.Tanh"(%3) {device = ""} : (tensor<128x512xf16>) -> tensor<128x512xf16>
  %5 = "tf.AddV2"(%4, %cst_209) {device = ""} : (tensor<128x512xf16>, tensor<f16>) -> tensor<128x512xf16>
  %6 = "tf.Mul"(%5, %cst_101) {_grappler_ArithmeticOptimizer_MinimizeBroadcasts = true, device = ""} : (tensor<128x512xf16>, tensor<f16>) -> tensor<128x512xf16>
  %7 = "tf.Mul"(%arg0, %6) {_grappler_ArithmeticOptimizer_MinimizeBroadcasts = true, device = ""} : (tensor<128x512xf16>, tensor<128x512xf16>) -> tensor<128x512xf16>
  func.return %7 : tensor<128x512xf16>
}
// CHECK-LABEL:  func.func @gelu_tanh_case1(%arg0: tensor<128x512xf16>) -> tensor<128x512xf16> {
// CHECK:  mhlo.custom_call
// CHECK-SAME: @byteir.gelu
// CHECK-SAME: byteir_attrs = {approximate = "tanh"}

func.func @gelu_tanh_case2(%arg0: tensor<1x16x768xf32>) -> tensor<1x16x768xf32> {
  %cst_670 = "tf.Const"() {value = dense<3.000000e+00> : tensor<f32>} : () -> tensor<f32>
  %cst_650 = "tf.Const"() {value = dense<4.471500e-02> : tensor<f32>} : () -> tensor<f32>
  %cst_649 = "tf.Const"() {value = dense<0.797884583> : tensor<f32>} : () -> tensor<f32>
  %cst_472 = "tf.Const"() {value = dense<1.000000e+00> : tensor<f32>} : () -> tensor<f32>
  %cst_651 = "tf.Const"() {value = dense<5.000000e-01> : tensor<f32>} : () -> tensor<f32>
  %0 = "tf.Pow"(%arg0, %cst_670) {device = ""} : (tensor<1x16x768xf32>, tensor<f32>) -> tensor<1x16x768xf32>
  %1 = "tf.Mul"(%0, %cst_650) {device = ""} : (tensor<1x16x768xf32>, tensor<f32>) -> tensor<1x16x768xf32>
  %2 = "tf.AddV2"(%1, %arg0) {device = ""} : (tensor<1x16x768xf32>, tensor<1x16x768xf32>) -> tensor<1x16x768xf32>
  %3 = "tf.Mul"(%2, %cst_649) {device = ""} : (tensor<1x16x768xf32>, tensor<f32>) -> tensor<1x16x768xf32>
  %4 = "tf.Tanh"(%3) {device = ""} : (tensor<1x16x768xf32>) -> tensor<1x16x768xf32>
  %5 = "tf.AddV2"(%4, %cst_472) {device = ""} : (tensor<1x16x768xf32>, tensor<f32>) -> tensor<1x16x768xf32>
  %6 = "tf.Mul"(%arg0, %cst_651) {device = ""} : (tensor<1x16x768xf32>, tensor<f32>) -> tensor<1x16x768xf32>
  %7 = "tf.Mul"(%5, %6) {device = ""} : (tensor<1x16x768xf32>, tensor<1x16x768xf32>) -> tensor<1x16x768xf32>
  func.return %7 : tensor<1x16x768xf32>
}
// CHECK-LABEL: func.func @gelu_tanh_case2
// CHECK: mhlo.custom_call
// CHECK-SAME: @byteir.gelu
// CHECK-SAME: byteir_attrs = {approximate = "tanh"}

func.func @gelu_tanh_case3(%arg0: tensor<128x512xf16>) -> tensor<128x512xf16> {
  %cst_212 = "tf.Const"() {value = dense<3.000000e+00> : tensor<f16>} : () -> tensor<f16>
  %cst_99 = "tf.Const"() {value = dense<4.470830e-02> : tensor<f16>} : () -> tensor<f16>
  %cst_95 = "tf.Const"() {value = dense<7.978520e-01> : tensor<f16>} : () -> tensor<f16>
  %cst_209 = "tf.Const"() {value = dense<1.000000e+00> : tensor<f16>} : () -> tensor<f16>
  %cst_101 = "tf.Const"() {value = dense<5.000000e-01> : tensor<f16>} : () -> tensor<f16>
  %0 = "tf.Pow"(%arg0, %cst_212) {device = ""} : (tensor<128x512xf16>, tensor<f16>) -> tensor<128x512xf16>
  %1 = "tf.Mul"(%0, %cst_99) {device = ""} : (tensor<128x512xf16>, tensor<f16>) -> tensor<128x512xf16>
  %2 = "tf.AddV2"(%arg0, %1) {device = ""} : (tensor<128x512xf16>, tensor<128x512xf16>) -> tensor<128x512xf16>
  %3 = "tf.Mul"(%2, %cst_95) {device = ""} : (tensor<128x512xf16>, tensor<f16>) -> tensor<128x512xf16>
  %4 = "tf.Tanh"(%3) {device = ""} : (tensor<128x512xf16>) -> tensor<128x512xf16>
  %5 = "tf.AddV2"(%4, %cst_209) {device = ""} : (tensor<128x512xf16>, tensor<f16>) -> tensor<128x512xf16>
  %6 = "tf.Mul"(%arg0, %cst_101) {device = ""} : (tensor<128x512xf16>, tensor<f16>) -> tensor<128x512xf16>
  %7 = "tf.Mul"(%5, %6) {device = ""} : (tensor<128x512xf16>, tensor<128x512xf16>) -> tensor<128x512xf16>
  func.return %7 : tensor<128x512xf16>
}
// CHECK-LABEL:  func.func @gelu_tanh_case3(%arg0: tensor<128x512xf16>) -> tensor<128x512xf16> {
// CHECK:  mhlo.custom_call
// CHECK-SAME: @byteir.gelu
// CHECK-SAME: byteir_attrs = {approximate = "tanh"}

func.func @arg_min_case0(%arg0: tensor<100x?x?xf32>) -> tensor<100x?xi64> {
  %cst = "tf.Const"() {value = dense<2> : tensor<i32>} : () -> tensor<i32>
  %0 = "tf.ArgMin"(%arg0, %cst) : (tensor<100x?x?xf32>, tensor<i32>) -> tensor<100x?xi64>
  func.return %0 : tensor<100x?xi64>
}
// CHECK-LABEL:  func.func @arg_min_case0(%arg0: tensor<100x?x?xf32>) -> tensor<100x?xi64> {
// CHECK: mhlo.custom_call
// CHECK-SAME: @byteir.arg_min
// CHECK-SAME: byteir_attrs = {axis = 2 : i64, keep_dims = false, select_last_index = false}

func.func @arg_max_case0(%arg0: tensor<100x?x?xf32>) -> tensor<100x?xi64> {
  %cst = "tf.Const"() {value = dense<2> : tensor<i32>} : () -> tensor<i32>
  %0 = "tf.ArgMax"(%arg0, %cst) : (tensor<100x?x?xf32>, tensor<i32>) -> tensor<100x?xi64>
  func.return %0 : tensor<100x?xi64>
}
// CHECK-LABEL:  func.func @arg_max_case0(%arg0: tensor<100x?x?xf32>) -> tensor<100x?xi64> {
// CHECK: mhlo.custom_call
// CHECK-SAME: @byteir.arg_max
// CHECK-SAME: byteir_attrs = {axis = 2 : i64, keep_dims = false, select_last_index = false}

func.func @topk_index_needed_case0(%arg0: tensor<16x32xf32>) -> (tensor<16x10xf32>, tensor<16x10xi32>) {
    %0 = "tf.Const"() {value = dense<10> : tensor<i32>} : () -> tensor<i32>
    %1, %2 = "tf.TopKV2"(%arg0, %0) {sorted = true} : (tensor<16x32xf32>, tensor<i32>) -> (tensor<16x10xf32>, tensor<16x10xi32>)
    func.return %1, %2: tensor<16x10xf32>, tensor<16x10xi32>
}
// CHECK-LABEL: func.func @topk_index_needed_case0
// CHECK: mhlo.custom_call
// CHECK-SAME: @byteir.top_k
// CHECK-SAME: byteir_attrs = {axis = [1], k = 10 : i64, sorted = true}

func.func @addn_case0(%arg0: tensor<16x32xf32>, %arg1: tensor<16x32xf32>, %arg2: tensor<16x32xf32>) -> tensor<16x32xf32> {
  %0 = "tf.AddN"(%arg0, %arg1, %arg2) : (tensor<16x32xf32>, tensor<16x32xf32>, tensor<16x32xf32>) -> tensor<16x32xf32>
  return %0 : tensor<16x32xf32>
}
// CHECK-LABEL: func.func @addn_case0
// CHECK: mhlo.custom_call
// CHECK-SAME: @byteir.addn

func.func @layer_norm(%arg0: tensor<1x32x3xf32>) -> tensor<1x32x3xf32> {
  %cst = "tf.Const"() {value = dense<9.99999997E-7> : tensor<f32>} : () -> tensor<f32>
  %cst_0 = "tf.Const"() {value = dense<[0.0401659757, -0.11370486, 0.432680517]> : tensor<3xf32>} : () -> tensor<3xf32>
  %cst_1 = "tf.Const"() {value = dense<[0.445568085, 0.45303449, 3.227140e-01]> : tensor<3xf32>} : () -> tensor<3xf32>
  %cst_2 = "tf.Const"() {value = dense<2> : tensor<1xi32>} : () -> tensor<1xi32>
  %0 = "tf.Mean"(%arg0, %cst_2) {keep_dims = true} : (tensor<1x32x3xf32>, tensor<1xi32>) -> tensor<1x32x1xf32>
  %1 = "tf.SquaredDifference"(%arg0, %0) : (tensor<1x32x3xf32>, tensor<1x32x1xf32>) -> tensor<1x32x3xf32>
  %2 = "tf.Mean"(%1, %cst_2) {keep_dims = true} : (tensor<1x32x3xf32>, tensor<1xi32>) -> tensor<1x32x1xf32>
  %3 = "tf.AddV2"(%cst, %2) : (tensor<f32>, tensor<1x32x1xf32>) -> tensor<1x32x1xf32>
  %4 = "tf.Rsqrt"(%3) : (tensor<1x32x1xf32>) -> tensor<1x32x1xf32>
  %5 = "tf.Mul"(%4, %cst_1) : (tensor<1x32x1xf32>, tensor<3xf32>) -> tensor<1x32x3xf32>
  %6 = "tf.Mul"(%arg0, %5) : (tensor<1x32x3xf32>, tensor<1x32x3xf32>) -> tensor<1x32x3xf32>
  %7 = "tf.Mul"(%5, %0) : (tensor<1x32x3xf32>, tensor<1x32x1xf32>) -> tensor<1x32x3xf32>
  %8 = "tf.Sub"(%cst_0, %7) : (tensor<3xf32>, tensor<1x32x3xf32>) -> tensor<1x32x3xf32>
  %9 = "tf.AddV2"(%6, %8) : (tensor<1x32x3xf32>, tensor<1x32x3xf32>) -> tensor<1x32x3xf32>
  func.return %9 : tensor<1x32x3xf32>
}
// CHECK-LABEL:  func.func @layer_norm(%arg0: tensor<1x32x3xf32>) -> tensor<1x32x3xf32> {
// CHECK-NEXT:    %cst = "tf.Const"() <{value = dense<[0.0401659757, -0.11370486, 0.432680517]> : tensor<3xf32>}> : () -> tensor<3xf32>
// CHECK-NEXT:    %cst_0 = "tf.Const"() <{value = dense<[0.445568085, 0.45303449, 3.227140e-01]> : tensor<3xf32>}> : () -> tensor<3xf32>
// CHECK: mhlo.custom_call
// CHECK-SAME: @byteir.layer_norm
// CHECK-SAME: byteir_attrs = {axis = [2], epsilon = 9.9999999747524271E-7 : f64}

func.func @layer_norm_negative_axis(%arg0: tensor<1x32x3xf32>) -> tensor<1x32x3xf32> {
  %cst = "tf.Const"() {value = dense<9.99999997E-7> : tensor<f32>} : () -> tensor<f32>
  %cst_0 = "tf.Const"() {value = dense<[0.0401659757, -0.11370486, 0.432680517]> : tensor<3xf32>} : () -> tensor<3xf32>
  %cst_1 = "tf.Const"() {value = dense<[0.445568085, 0.45303449, 3.227140e-01]> : tensor<3xf32>} : () -> tensor<3xf32>
  %cst_2 = "tf.Const"() {value = dense<-1> : tensor<1xi32>} : () -> tensor<1xi32>
  %0 = "tf.Mean"(%arg0, %cst_2) {keep_dims = true} : (tensor<1x32x3xf32>, tensor<1xi32>) -> tensor<1x32x1xf32>
  %1 = "tf.SquaredDifference"(%arg0, %0) : (tensor<1x32x3xf32>, tensor<1x32x1xf32>) -> tensor<1x32x3xf32>
  %2 = "tf.Mean"(%1, %cst_2) {keep_dims = true} : (tensor<1x32x3xf32>, tensor<1xi32>) -> tensor<1x32x1xf32>
  %3 = "tf.AddV2"(%cst, %2) : (tensor<f32>, tensor<1x32x1xf32>) -> tensor<1x32x1xf32>
  %4 = "tf.Rsqrt"(%3) : (tensor<1x32x1xf32>) -> tensor<1x32x1xf32>
  %5 = "tf.Mul"(%4, %cst_1) : (tensor<1x32x1xf32>, tensor<3xf32>) -> tensor<1x32x3xf32>
  %6 = "tf.Mul"(%arg0, %5) : (tensor<1x32x3xf32>, tensor<1x32x3xf32>) -> tensor<1x32x3xf32>
  %7 = "tf.Mul"(%5, %0) : (tensor<1x32x3xf32>, tensor<1x32x1xf32>) -> tensor<1x32x3xf32>
  %8 = "tf.Sub"(%cst_0, %7) : (tensor<3xf32>, tensor<1x32x3xf32>) -> tensor<1x32x3xf32>
  %9 = "tf.AddV2"(%6, %8) : (tensor<1x32x3xf32>, tensor<1x32x3xf32>) -> tensor<1x32x3xf32>
  func.return %9 : tensor<1x32x3xf32>
}
// CHECK-LABEL:  func.func @layer_norm_negative_axis(%arg0: tensor<1x32x3xf32>) -> tensor<1x32x3xf32> {
// CHECK-NEXT:    %cst = "tf.Const"() <{value = dense<[0.0401659757, -0.11370486, 0.432680517]> : tensor<3xf32>}> : () -> tensor<3xf32>
// CHECK-NEXT:    %cst_0 = "tf.Const"() <{value = dense<[0.445568085, 0.45303449, 3.227140e-01]> : tensor<3xf32>}> : () -> tensor<3xf32>
// CHECK: mhlo.custom_call
// CHECK-SAME: @byteir.layer_norm
// CHECK-SAME: byteir_attrs = {axis = [2], epsilon = 9.9999999747524271E-7 : f64}

func.func @layer_norm_without_beta(%arg0: tensor<512x128xf16>) -> tensor<512x128xf16> {
  %cst_1 = "tf.Const"() {value = dense<"0xED00B183330094806480C0007F80B180948032005E8238804400C9801303E500E100EB014D81380015003901BB80718051819B00A102C780DF00510345010581F0007182D7014F030F00D7004E04A40049812B80CA849181BF800480E500D380C30191819200648228814380E5002900AF00FA804302D80166821601EC0233803500D580CF80B9801500538226808C817180E9012D0010018481DB000B004280A9806201248089001501A7019B000100138252819D007C8186808102F180FF002E819380BB0020003A003702C80092821080A800D800C681C0027C81D1805500B1810F006480150184802D0102841D8120838F002F000501E90005806A01BC00"> : tensor<128xf16>} : () -> tensor<128xf16>
  %cst_2 = "tf.Const"() {value = dense<1.000170e-04> : tensor<f16>} : () -> tensor<f16>
  %cst_3 = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %0 = "tf.Mean"(%arg0, %cst_3) {device = "", keep_dims = true} : (tensor<512x128xf16>, tensor<i32>) -> tensor<512x1xf16>
  %1 = "tf.SquaredDifference"(%arg0, %0) {device = ""} : (tensor<512x128xf16>, tensor<512x1xf16>) -> tensor<512x128xf16>
  %2 = "tf.Mean"(%1, %cst_3) {device = "", keep_dims = true} : (tensor<512x128xf16>, tensor<i32>) -> tensor<512x1xf16>
  %3 = "tf.AddV2"(%2, %cst_2) {device = ""} : (tensor<512x1xf16>, tensor<f16>) -> tensor<512x1xf16>
  %4 = "tf.Rsqrt"(%3) {device = ""} : (tensor<512x1xf16>) -> tensor<512x1xf16>
  %5 = "tf.Mul"(%4, %cst_1) {device = ""} : (tensor<512x1xf16>, tensor<128xf16>) -> tensor<512x128xf16>
  %6 = "tf.Mul"(%arg0, %5) {device = ""} : (tensor<512x128xf16>, tensor<512x128xf16>) -> tensor<512x128xf16>
  %7 = "tf.Mul"(%5, %0) {device = ""} : (tensor<512x128xf16>, tensor<512x1xf16>) -> tensor<512x128xf16>
  %8 = "tf.Sub"(%6, %7) {device = ""} : (tensor<512x128xf16>, tensor<512x128xf16>) -> tensor<512x128xf16>
  func.return %8 : tensor<512x128xf16>
}
// CHECK-LABEL:  func.func @layer_norm_without_beta(%arg0: tensor<512x128xf16>) -> tensor<512x128xf16> {
// CHECH-DAG:    "tf.Const"() <{value = dense<0.000000e+00> : tensor<128xf16>}> : () -> tensor<128xf16>
// CHECK-DAG:    "tf.Const"() <{value = dense<"0xED00B183330094806480C0007F80B180948032005E8238804400C9801303E500E100EB014D81380015003901BB80718051819B00A102C780DF00510345010581F0007182D7014F030F00D7004E04A40049812B80CA849181BF800480E500D380C30191819200648228814380E5002900AF00FA804302D80166821601EC0233803500D580CF80B9801500538226808C817180E9012D0010018481DB000B004280A9806201248089001501A7019B000100138252819D007C8186808102F180FF002E819380BB0020003A003702C80092821080A800D800C681C0027C81D1805500B1810F006480150184802D0102841D8120838F002F000501E90005806A01BC00"> : tensor<128xf16>}> : () -> tensor<128xf16>
// CHECK: mhlo.custom_call
// CHECK-SAME: @byteir.layer_norm
// CHECK-SAME: byteir_attrs = {axis = [1], epsilon = 1.0001659393310547E-4 : f64}

func.func @layer_norm_three_dim(%arg0: tensor<2x8x4xf32>) -> tensor<2x8x4xf32> {
  %cst = "tf.Const"() <{value = dense<-1> : tensor<1xi32>}> : () -> tensor<1xi32>
  %cst_1 = "tf.Const"() <{value = dense<9.99999997E-7> : tensor<f32>}> : () -> tensor<f32>
  %cst_2 = "tf.Const"() <{value = dense<[[[0.1, 0.2, 0.3, 0.4]], [[0.5, 0.6, 0.7, 0.8]]]> : tensor<2x1x4xf32>}> : () -> tensor<2x1x4xf32>
  %cst_3 = "tf.Const"() <{value = dense<[[[0.01, 0.02, 0.03, 0.04]], [[0.05, 0.06, 0.07, 0.08]]]> : tensor<2x1x4xf32>}> : () -> tensor<2x1x4xf32>
  %0 = "tf.Mean"(%arg0, %cst) <{keep_dims = true}> {device = ""} : (tensor<2x8x4xf32>, tensor<1xi32>) -> tensor<2x8x1xf32>
  %1 = "tf.SquaredDifference"(%0, %arg0) {device = ""} : (tensor<2x8x1xf32>, tensor<2x8x4xf32>) -> tensor<2x8x4xf32>
  %2 = "tf.Mean"(%1, %cst) <{keep_dims = true}> {device = ""} : (tensor<2x8x4xf32>, tensor<1xi32>) -> tensor<2x8x1xf32>
  %3 = "tf.AddV2"(%2, %cst_1) {device = ""} : (tensor<2x8x1xf32>, tensor<f32>) -> tensor<2x8x1xf32>
  %4 = "tf.Rsqrt"(%3) {device = ""} : (tensor<2x8x1xf32>) -> tensor<2x8x1xf32>
  %5 = "tf.Mul"(%4, %cst_2) {device = ""} : (tensor<2x8x1xf32>, tensor<2x1x4xf32>) -> tensor<2x8x4xf32>
  %6 = "tf.Mul"(%5, %arg0) {device = ""} : (tensor<2x8x4xf32>, tensor<2x8x4xf32>) -> tensor<2x8x4xf32>
  %7 = "tf.Mul"(%5, %0) {device = ""} : (tensor<2x8x4xf32>, tensor<2x8x1xf32>) -> tensor<2x8x4xf32>
  %8 = "tf.Sub"(%cst_3, %7) {device = ""} : (tensor<2x1x4xf32>, tensor<2x8x4xf32>) -> tensor<2x8x4xf32>
  %9 = "tf.AddV2"(%6, %8) {device = ""} : (tensor<2x8x4xf32>, tensor<2x8x4xf32>) -> tensor<2x8x4xf32>
  func.return %9 : tensor<2x8x4xf32>
}
// CHECK-LABEL:  func.func @layer_norm_three_dim(%arg0: tensor<2x8x4xf32>) -> tensor<2x8x4xf32> {
// CHECH-DAG:    "tf.Const"() <{value = dense<[[[1.000000e-01, 2.000000e-01, 3.000000e-01, 4.000000e-01]], [[5.000000e-01, 6.000000e-01, 0.699999988, 8.000000e-01]]]> : tensor<2x1x4xf32>
// CHECK-DAG:    "tf.Const"() <{value = dense<[[[0.00999999977, 2.000000e-02, 3.000000e-02, 4.000000e-02]], [[5.000000e-02, 6.000000e-02, 7.000000e-02, 8.000000e-02]]]> : tensor<2x1x4xf3
// CHECK: mhlo.custom_call
// CHECK-SAME: @byteir.layer_norm
// CHECK-SAME: byteir_attrs = {axis = [2], epsilon = 9.9999999747524271E-7 : f64}

func.func @layer_norm_swap_add(%arg0: tensor<2x32x3xf32>) -> tensor<2x32x3xf32> {
  %cst_15 = "tf.Const"() {value = dense<9.99999997E-7> : tensor<f32>} : () -> tensor<f32>
  %cst_0 = "tf.Const"() {value = dense<[0.0401659757, -0.11370486, 0.432680517]> : tensor<3xf32>} : () -> tensor<3xf32>
  %cst_1 = "tf.Const"() {value = dense<[0.445568085, 0.45303449, 3.227140e-01]> : tensor<3xf32>} : () -> tensor<3xf32>
  %cst_152 = "tf.Const"() {value = dense<2> : tensor<1xi32>} : () -> tensor<1xi32>
  %0 = "tf.Mean"(%arg0, %cst_152) {device = "", keep_dims = true} : (tensor<2x32x3xf32>, tensor<1xi32>) -> tensor<2x32x1xf32>
  %1 = "tf.SquaredDifference"(%arg0, %0) {device = ""} : (tensor<2x32x3xf32>, tensor<2x32x1xf32>) -> tensor<2x32x3xf32>
  %2 = "tf.Mean"(%1, %cst_152) {device = "", keep_dims = true} : (tensor<2x32x3xf32>, tensor<1xi32>) -> tensor<2x32x1xf32>
  %3 = "tf.AddV2"(%2, %cst_15) {device = ""} : (tensor<2x32x1xf32>, tensor<f32>) -> tensor<2x32x1xf32>
  %4 = "tf.Rsqrt"(%3) {device = ""} : (tensor<2x32x1xf32>) -> tensor<2x32x1xf32>
  %5 = "tf.Mul"(%4, %cst_0) {device = ""} : (tensor<2x32x1xf32>, tensor<3xf32>) -> tensor<2x32x3xf32>
  %6 = "tf.Mul"(%arg0, %5) {device = ""} : (tensor<2x32x3xf32>, tensor<2x32x3xf32>) -> tensor<2x32x3xf32>
  %7 = "tf.Mul"(%5, %0) {device = ""} : (tensor<2x32x3xf32>, tensor<2x32x1xf32>) -> tensor<2x32x3xf32>
  %8 = "tf.Sub"(%cst_1, %7) {device = ""} : (tensor<3xf32>, tensor<2x32x3xf32>) -> tensor<2x32x3xf32>
  %9 = "tf.AddV2"(%6, %8) {device = ""} : (tensor<2x32x3xf32>, tensor<2x32x3xf32>) -> tensor<2x32x3xf32>
  return %9 : tensor<2x32x3xf32>
}
// CHECK-LABEL: @layer_norm_swap_add
// CHECK: mhlo.custom_call
// CHECK-SAME: @byteir.layer_norm

func.func @layer_norm_swap_mul(%arg0: tensor<1x16x3xf32>) -> tensor<1x16x3xf32> {
  %cst_685 = "tf.Const"() {value = dense<9.99999997E-7> : tensor<f32>} : () -> tensor<f32>
  %cst_684 = "tf.Const"() {value = dense<[0.0401659757, -0.11370486, 0.432680517]> : tensor<3xf32>} : () -> tensor<3xf32>
  %cst_683 = "tf.Const"() {value = dense<[0.445568085, 0.45303449, 3.227140e-01]> : tensor<3xf32>} : () -> tensor<3xf32>
  %cst_682 = "tf.Const"() {value = dense<2> : tensor<1xi32>} : () -> tensor<1xi32>
  %0 = "tf.Mean"(%arg0, %cst_682) {device = "", keep_dims = true} : (tensor<1x16x3xf32>, tensor<1xi32>) -> tensor<1x16x1xf32>
  %1 = "tf.SquaredDifference"(%arg0, %0) {device = ""} : (tensor<1x16x3xf32>, tensor<1x16x1xf32>) -> tensor<1x16x3xf32>
  %2 = "tf.Mean"(%1, %cst_682) {device = "", keep_dims = true} : (tensor<1x16x3xf32>, tensor<1xi32>) -> tensor<1x16x1xf32>
  %3 = "tf.AddV2"(%2, %cst_685) {device = ""} : (tensor<1x16x1xf32>, tensor<f32>) -> tensor<1x16x1xf32>
  %4 = "tf.Rsqrt"(%3) {device = ""} : (tensor<1x16x1xf32>) -> tensor<1x16x1xf32>
  %5 = "tf.Mul"(%4, %cst_683) {device = ""} : (tensor<1x16x1xf32>, tensor<3xf32>) -> tensor<1x16x3xf32>
  %6 = "tf.Mul"(%5, %arg0) {device = ""} : (tensor<1x16x3xf32>, tensor<1x16x3xf32>) -> tensor<1x16x3xf32>
  %7 = "tf.Mul"(%5, %0) {device = ""} : (tensor<1x16x3xf32>, tensor<1x16x1xf32>) -> tensor<1x16x3xf32>
  %8 = "tf.Sub"(%cst_684, %7) {device = ""} : (tensor<3xf32>, tensor<1x16x3xf32>) -> tensor<1x16x3xf32>
  %9 = "tf.AddV2"(%6, %8) {device = ""} : (tensor<1x16x3xf32>, tensor<1x16x3xf32>) -> tensor<1x16x3xf32>
  return %9 : tensor<1x16x3xf32>
}
// CHECK-LABEL: @layer_norm_swap_mul
// CHECK: mhlo.custom_call
// CHECK-SAME: @byteir.layer_norm

func.func @layer_norm_swap_squarediff(%arg0: tensor<256x4xf32>) -> tensor<256x4xf32> {
  %cst_136 = "tf.Const"() {value = dense<1> : tensor<1xi32>} : () -> tensor<1xi32>
  %cst_400 = "tf.Const"() {value = dense<9.99999996E-13> : tensor<f32>} : () -> tensor<f32>
  %cst_395 = "tf.Const"() {value = dense<[0.0401659757, -0.11370486, 0.432680517, 0.4000000]> : tensor<4xf32>} : () -> tensor<4xf32>
  %cst_396 = "tf.Const"() {value = dense<[0.445568085, 0.45303449, 3.227140e-01, 0.4000000]> : tensor<4xf32>} : () -> tensor<4xf32>
  %0 = "tf.Mean"(%arg0, %cst_136) {device = "", keep_dims = true} : (tensor<256x4xf32>, tensor<1xi32>) -> tensor<256x1xf32>
  %1 = "tf.SquaredDifference"(%0, %arg0) {device = ""} : (tensor<256x1xf32>, tensor<256x4xf32>) -> tensor<256x4xf32>
  %2 = "tf.Mean"(%1, %cst_136) {device = "", keep_dims = true} : (tensor<256x4xf32>, tensor<1xi32>) -> tensor<256x1xf32>
  %3 = "tf.AddV2"(%2, %cst_400) {device = ""} : (tensor<256x1xf32>, tensor<f32>) -> tensor<256x1xf32>
  %4 = "tf.Rsqrt"(%3) {device = ""} : (tensor<256x1xf32>) -> tensor<256x1xf32>
  %5 = "tf.Mul"(%4, %cst_395) {device = ""} : (tensor<256x1xf32>, tensor<4xf32>) -> tensor<256x4xf32>
  %6 = "tf.Mul"(%5, %arg0) {device = ""} : (tensor<256x4xf32>, tensor<256x4xf32>) -> tensor<256x4xf32>
  %7 = "tf.Mul"(%5, %0) {device = ""} : (tensor<256x4xf32>, tensor<256x1xf32>) -> tensor<256x4xf32>
  %8 = "tf.Sub"(%cst_396, %7) {device = ""} : (tensor<4xf32>, tensor<256x4xf32>) -> tensor<256x4xf32>
  %9 = "tf.AddV2"(%6, %8) {device = ""} : (tensor<256x4xf32>, tensor<256x4xf32>) -> tensor<256x4xf32>
  return %9 : tensor<256x4xf32>
}
// CHECK-LABEL: @layer_norm_swap_squarediff
// CHECK: mhlo.custom_call
// CHECK-SAME: @byteir.layer_norm

func.func @layer_norm_V2(%arg0: tensor<1x32x3xf32>) -> tensor<1x32x3xf32> {
  %cst = "tf.Const"() {value = dense<9.99999997E-7> : tensor<f32>} : () -> tensor<f32>
  %cst_0 = "tf.Const"() {value = dense<[0.0401659757, -0.11370486, 0.432680517]> : tensor<3xf32>} : () -> tensor<3xf32>
  %cst_1 = "tf.Const"() {value = dense<[0.445568085, 0.45303449, 3.227140e-01]> : tensor<3xf32>} : () -> tensor<3xf32>
  %cst_2 = "tf.Const"() {value = dense<2> : tensor<1xi32>} : () -> tensor<1xi32>
  %0 = "tf.Mean"(%arg0, %cst_2) {keep_dims = true} : (tensor<1x32x3xf32>, tensor<1xi32>) -> tensor<1x32x1xf32>
  %1 = "tf.Sub"(%arg0, %0) : (tensor<1x32x3xf32>, tensor<1x32x1xf32>) -> tensor<1x32x3xf32>
  %2 = "tf.SquaredDifference"(%arg0, %0) : (tensor<1x32x3xf32>, tensor<1x32x1xf32>) -> tensor<1x32x3xf32>
  %3 = "tf.Mean"(%2, %cst_2) {keep_dims = true} : (tensor<1x32x3xf32>, tensor<1xi32>) -> tensor<1x32x1xf32>
  %4 = "tf.AddV2"(%3, %cst) : (tensor<1x32x1xf32>, tensor<f32>) -> tensor<1x32x1xf32>
  %5 = "tf.Rsqrt"(%4) : (tensor<1x32x1xf32>) -> tensor<1x32x1xf32>
  %6 = "tf.Mul"(%5, %cst_0) : (tensor<1x32x1xf32>, tensor<3xf32>) -> tensor<1x32x3xf32>
  %7 = "tf.Mul"(%1, %6) : (tensor<1x32x3xf32>, tensor<1x32x3xf32>) -> tensor<1x32x3xf32>
  %8 = "tf.AddV2"(%7, %cst_1) : (tensor<1x32x3xf32>, tensor<3xf32>) -> tensor<1x32x3xf32>
  func.return %8 : tensor<1x32x3xf32>
}
// CHECK-LABEL:  func.func @layer_norm_V2(%arg0: tensor<1x32x3xf32>) -> tensor<1x32x3xf32> {
// CHECK-NEXT:    %cst = "tf.Const"() <{value = dense<[0.0401659757, -0.11370486, 0.432680517]> : tensor<3xf32>}> : () -> tensor<3xf32>
// CHECK-NEXT:    %cst_0 = "tf.Const"() <{value = dense<[0.445568085, 0.45303449, 3.227140e-01]> : tensor<3xf32>}> : () -> tensor<3xf32>
// CHECK: mhlo.custom_call
// CHECK-SAME: @byteir.layer_norm
// CHECK-SAME: byteir_attrs = {axis = [2], epsilon = 9.9999999747524271E-7 : f64}

func.func @layer_norm_V3_disable_minimize_broadcast(%113: tensor<1x3xf32>) -> tensor<1x3xf32> {
  %cst_76 = "tf.Const"() {value = dense<9.99999997E-7> : tensor<f32>} : () -> tensor<f32>
  %cst_79 = "tf.Const"() {value = dense<1> : tensor<1xi32>} : () -> tensor<1xi32>
  %cst_55 = "tf.Const"() {value = dense<[0.0401659757, -0.11370486, 0.432680517]> : tensor<3xf32>} : () -> tensor<3xf32>
  %cst_54 = "tf.Const"() {value = dense<[0.445568085, 0.45303449, 3.227140e-01]> : tensor<3xf32>} : () -> tensor<3xf32>
  %114 = "tf.Mean"(%113, %cst_79) {device = "", keep_dims = true} : (tensor<1x3xf32>, tensor<1xi32>) -> tensor<1x1xf32>
  %115 = "tf.SquaredDifference"(%113, %114) {device = ""} : (tensor<1x3xf32>, tensor<1x1xf32>) -> tensor<1x3xf32>
  %116 = "tf.Mean"(%115, %cst_79) {device = "", keep_dims = true} : (tensor<1x3xf32>, tensor<1xi32>) -> tensor<1x1xf32>
  %117 = "tf.AddV2"(%116, %cst_76) {device = ""} : (tensor<1x1xf32>, tensor<f32>) -> tensor<1x1xf32>
  %118 = "tf.Rsqrt"(%117) {device = ""} : (tensor<1x1xf32>) -> tensor<1x1xf32>
  %119 = "tf.Sub"(%113, %114) {device = ""} : (tensor<1x3xf32>, tensor<1x1xf32>) -> tensor<1x3xf32>
  %120 = "tf.Mul"(%119, %cst_55) {device = ""} : (tensor<1x3xf32>, tensor<3xf32>) -> tensor<1x3xf32>
  %121 = "tf.Mul"(%118, %120) {device = ""} : (tensor<1x1xf32>, tensor<1x3xf32>) -> tensor<1x3xf32>
  %122 = "tf.AddV2"(%121, %cst_54) {device = ""} : (tensor<1x3xf32>, tensor<3xf32>) -> tensor<1x3xf32>
  func.return %122 : tensor<1x3xf32>
}
// CHECK-LABEL: func.func @layer_norm_V3_disable_minimize_broadcast
// CHECK: mhlo.custom_call
// CHECK-SAME: @byteir.layer_norm

func.func @layer_norm_V4(%1735: tensor<128x20x8xf16>) -> tensor<128x20x8xf16> {
  %cst_340 = "tf.Const"() {value = dense<-1> : tensor<1xi32>} : () -> tensor<1xi32>
  %cst_341 = "tf.Const"() {value = dense<[9.985350e-01, 9.995110e-01, 9.990230e-01, 9.980460e-01, 9.965820e-01, 9.990230e-01, 9.980460e-01, 1.000000e+00]> : tensor<8xf16>} : () -> tensor<8xf16>
  %cst_342 = "tf.Const"() {value = dense<[-6.313320e-04, -2.541540e-04, 2.572540e-04, -9.641640e-04, 6.055830e-04, 5.350110e-04, 1.099590e-03, -4.178290e-05]> : tensor<8xf16>} : () -> tensor<8xf16>
  %cst_343 = "tf.Const"() {value = dense<1.013280e-06> : tensor<f16>} : () -> tensor<f16>
  %1736 = "tf.Mean"(%1735, %cst_340) {device = "", keep_dims = true} : (tensor<128x20x8xf16>, tensor<1xi32>) -> tensor<128x20x1xf16>
  %1737 = "tf.SquaredDifference"(%1735, %1736) {device = ""} : (tensor<128x20x8xf16>, tensor<128x20x1xf16>) -> tensor<128x20x8xf16>
  %1738 = "tf.Mean"(%1737, %cst_340) {device = "", keep_dims = true} : (tensor<128x20x8xf16>, tensor<1xi32>) -> tensor<128x20x1xf16>
  %1739 = "tf.AddV2"(%1738, %cst_343) {device = ""} : (tensor<128x20x1xf16>, tensor<f16>) -> tensor<128x20x1xf16>
  %1740 = "tf.Rsqrt"(%1739) {device = ""} : (tensor<128x20x1xf16>) -> tensor<128x20x1xf16>
  %1741 = "tf.Sub"(%1735, %1736) {device = ""} : (tensor<128x20x8xf16>, tensor<128x20x1xf16>) -> tensor<128x20x8xf16>
  %1742 = "tf.Mul"(%1740, %1741) {device = ""} : (tensor<128x20x1xf16>, tensor<128x20x8xf16>) -> tensor<128x20x8xf16>
  %1743 = "tf.Mul"(%1742, %cst_341) {device = ""} : (tensor<128x20x8xf16>, tensor<8xf16>) -> tensor<128x20x8xf16>
  %1744 = "tf.AddV2"(%1743, %cst_342) {device = ""} : (tensor<128x20x8xf16>, tensor<8xf16>) -> tensor<128x20x8xf16>
  return %1744 : tensor<128x20x8xf16>
}
// CHECK-LABEL: func.func @layer_norm_V4
// CHECK: mhlo.custom_call
// CHECK-SAME: @byteir.layer_norm

func.func @layer_norm_V4_swap_squarediff(%1735: tensor<128x20x8xf16>) -> tensor<128x20x8xf16> {
  %cst_340 = "tf.Const"() {value = dense<-1> : tensor<1xi32>} : () -> tensor<1xi32>
  %cst_341 = "tf.Const"() {value = dense<[9.985350e-01, 9.995110e-01, 9.990230e-01, 9.980460e-01, 9.965820e-01, 9.990230e-01, 9.980460e-01, 1.000000e+00]> : tensor<8xf16>} : () -> tensor<8xf16>
  %cst_342 = "tf.Const"() {value = dense<[-6.313320e-04, -2.541540e-04, 2.572540e-04, -9.641640e-04, 6.055830e-04, 5.350110e-04, 1.099590e-03, -4.178290e-05]> : tensor<8xf16>} : () -> tensor<8xf16>
  %cst_343 = "tf.Const"() {value = dense<1.013280e-06> : tensor<f16>} : () -> tensor<f16>
  %1736 = "tf.Mean"(%1735, %cst_340) {device = "", keep_dims = true} : (tensor<128x20x8xf16>, tensor<1xi32>) -> tensor<128x20x1xf16>
  %1737 = "tf.SquaredDifference"(%1736, %1735) {device = ""} : (tensor<128x20x1xf16>, tensor<128x20x8xf16>) -> tensor<128x20x8xf16>
  %1738 = "tf.Mean"(%1737, %cst_340) {device = "", keep_dims = true} : (tensor<128x20x8xf16>, tensor<1xi32>) -> tensor<128x20x1xf16>
  %1739 = "tf.AddV2"(%1738, %cst_343) {device = ""} : (tensor<128x20x1xf16>, tensor<f16>) -> tensor<128x20x1xf16>
  %1740 = "tf.Rsqrt"(%1739) {device = ""} : (tensor<128x20x1xf16>) -> tensor<128x20x1xf16>
  %1741 = "tf.Sub"(%1735, %1736) {device = ""} : (tensor<128x20x8xf16>, tensor<128x20x1xf16>) -> tensor<128x20x8xf16>
  %1742 = "tf.Mul"(%1740, %1741) {device = ""} : (tensor<128x20x1xf16>, tensor<128x20x8xf16>) -> tensor<128x20x8xf16>
  %1743 = "tf.Mul"(%1742, %cst_341) {device = ""} : (tensor<128x20x8xf16>, tensor<8xf16>) -> tensor<128x20x8xf16>
  %1744 = "tf.AddV2"(%1743, %cst_342) {device = ""} : (tensor<128x20x8xf16>, tensor<8xf16>) -> tensor<128x20x8xf16>
  return %1744 : tensor<128x20x8xf16>
}
// CHECK-LABEL: func.func @layer_norm_V4_swap_squarediff
// CHECK: mhlo.custom_call
// CHECK-SAME: @byteir.layer_norm

func.func @layer_norm_with_cast(%79: tensor<150x3xf16>) -> tensor<150x3xf16> {
  %cst_61 = "tf.Const"() {value = dense<[0.0401659757, -0.11370486, 0.432680517]> : tensor<3xf16>} : () -> tensor<3xf16>
  %cst_62 = "tf.Const"() {value = dense<[0.445568085, 0.45303449, 3.227140e-01]> : tensor<3xf16>} : () -> tensor<3xf16>
  %cst_157 = "tf.Const"() {value = dense<1.013280e-06> : tensor<f16>} : () -> tensor<f16>
  %cst_158 = "tf.Const"() {value = dense<-1> : tensor<i32>} : () -> tensor<i32>
  %80 = "tf.Cast"(%79) {Truncate = false, device = ""} : (tensor<150x3xf16>) -> tensor<150x3xf32>
  %81 = "tf.Mean"(%80, %cst_158) {device = "", keep_dims = true} : (tensor<150x3xf32>, tensor<i32>) -> tensor<150x1xf32>
  %82 = "tf.Cast"(%81) {Truncate = false, device = ""} : (tensor<150x1xf32>) -> tensor<150x1xf16>
  %83 = "tf.SquaredDifference"(%80, %81) {device = ""} : (tensor<150x3xf32>, tensor<150x1xf32>) -> tensor<150x3xf32>
  %84 = "tf.Mean"(%83, %cst_158) {device = "", keep_dims = true} : (tensor<150x3xf32>, tensor<i32>) -> tensor<150x1xf32>
  %85 = "tf.Cast"(%84) {Truncate = false, device = ""} : (tensor<150x1xf32>) -> tensor<150x1xf16>
  %86 = "tf.AddV2"(%85, %cst_157) {device = ""} : (tensor<150x1xf16>, tensor<f16>) -> tensor<150x1xf16>
  %87 = "tf.Rsqrt"(%86) {device = ""} : (tensor<150x1xf16>) -> tensor<150x1xf16>
  %88 = "tf.Mul"(%87, %cst_61) {_grappler_ArithmeticOptimizer_MinimizeBroadcasts = true, device = ""} : (tensor<150x1xf16>, tensor<3xf16>) -> tensor<150x3xf16>
  %89 = "tf.Sub"(%79, %82) {device = ""} : (tensor<150x3xf16>, tensor<150x1xf16>) -> tensor<150x3xf16>
  %90 = "tf.Mul"(%89, %88) {_grappler_ArithmeticOptimizer_MinimizeBroadcasts = true, device = ""} : (tensor<150x3xf16>, tensor<150x3xf16>) -> tensor<150x3xf16>
  %91 = "tf.AddV2"(%90, %cst_62) {device = ""} : (tensor<150x3xf16>, tensor<3xf16>) -> tensor<150x3xf16>
  return %91 : tensor<150x3xf16>
}
// CHECK-LABEL:  func.func @layer_norm_with_cast(%arg0: tensor<150x3xf16>) -> tensor<150x3xf16> {
// CHECK-NEXT:    %cst = "tf.Const"() <{value = dense<[4.016110e-02, -1.137080e-01, 4.326170e-01]> : tensor<3xf16>}> : () -> tensor<3xf16>
// CHECK-NEXT:    %cst_0 = "tf.Const"() <{value = dense<[4.455570e-01, 4.531250e-01, 3.227540e-01]> : tensor<3xf16>}> : () -> tensor<3xf16>
// CHECK: mhlo.custom_call
// CHECK-SAME: @byteir.layer_norm
// CHECK-SAME: byteir_attrs = {axis = [1], epsilon = 1.0132789611816406E-6 : f64}

func.func @layer_norm_with_cast_v2(%79: tensor<150x3xf16>) -> tensor<150x3xf16> {
  %cst_61 = "tf.Const"() {value = dense<[0.0401659757, -0.11370486, 0.432680517]> : tensor<3xf16>} : () -> tensor<3xf16>
  %cst_62 = "tf.Const"() {value = dense<[0.445568085, 0.45303449, 3.227140e-01]> : tensor<3xf16>} : () -> tensor<3xf16>
  %cst_157 = "tf.Const"() {value = dense<1.013280e-06> : tensor<f16>} : () -> tensor<f16>
  %cst_158 = "tf.Const"() {value = dense<-1> : tensor<i32>} : () -> tensor<i32>
  %80 = "tf.Cast"(%79) {Truncate = false, device = ""} : (tensor<150x3xf16>) -> tensor<150x3xf32>
  %81 = "tf.Mean"(%80, %cst_158) {device = "", keep_dims = true} : (tensor<150x3xf32>, tensor<i32>) -> tensor<150x1xf32>
  %82 = "tf.Cast"(%81) {Truncate = false, device = ""} : (tensor<150x1xf32>) -> tensor<150x1xf16>
  %83 = "tf.SquaredDifference"(%80, %81) {device = ""} : (tensor<150x3xf32>, tensor<150x1xf32>) -> tensor<150x3xf32>
  %84 = "tf.Mean"(%83, %cst_158) {device = "", keep_dims = true} : (tensor<150x3xf32>, tensor<i32>) -> tensor<150x1xf32>
  %85 = "tf.Cast"(%84) {Truncate = false, device = ""} : (tensor<150x1xf32>) -> tensor<150x1xf16>
  %86 = "tf.AddV2"(%85, %cst_157) {device = ""} : (tensor<150x1xf16>, tensor<f16>) -> tensor<150x1xf16>
  %87 = "tf.Rsqrt"(%86) {device = ""} : (tensor<150x1xf16>) -> tensor<150x1xf16>
  %88 = "tf.Mul"(%87, %cst_61) {_grappler_ArithmeticOptimizer_MinimizeBroadcasts = true, device = ""} : (tensor<150x1xf16>, tensor<3xf16>) -> tensor<150x3xf16>
  %89 = "tf.Mul"(%79, %88) {_grappler_ArithmeticOptimizer_MinimizeBroadcasts = true, device = ""} : (tensor<150x3xf16>, tensor<150x3xf16>) -> tensor<150x3xf16>
  %90 = "tf.Mul"(%88, %82) {_grappler_ArithmeticOptimizer_MinimizeBroadcasts = true, device = ""} : (tensor<150x3xf16>, tensor<150x1xf16>) -> tensor<150x3xf16>
  %91 = "tf.Sub"(%cst_62, %90) {device = ""} : (tensor<3xf16>, tensor<150x3xf16>) -> tensor<150x3xf16>
  %92 = "tf.AddV2"(%89, %91) {device = ""} : (tensor<150x3xf16>, tensor<150x3xf16>) -> tensor<150x3xf16>
  return %92 : tensor<150x3xf16>
}
// CHECK-LABEL:  func.func @layer_norm_with_cast_v2(%arg0: tensor<150x3xf16>) -> tensor<150x3xf16> {
// CHECK-NEXT:    %cst = "tf.Const"() <{value = dense<[4.016110e-02, -1.137080e-01, 4.326170e-01]> : tensor<3xf16>}> : () -> tensor<3xf16>
// CHECK-NEXT:    %cst_0 = "tf.Const"() <{value = dense<[4.455570e-01, 4.531250e-01, 3.227540e-01]> : tensor<3xf16>}> : () -> tensor<3xf16>
// CHECK: mhlo.custom_call
// CHECK-SAME: @byteir.layer_norm
// CHECK-SAME: byteir_attrs = {axis = [1], epsilon = 1.0132789611816406E-6 : f64}

func.func @layer_norm_with_cast_disable_minimize_broadcast(%46: tensor<1024x4xf16>) -> tensor<1024x4xf16> {
  %cst_103 = "tf.Const"() {value = dense<[0.0401659757, -0.11370486, 0.432680517, 0.4000000]> : tensor<4xf16>} : () -> tensor<4xf16>
  %cst_104 = "tf.Const"() {value = dense<[0.445568085, 0.45303449, 3.227140e-01, 0.4000000]> : tensor<4xf16>} : () -> tensor<4xf16>
  %cst_119 = "tf.Const"() {value = dense<1.013280e-06> : tensor<f16>} : () -> tensor<f16>
  %cst_120 = "tf.Const"() {value = dense<-1> : tensor<i32>} : () -> tensor<i32>
  %47 = "tf.Cast"(%46) {Truncate = false, device = ""} : (tensor<1024x4xf16>) -> tensor<1024x4xf32>
  %48 = "tf.Mean"(%47, %cst_120) {device = "", keep_dims = true} : (tensor<1024x4xf32>, tensor<i32>) -> tensor<1024x1xf32>
  %49 = "tf.Cast"(%48) {Truncate = false, device = ""} : (tensor<1024x1xf32>) -> tensor<1024x1xf16>
  %50 = "tf.SquaredDifference"(%47, %48) {device = ""} : (tensor<1024x4xf32>, tensor<1024x1xf32>) -> tensor<1024x4xf32>
  %51 = "tf.Mean"(%50, %cst_120) {device = "", keep_dims = true} : (tensor<1024x4xf32>, tensor<i32>) -> tensor<1024x1xf32>
  %52 = "tf.Cast"(%51) {Truncate = false, device = ""} : (tensor<1024x1xf32>) -> tensor<1024x1xf16>
  %53 = "tf.AddV2"(%52, %cst_119) {device = ""} : (tensor<1024x1xf16>, tensor<f16>) -> tensor<1024x1xf16>
  %54 = "tf.Rsqrt"(%53) {device = ""} : (tensor<1024x1xf16>) -> tensor<1024x1xf16>
  %55 = "tf.Sub"(%46, %49) {device = ""} : (tensor<1024x4xf16>, tensor<1024x1xf16>) -> tensor<1024x4xf16> 
  %56 = "tf.Mul"(%54, %55) {device = ""} : (tensor<1024x1xf16>, tensor<1024x4xf16>) -> tensor<1024x4xf16>
  %57 = "tf.Mul"(%56, %cst_103) {device = ""} : (tensor<1024x4xf16>, tensor<4xf16>) -> tensor<1024x4xf16>
  %58 = "tf.AddV2"(%57, %cst_104) {device = ""} : (tensor<1024x4xf16>, tensor<4xf16>) -> tensor<1024x4xf16>
  return %58 : tensor<1024x4xf16>
}
// CHECK-LABEL: func.func @layer_norm_with_cast_disable_minimize_broadcast
// CHECK: mhlo.custom_call
// CHECK-SAME: @byteir.layer_norm

func.func @l2_norm_V1(%arg0: tensor<1x32x3xf32>) -> tensor<1x32x3xf32> {
  %cst = "tf.Const"() {value = dense<9.99999997E-7> : tensor<f32>} : () -> tensor<f32>
  %cst_0 = "tf.Const"() {value = dense<-1> : tensor<1xi32>} : () -> tensor<1xi32>
  %0 = "tf.Square"(%arg0) : (tensor<1x32x3xf32>) -> tensor<1x32x3xf32>
  %1 = "tf.Sum"(%0, %cst_0) {keep_dims = true} : (tensor<1x32x3xf32>, tensor<1xi32>) -> tensor<1x32x1xf32>
  %2 = "tf.Maximum"(%1, %cst): (tensor<1x32x1xf32>, tensor<f32>) -> tensor<1x32x1xf32>
  %3 = "tf.Rsqrt"(%2) : (tensor<1x32x1xf32>) -> tensor<1x32x1xf32>
  %4 = "tf.Mul"(%arg0, %3) : (tensor<1x32x3xf32>, tensor<1x32x1xf32>) -> tensor<1x32x3xf32>
  func.return %4 : tensor<1x32x3xf32>
}
// CHECK-LABEL:  func.func @l2_norm_V1(%arg0: tensor<1x32x3xf32>) -> tensor<1x32x3xf32> {
// CHECK: mhlo.custom_call
// CHECK-SAME: @byteir.l2_norm
// CHECK-SAME: byteir_attrs = {axis = [2], epsilon = 9.9999999747524271E-7 : f64}

func.func @l2_norm_V1_with_multiplyer(%arg0: tensor<2x4x8xf32>) -> tensor<2x4x8xf32> {
  %cst = "tf.Const"() <{value = dense<-1> : tensor<i32>}> : () -> tensor<i32>
  %cst_1 = "tf.Const"() <{value = dense<9.99999996E-13> : tensor<f32>}> : () -> tensor<f32>
  %cst_2 = "tf.Const"() <{value = dense<9.34093475> : tensor<1xf32>}> : () -> tensor<1xf32>
  %0 = "tf.Square"(%arg0) {device = ""} : (tensor<2x4x8xf32>) -> tensor<2x4x8xf32>
  %1 = "tf.Sum"(%0, %cst) <{keep_dims = true}> {device = ""} : (tensor<2x4x8xf32>, tensor<i32>) -> tensor<2x4x1xf32>
  %2 = "tf.Maximum"(%1, %cst_1) {device = ""} : (tensor<2x4x1xf32>, tensor<f32>) -> tensor<2x4x1xf32>
  %3 = "tf.Rsqrt"(%2) {device = ""} : (tensor<2x4x1xf32>) -> tensor<2x4x1xf32>
  %4 = "tf.Mul"(%3, %cst_2) {device = ""} : (tensor<2x4x1xf32>, tensor<1xf32>) -> tensor<2x4x1xf32>
  %5 = "tf.Mul"(%arg0, %4) {device = ""} : (tensor<2x4x8xf32>, tensor<2x4x1xf32>) -> tensor<2x4x8xf32>
  func.return %5 : tensor<2x4x8xf32>
}
// CHECK-LABEL:  func.func @l2_norm_V1_with_multiplyer(%arg0: tensor<2x4x8xf32>) -> tensor<2x4x8xf32> {
// CHECK-NEXT:  %0 = mhlo.constant dense<9.34093475> : tensor<2x4x8xf32>
// CHECK-NEXT:  %1 = mhlo.custom_call @byteir.l2_norm(%arg0) {backend_config = "", byteir_attrs = {axis = [2], epsilon = 9.999999960041972E-13 : f64}} : (tensor<2x4x8xf32>) -> tensor<2x4
// CHECK-NEXT:  %2 = mhlo.multiply %1, %0 : tensor<2x4x8xf32>

func.func @l2_norm_V1_swap_mul(%54: tensor<1x64xf32>) -> tensor<1x64xf32> {
  %cst_0 = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %cst_76 = "tf.Const"() {value = dense<9.99999993E-9> : tensor<f32>} : () -> tensor<f32>
  %55 = "tf.Square"(%54) {device = ""} : (tensor<1x64xf32>) -> tensor<1x64xf32>
  %56 = "tf.Sum"(%55, %cst_0) {device = "", keep_dims = true} : (tensor<1x64xf32>, tensor<i32>) -> tensor<1x1xf32>
  %57 = "tf.Maximum"(%56, %cst_76) {device = ""} : (tensor<1x1xf32>, tensor<f32>) -> tensor<1x1xf32>
  %58 = "tf.Rsqrt"(%57) {device = ""} : (tensor<1x1xf32>) -> tensor<1x1xf32>
  %59 = "tf.Mul"(%58, %54) {device = ""} : (tensor<1x1xf32>, tensor<1x64xf32>) -> tensor<1x64xf32>
  func.return %59 : tensor<1x64xf32>
}
// CHECK-LABEL: func.func @l2_norm_V1_swap_mul
// CHECK: mhlo.custom_call
// CHECK-SAME: @byteir.l2_norm

func.func @l2_norm_V2(%1871: tensor<1x128xf16>) -> tensor<1x128xf16> {
  %cst_5 = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %1872 = "tf.Square"(%1871) {device = ""} : (tensor<1x128xf16>) -> tensor<1x128xf16>
  %1873 = "tf.Sum"(%1872, %cst_5) {device = "", keep_dims = true} : (tensor<1x128xf16>, tensor<i32>) -> tensor<1x1xf16>
  %1874 = "tf.Relu"(%1873) : (tensor<1x1xf16>) -> tensor<1x1xf16>
  %1875 = "tf.Rsqrt"(%1874) {device = ""} : (tensor<1x1xf16>) -> tensor<1x1xf16>
  %1876 = "tf.Mul"(%1875, %1871) {device = ""} : (tensor<1x1xf16>, tensor<1x128xf16>) -> tensor<1x128xf16>
  return %1876 : tensor<1x128xf16>
}
// CHECK-LABEL: @l2_norm_V2
// CHECK: mhlo.custom_call
// CHECK-SAME: @byteir.l2_norm
// CHECK-SAME: byteir_attrs = {axis = [1], epsilon = 0.000000e+00 : f64}

func.func @l2_norm_V2_swap_mul(%1871: tensor<1x128xf16>) -> tensor<1x128xf16> {
  %cst_5 = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %1872 = "tf.Square"(%1871) {device = ""} : (tensor<1x128xf16>) -> tensor<1x128xf16>
  %1873 = "tf.Sum"(%1872, %cst_5) {device = "", keep_dims = true} : (tensor<1x128xf16>, tensor<i32>) -> tensor<1x1xf16>
  %1874 = "tf.Relu"(%1873) : (tensor<1x1xf16>) -> tensor<1x1xf16>
  %1875 = "tf.Rsqrt"(%1874) {device = ""} : (tensor<1x1xf16>) -> tensor<1x1xf16>
  %1876 = "tf.Mul"(%1871, %1875) {device = ""} : (tensor<1x128xf16>, tensor<1x1xf16>) -> tensor<1x128xf16>
  return %1876 : tensor<1x128xf16>
}
// CHECK-LABEL: @l2_norm_V2_swap_mul
// CHECK: mhlo.custom_call
// CHECK-SAME: @byteir.l2_norm
// CHECK-SAME: byteir_attrs = {axis = [1], epsilon = 0.000000e+00 : f64}

func.func @l2_norm_V3(%15: tensor<1x100x512xf32>) -> tensor<1x100x512xf32> {
  %cst_96 = "tf.Const"() {value = dense<2> : tensor<1xi32>} : () -> tensor<1xi32>
  %16 = "tf.Square"(%15) {device = ""} : (tensor<1x100x512xf32>) -> tensor<1x100x512xf32>
  %17 = "tf.Sum"(%16, %cst_96) {device = "", keep_dims = true} : (tensor<1x100x512xf32>, tensor<1xi32>) -> tensor<1x100x1xf32>
  %18 = "tf.Rsqrt"(%17) {device = ""} : (tensor<1x100x1xf32>) -> tensor<1x100x1xf32>
  %19 = "tf.Mul"(%15, %18) {device = ""} : (tensor<1x100x512xf32>, tensor<1x100x1xf32>) -> tensor<1x100x512xf32>
  return %19 : tensor<1x100x512xf32>
}
// CHECK-LABEL: @l2_norm_V3
// CHECK: mhlo.custom_call
// CHECK-SAME: @byteir.l2_norm
// CHECK-SAME: byteir_attrs = {axis = [2], epsilon = 0.000000e+00 : f64}

func.func @dynamic_mask_stitch(%arg0: tensor<4x4xf32>, %arg1: tensor<4xi32>) -> tensor<?x4xf32> {
  %cst = "tf.Const"() {value = dense<[[-0.916170597, -0.884184718, 1.60242105, -1.19678485], [0.33643803, -0.431175768, 1.71861267, 0.126368985], [-1.07191086, -1.00517535, -0.666032254, 0.776807785], [1.53380013, 0.83925873, -0.24277249, 1.53341103]]> : tensor<4x4xf32>} : () -> tensor<4x4xf32>
  %cst_0 = "tf.Const"() {value = dense<[0.553816557, -0.920699775, 0.418103188, -0.261674613]> : tensor<4xf32>} : () -> tensor<4xf32>
  %cst_1 = "tf.Const"() {value = dense<[[-0.705530286, 0.87041223, 0.972314774, -0.0584422052], [-1.43617868, 6.772900e-01, 0.880922436, 0.56821847], [0.57929492, 0.470399499, -1.0485183, -1.27004325], [-0.32425791, 1.88410747, 0.220974803, -0.238485783]]> : tensor<4x4xf32>} : () -> tensor<4x4xf32>
  %cst_2 = "tf.Const"() {value = dense<[2.63629675, 2.68127704, 2.14741468, -1.6519475]> : tensor<4xf32>} : () -> tensor<4xf32>
  %cst_3 = "tf.Const"() {value = dense<[[-1.87686706, 0.286330104, -0.044809185, -0.178677231], [-1.14233077, -0.446333855, -1.2957921, 0.446576297], [0.985618114, 0.699275255, 0.609199941, -0.726590812], [0.0366623849, -0.640842735, -1.72003555, -0.383472085]]> : tensor<4x4xf32>} : () -> tensor<4x4xf32>
  %cst_4 = "tf.Const"() {value = dense<[0.478572756, 0.458867788, -1.44476604, 0.189240679]> : tensor<4xf32>} : () -> tensor<4xf32>
  %cst_5 = "tf.Const"() {value = dense<[[0.473984271, 0.173930168, 0.465745121, 1.14254773], [-0.384602815, -0.673360229, 1.13109767, 0.761463344], [-0.171464354, -0.908823907, 1.19337058, -1.78143835], [1.40376866, -0.529214859, -1.9030931, 1.25083804]]> : tensor<4x4xf32>} : () -> tensor<4x4xf32>
  %cst_6 = "tf.Const"() {value = dense<[1.56364501, -0.948736965, 0.0843383893, 0.502355933]> : tensor<4xf32>} : () -> tensor<4xf32>
  %0 = "tf.MatMul"(%arg0, %cst) {device = "", transpose_a = false, transpose_b = false} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  %1 = "tf.AddV2"(%cst_0, %0) {device = ""} : (tensor<4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  %2:2 = "tf.DynamicPartition"(%1, %arg1) {T = f32, device = "", num_partitions = 2 : i64} : (tensor<4x4xf32>, tensor<4xi32>) -> (tensor<?x4xf32>, tensor<?x4xf32>)
  %3 = "tf.MatMul"(%2#0, %cst_1) {device = "", transpose_a = false, transpose_b = false} : (tensor<?x4xf32>, tensor<4x4xf32>) -> tensor<?x4xf32>
  %4 = "tf.AddV2"(%cst_2, %3) {device = ""} : (tensor<4xf32>, tensor<?x4xf32>) -> tensor<?x4xf32>
  %5 = "tf.MatMul"(%2#1, %cst_3) {device = "", transpose_a = false, transpose_b = false} : (tensor<?x4xf32>, tensor<4x4xf32>) -> tensor<?x4xf32>
  %6 = "tf.AddV2"(%cst_4, %5) {device = ""} : (tensor<4xf32>, tensor<?x4xf32>) -> tensor<?x4xf32>
  %cst_7 = "tf.Const"() {value = dense<[0, 1, 2, 3]> : tensor<4xi32>} : () -> tensor<4xi32>
  %7:2 = "tf.DynamicPartition"(%cst_7, %arg1) {T = i32, device = "", num_partitions = 2 : i64} : (tensor<4xi32>, tensor<4xi32>) -> (tensor<?xi32>, tensor<?xi32>)
  %8 = "tf.DynamicStitch"(%7#0, %7#1, %4, %6) {device = ""} : (tensor<?xi32>, tensor<?xi32>, tensor<?x4xf32>, tensor<?x4xf32>) -> tensor<?x4xf32>
  %9 = "tf.MatMul"(%8, %cst_5) {device = "", transpose_a = false, transpose_b = false} : (tensor<?x4xf32>, tensor<4x4xf32>) -> tensor<?x4xf32>
  %10 = "tf.AddV2"(%cst_6, %9) {device = ""} : (tensor<4xf32>, tensor<?x4xf32>) -> tensor<?x4xf32>
  func.return %10 : tensor<?x4xf32>
}
// CHECK-LABEL: func.func @dynamic_mask_stitch
// CHECK: mhlo.custom_call
// CHECK-SAME: @tf.DynamicPartition
// CHECK-SAME: byteir_attrs = {num_partitions = 2 : i64}
// CHECK: mhlo.custom_call
// CHECK-SAME: @tf.DynamicMaskStitch
// CHECK-SAME: byteir_attrs = {}

func.func @dynamic_partition(%1: tensor<4x4xf32>, %arg1: tensor<4xi32>) -> (tensor<?x4xf32>, tensor<?x4xf32>) {
  %2:2 = "tf.DynamicPartition"(%1, %arg1) {T = f32, device = "", num_partitions = 2 : i64} : (tensor<4x4xf32>, tensor<4xi32>) -> (tensor<?x4xf32>, tensor<?x4xf32>)
  return %2#0, %2#1 : tensor<?x4xf32>, tensor<?x4xf32>
}
// CHECK-LABEL: func.func @dynamic_partition
// CHECK: mhlo.custom_call
// CHECK-SAME: @tf.DynamicPartition
// CHECK-SAME: byteir_attrs = {num_partitions = 2 : i64}

func.func @dynamic_stitch(%part0: tensor<?xi32>, %part1: tensor<?xi32>, %4: tensor<?x4xf32>, %6: tensor<?x4xf32>) -> tensor<?x4xf32> {
  %8 = "tf.DynamicStitch"(%part0, %part1, %4, %6) {device = ""} : (tensor<?xi32>, tensor<?xi32>, tensor<?x4xf32>, tensor<?x4xf32>) -> tensor<?x4xf32>
  return %8 : tensor<?x4xf32>
}
// CHECK-LABEL: func.func @dynamic_stitch
// CHECK: mhlo.custom_call
// CHECK-SAME: @tf.DynamicStitch
// CHECK-SAME: byteir_attrs = {}

func.func @onehot_case0(%arg0: tensor<150xi32>) -> tensor<150x16xf32> {
  %off_value = "tf.Const"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<f32>
  %depth = "tf.Const"() {value = dense<16> : tensor<i32>} : () -> tensor<i32>
  %on_value = "tf.Const"() {value = dense<1.000000e+00> : tensor<f32>} : () -> tensor<f32>
  %0 = "tf.OneHot"(%arg0, %depth, %on_value, %off_value) {axis = -1 : i64, device = ""} : (tensor<150xi32>, tensor<i32>, tensor<f32>, tensor<f32>) -> tensor<150x16xf32>
  return %0 : tensor<150x16xf32>
}
// CHECK-LABEL: func.func @onehot_case0(%arg0: tensor<150xi32>) -> tensor<150x16xf32> {
// CHECK: mhlo.custom_call
// CHECK-SAME: @byteir.one_hot
// CHECK-SAME: byteir_attrs = {axis = 1 : i64, depth = 16 : i64, off_value = 0.000000e+00 : f32, on_value = 1.000000e+00 : f32}
