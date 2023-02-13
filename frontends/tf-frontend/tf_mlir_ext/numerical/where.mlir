// RUN: tf-ext-opt --canonicalize %s -o %t1
// RUN: FileCheck %s < %t1
// RUN: python3 numerical_test.py %s %t1
// RUN: tf-ext-opt --canonicalize --constant-folding --cse %s -o %t2
// RUN: FileCheck %s < %t2 --check-prefix AFTER
// RUN: python3 numerical_test.py %s %t2

func.func @where() -> tensor<2x1xi64> {
    %cst = "tf.Const"() {value = dense<"0x03"> : tensor<8xi1>} : () -> tensor<8xi1>
    %0 = "tf.Where"(%cst) {device = ""} : (tensor<8xi1>) -> tensor<2x1xi64>
    func.return %0 : tensor<2x1xi64>
}
// CHECK-LABEL: func.func @where
// CHECK-NEXT{{LITERAL}}:  %cst = "tf.Const"() {value = dense<[[0], [1]]> : tensor<2x1xi64>} : () -> tensor<2x1xi64>

func.func @where_splat() -> tensor<9x1xi64> {
    %cst = "tf.Const"() {value = dense<true> : tensor<9xi1>} : () -> tensor<9xi1>
    %0 = "tf.Where"(%cst) {device = ""} : (tensor<9xi1>) -> tensor<9x1xi64>
    func.return %0 : tensor<9x1xi64>
}
// CHECK-LABEL: func.func @where_splat
// CHECK-NEXT{{LITERAL}}:  %cst = "tf.Const"() {value = dense<[[0], [1], [2], [3], [4], [5], [6], [7], [8]]> : tensor<9x1xi64>} : () -> tensor<9x1xi64>

func.func @main() -> tensor<161280x1xi64> {
    %cst_54 = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
    %cst_55 = "tf.Const"() {value = dense<-1> : tensor<1xi32>} : () -> tensor<1xi32>
    %cst_77 = "tf.Const"() {value = dense<"0x008000C000E000F000F800FC00FE00FF80FFC0FFE0FFF0FFF8FFFCFF00"> : tensor<1x225xi1>} : () -> tensor<1x225xi1>
    %cst_80 = "tf.Const"() {value = dense<12> : tensor<i32>} : () -> tensor<i32>
    %cst_170 = "tf.Const"() {value = dense<[128, 32]> : tensor<2xi32>} : () -> tensor<2xi32>
    %cst_172 = "tf.Const"() {value = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
    %cst_173 = "tf.Const"() {value = dense<1> : tensor<1xi32>} : () -> tensor<1xi32>
    %364 = "tf.StridedSlice"(%cst_170, %cst_172, %cst_173, %cst_173) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i32>
    %371 = "tf.Mul"(%cst_80, %364) {device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %375 = "tf.Pack"(%371, %cst_54) {axis = 0 : i64, device = ""} : (tensor<i32>, tensor<i32>) -> tensor<2xi32>
    %376 = "tf.Tile"(%cst_77, %375) {device = ""} : (tensor<1x225xi1>, tensor<2xi32>) -> tensor<1536x225xi1>
    %377 = "tf.Reshape"(%376, %cst_55) {device = ""} : (tensor<1536x225xi1>, tensor<1xi32>) -> tensor<345600xi1>
    %378 = "tf.Where"(%377) {device = ""} : (tensor<345600xi1>) -> tensor<161280x1xi64>
    func.return %378 : tensor<161280x1xi64>
}
// CHECK-LABEL: func.func @main
// CHECK:  tf.Where

// AFTER-LABEL: func.func @main
// AFTER-NOT:  tf.Where
