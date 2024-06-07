// RUN: byteir-opt %s -apply-pdl-patterns="pdl-file=%S/Pattern_0.mlir" -allow-unregistered-dialect | FileCheck %s

func.func @foo(%arg0: index) -> index {
    // CHECK: test.test_op_B
    %0 = "test.test_op_A"(%arg0) {__rewrite__} : (index) -> index
    // CHECK: test.test_op_A
    %1 = "test.test_op_A"(%0) : (index) -> index
    return %1 : index
}