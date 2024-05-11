// RUN: byteir-opt %s -remove-func-tag="attr-name=libfile func-name=test" | FileCheck %s -check-prefix=PAT0
// RUN: byteir-opt %s -remove-func-tag="attr-name=libfile func-name=test1" | FileCheck %s -check-prefix=PAT1
// RUN: byteir-opt %s -remove-func-tag="attr-name=libfile func-name=test2" | FileCheck %s -check-prefix=PAT2


module attributes {byre.container_module} {
  func.func @test() attributes {byre.entry_point, libfile = "kernel.ll.bc"} {
      return
  }

  // DEFAULT: func.func @test() attributes {testAttr}
  // UNIT: func.func @test() attributes {testAttr}
  // STRING: func.func @test() attributes {testAttr = "test"}
  // I32: func.func @test() attributes {testAttr = 5 : i32}
  // F32: func.func @test() attributes {testAttr = 2.500000e+00 : f32}
  // ANCHOR: func.func @test()

  func.func @test1() attributes {libfile = "kernel.ll.bc"} {
      return
  }

  // ANCHOR: func.func @test2() attributes {testAnchor, testAttr}

  func.func @test2() attributes {byre.entry_point = "forward"} {
      return
  }
}

// PAT0: func.func @test() attributes {byre.entry_point} {
// PAT1: func.func @test1() {
// PAT2: func.func @test2() attributes {byre.entry_point = "forward"} {
