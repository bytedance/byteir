// RUN: byteir-opt %s -collect-func="anchor-attr=testAttr" | FileCheck %s


func.func private @test_private() {
    return
}
// CHECK-LABEL: func.func private @test_private() 

func.func @test1() attributes {testAttr} {
    return
}
// CHECK-LABEL: func.func @test1() attributes {testAttr}

func.func @test2() attributes {testAttr2} {
    return
}
// CHECK-NOT: func.func @test2() 

