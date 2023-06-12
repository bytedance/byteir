// RUN: byteir-opt %s -collect-func="anchor-attr=testAttr" | FileCheck %s


func.func private @test_private1() {
    return
}
// CHECK-LABEL: func.func private @test_private1() 

func.func private @test_private2() {
    return
}
// CHECK-NOT: func.func private @test_private2() 

func.func @test1() attributes {testAttr} {
    call @test_private1() : () -> ()
    return
}
// CHECK-LABEL: func.func @test1() attributes {testAttr}

func.func @test2() attributes {testAttr2} {
    call @test_private2() : () -> ()
    return
}
// CHECK-NOT: func.func @test2() 

