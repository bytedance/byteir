// RUN: byteir-opt %s -remove-func-body="anchor-attr=testAttr" | FileCheck %s -check-prefix=DEFAULT
// RUN: byteir-opt %s -remove-func-body="anchor-attr=testAttr disable-force-private" | FileCheck %s -check-prefix=DISALBE


func.func private @test1() attributes {testAttr}  {
    return
}
// DEFAULT-LABEL: func.func private @test1()
// DEFAULT-NOT: return
// DISALBE-LABEL: func.func private @test1()
// DISALBE-NOT: return

func.func nested @test2() attributes {testAttr}  {
    return
}
// DEFAULT-LABEL: func.func nested @test2()
// DEFAULT-NOT: return
// DISALBE-LABEL: func.func nested @test2()
// DISALBE-NOT: return

func.func @test3() attributes {testAttr}  {
    return
}
// DEFAULT-LABEL: func.func private @test3()
// DEFAULT-NOT: return
// DISALBE-LABEL: func.func @test3()
// DISALBE-NEXT: return
