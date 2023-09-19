// RUN: byteir-opt %s -loop-tag="anchor-attr=testFuncAttr attach-attr=testAttr" | FileCheck %s -check-prefix=DEFAULT
// RUN: byteir-opt %s -loop-tag="anchor-attr=testFuncAttr attach-attr=testAttr depth=2" | FileCheck %s -check-prefix=DEPTH2
// RUN: byteir-opt %s -loop-tag="anchor-attr=testFuncAttr attach-attr=testAttr depth=-1" | FileCheck %s -check-prefix=DEPTHLAST

func.func @loops() attributes {testFuncAttr} {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c128 = arith.constant 128 : index
  scf.for %arg0 = %c0 to %c128 step %c8 {
    scf.for %arg1 = %c0 to %c128 step %c8 {
    }
  }

  scf.for %arg0 = %c0 to %c128 step %c8 {
    scf.for %arg1 = %c0 to %c128 step %c8 {
      scf.for %arg2 = %c0 to %c128 step %c8 {
      }
    }
  }

  return
}

// DEFAULT-LABEL: func.func @loops
// DEFAULT:       scf.for
// DEFAULT-NEXT:    scf.for
// DEFAULT-NEXT:    } 
// DEFAULT-NEXT:  } {testAttr}
// DEFAULT:       scf.for
// DEFAULT-NEXT:    scf.for
// DEFAULT-NEXT:      scf.for
// DEFAULT-NEXT:      } 
// DEFAULT-NEXT:    } 
// DEFAULT-NEXT:  } {testAttr}

// DEPTH2-LABEL: func.func @loops
// DEPTH2:       scf.for
// DEPTH2-NEXT:    scf.for
// DEPTH2-NEXT:    } {testAttr}
// DEPTH2-NEXT:  }
// DEPTH2:       scf.for
// DEPTH2-NEXT:    scf.for
// DEPTH2-NEXT:      scf.for
// DEPTH2-NEXT:      } 
// DEPTH2-NEXT:    } {testAttr}
// DEPTH2-NEXT:  }

// DEPTHLAST-LABEL: func.func @loops
// DEPTHLAST:       scf.for
// DEPTHLAST-NEXT:    scf.for
// DEPTHLAST-NEXT:    } {testAttr}
// DEPTHLAST-NEXT:  }
// DEPTHLAST:       scf.for
// DEPTHLAST-NEXT:    scf.for
// DEPTHLAST-NEXT:      scf.for
// DEPTHLAST-NEXT:      } {testAttr}
// DEPTHLAST-NEXT:    }
// DEPTHLAST-NEXT:  }
