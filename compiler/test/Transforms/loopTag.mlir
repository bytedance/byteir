// RUN: byteir-opt %s -loop-tag="anchor-attr=testAttr attach-attr=testAttr" | FileCheck %s -check-prefix=DEFAULT
// RUN: byteir-opt %s -loop-tag="anchor-attr=testAttr attach-attr=testAttr depth=2" | FileCheck %s -check-prefix=DEPTH2

func.func @loops() attributes {testAttr} {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c128 = arith.constant 128 : index
  scf.for %arg0 = %c0 to %c128 step %c8 {
    scf.for %arg1 = %c0 to %c128 step %c8 {
    }
  }

  scf.for %arg0 = %c0 to %c128 step %c8 {
    scf.for %arg1 = %c0 to %c128 step %c8 {
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
// DEFAULT-NEXT:    } 
// DEFAULT-NEXT:  } {testAttr}

// DEPTH2-LABEL: func.func @loops
// DEPTH2:       scf.for
// DEPTH2-NEXT:    scf.for
// DEPTH2-NEXT:    } {testAttr}
// DEPTH2-NEXT:  }
// DEPTH2:       scf.for
// DEPTH2-NEXT:    scf.for
// DEPTH2-NEXT:    } {testAttr}
// DEPTH2-NEXT:  }
