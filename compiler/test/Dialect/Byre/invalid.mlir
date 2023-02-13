// RUN: byteir-opt -allow-unregistered-dialect %s -split-input-file -verify-diagnostics

module attributes {byre.container_module} {
  func.func @invalid_entry_func(%arg0 : memref<100x?xf32>) {
    // expected-error @+1 {{expected 'byre.entry_point' attribute to be attached to 'func.func'}}
    byre.compute @some_kernel(%arg0) : memref<100x?xf32>
    return
  }
}

// -----

module {
// expected-error @+1 {{expected 'byre.entry_point' attribute to be attached to 'func.func' under 'builtin.module' with 'byre.container_module'}}
  func.func @invalid_entry_module(%arg0 : memref<100x?xf32>) attributes {byre.entry_point} {
    return
  }
}

// -----

module {
  // expected-error @+1 {{expected 'byre.container_module' attribute to be attached to 'builtin.module'}}
  func.func @invalid_entry_module(%arg0 : memref<100x?xf32>) attributes {byre.container_module} {
    return
  }
}

// -----

module attributes {byre.container_module} {
  // expected-error @+1 {{expected attribute 'byre.argtype'}}
  func.func @invalid_entry_func(%arg0 : memref<100x?xf32>) attributes {byre.entry_point} {
    return
  }
}

// -----

module attributes {byre.container_module} {
  // expected-error @+1 {{expected attribute 'byre.argname'}}
  func.func @invalid_entry_func(%arg0 : memref<100x?xf32> {byre.argtype = 2: i32}) attributes {byre.entry_point} {
    return
  }
}

// -----

module attributes {byre.container_module} {
  // expected-error @+1 {{expected at least 1 argument which was attached with 'byre.argtype' attribute contained 'Output'}}
  func.func @invalid_entry_func(%arg0 : memref<100x?xf32> {byre.argtype = 1: i32, byre.argname = "output"}) attributes {byre.entry_point} {
    return
  }
}

// -----

module attributes {byre.container_module} {
  // expected-error @+1 {{invalid argtype 'Input|Output'}}
  func.func @invalid_entry_func(%arg0 : memref<100x?xf32> {byre.argtype = 3: i32}) attributes {byre.entry_point} {
    return
  }
}

// -----

module attributes {byre.container_module} {
  // expected-error @+1 {{expected StringAttr in 'byre.argname'}}
  func.func @invalid_entry_func(%arg0 : memref<100x?xf32> {byre.argtype = 2: i32, byre.argname = 0: i32}) attributes {byre.entry_point} {
    return
  }
}
