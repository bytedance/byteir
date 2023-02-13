// RUN: byteir-translate -split-input-file -emit-cpp -verify-diagnostics %s

// expected-error@+1 {{'func.func' op with multiple blocks needs variables declared at top}}
func.func @multiple_blocks() {
^bb1:
    cf.br ^bb2
^bb2:
    return
}

// -----

// expected-error@+1 {{cannot emit integer type 'i80'}}
func.func @unsupported_integer_type(%arg0 : i80) {
  return
}

// -----

// expected-error@+1 {{cannot emit float type 'f80'}}
func.func @unsupported_float_type(%arg0 : f80) {
  return
}

// -----

// expected-error@+1 {{cannot emit type 'vector<100xf32>'}}
func.func @vector_type(%arg0 : vector<100xf32>) {
  return
}

// -----

// expected-error@+1 {{cannot emit tensor type with non static shape}}
func.func @non_static_shape(%arg0 : tensor<?xf32>) {
  return
}

// -----

// expected-error@+1 {{cannot emit unranked tensor type}}
func.func @unranked_tensor(%arg0 : tensor<*xf32>) {
  return
}
