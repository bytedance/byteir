// RUN: byteir-opt -split-input-file -verify-diagnostics %s

func.func @mesh_size_no_cluster() -> () attributes { mesh_cluster = @mesh0 } {
  // expected-error@+1 {{no cluster op}}
    %0 = mesh.size { axis = 1 : index} : index
    return
}

// -----

mesh.cluster @mesh0(rank = 3, dim_sizes = [4, 8, 12])

func.func @mesh_size_invalid_axis() -> () attributes { mesh_cluster = @mesh0 } {
  // expected-error@+1 {{axis is expected to within the range of cluster rank}}
    %0 = mesh.size { axis = 3 : index} : index
    return
}
