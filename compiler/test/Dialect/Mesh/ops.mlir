// RUN: byteir-opt %s | FileCheck %s

mesh.cluster @mesh0(rank = 3, dim_sizes = [4, 8, 12])
// CHECK: mesh.cluster @mesh0

func.func @distributed_tensor(%arg0 : tensor<2x5xf32, #mesh.shard<[[1]]>>, %arg1 : tensor<2x5xf32, #mesh.shard<[[1]]>>) 
                -> tensor<2x5xf32, #mesh.shard<[[1]]>> attributes { mesh_cluster = @mesh0 } {
    %2 = mhlo.add %arg0, %arg1 : tensor<2x5xf32, #mesh.shard<[[1]]>>
    return %2 : tensor<2x5xf32, #mesh.shard<[[1]]>>
}
// CHECK: func.func @distributed_tensor

func.func @annotate_simple(%arg0 : tensor<2x5xf32>, %arg1 : tensor<2x5xf32>) -> tensor<2x5xf32> attributes { mesh_cluster = @mesh0 } {
    %2 = mhlo.add %arg0, %arg1 : tensor<2x5xf32>
    %3 = mesh.annotate %2 {sharding = [[1]], required = true} : tensor<2x5xf32> -> tensor<2x5xf32>
    return %3 : tensor<2x5xf32>
}
// CHECK: func.func @annotate_simple

func.func @mesh_size_simple() -> () attributes { mesh_cluster = @mesh0 } {
    %0 = mesh.size { axis = 1 : index} : index
    return
}
// CHECK: func.func @mesh_size_simple
