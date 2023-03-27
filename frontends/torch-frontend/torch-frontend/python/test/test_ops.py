import torch
import torch as tu
import torch.fx
from functorch import make_fx

import torch_frontend
from torch_frontend import convert_to_mhlo_via_torch_mlir

# ==============================================================================

class NativeLayerNormModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, weight, bias):
        list = [2, 2, 3]
        return torch.ops.aten.native_layer_norm(
            x, list, weight, bias, eps=0.5)

def test_native_layer_norm():
    module = NativeLayerNormModule()
    inputs = [tu.rand(2, 5, 2, 2, 3).to(torch.float16), tu.rand(2, 2, 3).to(torch.float16), tu.rand(2, 2, 3).to(torch.float16)]
    module = convert_to_mhlo_via_torch_mlir(module, inputs)
    print(module.operation.get_asm(large_elements_limit=10, enable_debug_info=False))

# ==============================================================================

class OneHotModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.nn.functional.one_hot(x, num_classes=5)

def test_one_hot():
    module = OneHotModule()
    inputs = [tu.arange(0, 5).long()]
    module = convert_to_mhlo_via_torch_mlir(module, inputs)
    print(module.operation.get_asm(large_elements_limit=10, enable_debug_info=False))

# ==============================================================================

class TopKModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return torch.topk(x, 3, dim=1)

def test_topk():
    module = TopKModule()
    inputs = [tu.randn(3, 4)]
    module = convert_to_mhlo_via_torch_mlir(module, inputs)
    print(module.operation.get_asm(large_elements_limit=10, enable_debug_info=False))

# ==============================================================================

class CatModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        y = x.bool()
        return torch.cat([x, y], dim=0)

def test_cat():
    module = CatModule()
    inputs = [tu.randint(0, 2, (10, 20)).long()]
    module = torch.jit.trace(module, inputs) # x.bool() can't be scripted
    module = convert_to_mhlo_via_torch_mlir(module , inputs)
    print(module.operation.get_asm(large_elements_limit=10, enable_debug_info=False))

# ==============================================================================

class LinalgVectorNormModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return torch.ops.aten.linalg_vector_norm(x, ord=3.0, dim=[0, 1], keepdim=False)

def test_linalg_vector_norm():
    module = LinalgVectorNormModule()
    inputs = [tu.randn(3, 4, 5).to(torch.float16)]
    module = convert_to_mhlo_via_torch_mlir(module, inputs)
    print(module.operation.get_asm(large_elements_limit=10, enable_debug_info=False))

# ==============================================================================

torch.ops.load_library("build/lib/libcustom_op.so")
class DynamicPartitionStitchModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, data, partitions, index0, index1):
        dynamic_partition = torch.ops.custom.dynamic_partition(data, partitions, 2)
        dynamic_stitch = torch.ops.custom.dynamic_stitch([index0, index1], [dynamic_partition[0], dynamic_partition[1]], [5, 2]) 
        return dynamic_stitch


def test_dynamic_partition_stitch():
    module = DynamicPartitionStitchModule()
    data = torch.rand((5, 2))
    partitions = torch.tensor([0, 1, 0, 1, 0])
    indices = [torch.tensor([2, 3, 4]), torch.tensor([0, 1])]

    inputs = [data, partitions, indices[0], indices[1]]
    module = torch.jit.trace(module, inputs)
    module = convert_to_mhlo_via_torch_mlir(module, inputs)
    print(module.operation.get_asm(large_elements_limit=10, enable_debug_info=False))
