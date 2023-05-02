import torch
import torch as tu

import torch_frontend
from torch_frontend import convert_to_mhlo_via_torch_mlir

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

class MaxDimModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return torch.max(x, dim=1)[0]

def test_max_dim():
    inputs = [tu.randn(3, 4)]
    module = convert_to_mhlo_via_torch_mlir(MaxDimModule(), inputs)
    print(module.operation.get_asm())

class MaxDimKeepDimModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return torch.max(x, dim=1, keepdim=True)[0]

def test_max_dim_keepdim():
    inputs = [tu.randn(3, 4)]
    module = convert_to_mhlo_via_torch_mlir(MaxDimKeepDimModule(), inputs)
    print(module.operation.get_asm())

# ==============================================================================

torch.ops.load_library("build/lib/libcustom_op.so")
class DynamicPartitionStitchModule(torch.nn.Module):
    def __init__(self, *, output_shape):
        super().__init__()
        self.output_shape = output_shape
    
    def forward(self, data, partitions, index0, index1):
        dynamic_partition = torch.ops.custom.dynamic_partition(data, partitions, 2)
        dynamic_stitch = torch.ops.custom.dynamic_stitch(
            [index0, index1], [dynamic_partition[0], dynamic_partition[1]], self.output_shape)
        return dynamic_stitch


def test_dynamic_partition_stitch():
    module = DynamicPartitionStitchModule(output_shape=(5, 2))
    data = torch.rand((5, 2))
    partitions = torch.tensor([0, 1, 0, 1, 0])
    indices = [torch.tensor([2, 3, 4]), torch.tensor([0, 1])]
    inputs = [data, partitions, indices[0], indices[1]]
    module = torch.jit.trace(module, inputs)
    module = convert_to_mhlo_via_torch_mlir(module, inputs)
    print(module.operation.get_asm(large_elements_limit=10, enable_debug_info=False))


def test_dynamic_partition_stitch_gpu():
    module = DynamicPartitionStitchModule(output_shape=(5, 2, 2))
    device = torch.device("cuda")
    data = torch.rand((5, 2, 2)).to(device)
    partitions = torch.tensor([0, 1, 0, 1, 0]).to(device)
    indices = [torch.tensor([2, 3, 4]).to(device), torch.tensor([0, 1]).to(device)]
    inputs = [data, partitions, indices[0], indices[1]]
    output = module(*inputs)
    assert output.device.type == "cuda" and tuple(output.size()) == (5, 2, 2)


class DynamicPartitionMaskStitchModule(torch.nn.Module):
    def __init__(self, *, output_shape):
        super().__init__()
        self.output_shape = output_shape

    def forward(self, data, partitions):
        dynamic_partition = torch.ops.custom.dynamic_partition(data, partitions, 2)
        dynamic_stitch = torch.ops.custom.dynamic_mask_stitch(
            [dynamic_partition[0], dynamic_partition[1]], partitions, self.output_shape)
        return dynamic_stitch


def test_dynamic_partition_mask_stitch():
    module = DynamicPartitionMaskStitchModule(output_shape=(5, 2))
    data = torch.rand((5, 2))
    partitions = torch.tensor([0, 1, 0, 1, 0])
    inputs = [data, partitions]
    module = torch.jit.trace(module, inputs)
    module = convert_to_mhlo_via_torch_mlir(module, inputs)
    print(module.operation.get_asm(large_elements_limit=10, enable_debug_info=False))


def test_dynamic_partition_mask_stitch_gpu():
    module = DynamicPartitionMaskStitchModule(output_shape=(5, 2, 2))
    device = torch.device("cuda")
    data = torch.rand((5, 2, 2)).to(device)
    partitions = torch.tensor([0, 1, 0, 1, 0]).to(device)
    inputs = [data, partitions]
    output = module(*inputs)
    assert output.device.type == "cuda" and tuple(output.size()) == (5, 2, 2)
    assert torch.all(torch.eq(data, output))
