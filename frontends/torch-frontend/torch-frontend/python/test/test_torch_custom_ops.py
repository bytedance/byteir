import torch
import torch as tu

import torch_frontend
from torch_frontend import convert_to_mhlo_via_torch_mlir

# ==============================================================================

torch.ops.load_library("build/lib/libcustom_op.so")
class DynamicPartitionStitchModule(torch.nn.Module):
    def forward(self, data, partitions, index0, index1):
        dynamic_partition = torch.ops.custom.dynamic_partition(data, partitions, 2)
        dynamic_stitch = torch.ops.custom.dynamic_stitch(
            [index0, index1], [dynamic_partition[0], dynamic_partition[1]])
        return dynamic_stitch


def test_dynamic_partition_stitch():
    module = DynamicPartitionStitchModule()
    data = torch.rand((5, 2))
    partitions = torch.tensor([0, 1, 0, 1, 0])
    indices = [torch.tensor([2, 3, 4]), torch.tensor([0, 1])]
    inputs = [data, partitions, indices[0], indices[1]]
    module = torch.jit.script(module)
    module = convert_to_mhlo_via_torch_mlir(module, inputs)
    print(module.operation.get_asm(large_elements_limit=10, enable_debug_info=False))


def test_dynamic_partition_stitch_gpu():
    module = DynamicPartitionStitchModule()
    device = torch.device("cuda")
    data = torch.rand((5, 2, 2)).to(device)
    partitions = torch.tensor([0, 1, 0, 1, 0]).to(device)
    indices = [torch.tensor([2, 3, 4]).to(device), torch.tensor([0, 1]).to(device)]
    inputs = [data, partitions, indices[0], indices[1]]
    output = module(*inputs)
    assert output.device.type == "cuda" and tuple(output.size()) == (5, 2, 2)


class DynamicPartitionMaskStitchModule(torch.nn.Module):
    def forward(self, data, partitions):
        dynamic_partition = torch.ops.custom.dynamic_partition(data, partitions, 2)
        dynamic_stitch = torch.ops.custom.dynamic_mask_stitch(
            [dynamic_partition[0], dynamic_partition[1]], partitions)
        return dynamic_stitch


def test_dynamic_partition_mask_stitch():
    module = DynamicPartitionMaskStitchModule()
    data = torch.rand((5, 2))
    partitions = torch.tensor([0, 1, 0, 1, 0])
    inputs = [data, partitions]
    module = torch.jit.script(module)
    module = convert_to_mhlo_via_torch_mlir(module, inputs)
    print(module.operation.get_asm(large_elements_limit=10, enable_debug_info=False))


def test_dynamic_partition_mask_stitch_gpu():
    module = DynamicPartitionMaskStitchModule()
    device = torch.device("cuda")
    data = torch.rand((5, 2, 2)).to(device)
    partitions = torch.tensor([0, 1, 0, 1, 0]).to(device)
    inputs = [data, partitions]
    output = module(*inputs)
    assert output.device.type == "cuda" and tuple(output.size()) == (5, 2, 2)
    assert torch.all(torch.eq(data, output))
