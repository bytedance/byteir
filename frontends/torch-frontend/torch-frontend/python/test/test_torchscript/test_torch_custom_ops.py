import torch
import torch as tu

import torch_frontend
from torch_frontend import compile, replace_flash_attn 
from functorch.compile import aot_module
import functools
from torch.testing import FileCheck

# ==============================================================================

torch.ops.load_library("build/lib/libcustom_op.so")
class DynamicPartitionStitchModule(torch.nn.Module):
    def forward(self, data, partitions, index0, index1):
        dynamic_partition = torch.ops.byteir.dynamic_partition(data, partitions, 2)
        dynamic_stitch = torch.ops.byteir.dynamic_stitch(
            [index0, index1], [dynamic_partition[0], dynamic_partition[1]])
        return dynamic_stitch


def test_dynamic_partition_stitch():
    module = DynamicPartitionStitchModule()
    data = torch.rand((5, 2))
    partitions = torch.tensor([0, 1, 0, 1, 0])
    indices = [torch.tensor([2, 3, 4]), torch.tensor([0, 1])]
    inputs = [data, partitions, indices[0], indices[1]]
    module = torch.jit.script(module)
    module = compile(module, inputs, "stablehlo")
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
        dynamic_partition = torch.ops.byteir.dynamic_partition(data, partitions, 2)
        dynamic_stitch = torch.ops.byteir.dynamic_mask_stitch(
            [dynamic_partition[0], dynamic_partition[1]], partitions)
        return dynamic_stitch


def test_dynamic_partition_mask_stitch():
    module = DynamicPartitionMaskStitchModule()
    data = torch.rand((5, 2))
    partitions = torch.tensor([0, 1, 0, 1, 0])
    inputs = [data, partitions]
    module = torch.jit.script(module)
    module = compile(module, inputs, "stablehlo")
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


# ==============================================================================


class FlashAttnModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v):
        return torch.ops.aten.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True, scale=0.2)


def flash_attn_compile_fx_inner(graph: torch.fx.GraphModule, inputs, is_backward):
    all_formatted = "\n".join([n.format_node() for n in graph.graph.nodes])
    if not is_backward:
        FileCheck().check("call_function").check(
            "torch.ops.byteir.flash_attn_fwd.default").run(all_formatted)
    else:
        FileCheck().check("call_function").check(
            "torch.ops.byteir.flash_attn_bwd.default").run(all_formatted)
    fx_graph = torch_frontend.preprocess_fx_graph(graph)
    backend_legal_ops=torch_frontend.BYTEIR_CUSTOM_OPS + torch_frontend.GENERIC_CUSTOM_OPS
    compiled_graph = torch_frontend.compile(fx_graph, inputs, 'torch', backend_legal_ops=backend_legal_ops)
    if not is_backward:
        FileCheck().check("torch.aten.transpose").check("torch.aten.transpose").check("torch.aten.transpose").check("torch.operator \"byteir.flash_attn_fwd\"").check(
            "(!torch.vtensor<[2,256,12,128],f16>, !torch.vtensor<[2,256,12,128],f16>, !torch.vtensor<[2,256,12,128],f16>, !torch.float, !torch.float, !torch.bool, !torch.bool) -> (!torch.vtensor<[2,256,12,128],f16>, !torch.vtensor<[2,256,12,128],f16>, !torch.vtensor<[2,256,12,128],f16>, !torch.vtensor<[2,256,12,128],f16>, !torch.vtensor<[2,256,12,128],f16>, !torch.vtensor<[2,12,256],f32>, !torch.vtensor<[2,12,256,256],f16>, !torch.vtensor<[2],si64>)") \
            .run(str(compiled_graph))
    else:
        FileCheck().check("torch.operator \"byteir.flash_attn_bwd\"") \
            .check("(!torch.vtensor<[2,256,12,128],f16>, !torch.vtensor<[2,256,12,128],f16>, !torch.vtensor<[2,256,12,128],f16>, !torch.vtensor<[2,256,12,128],f16>, !torch.vtensor<[2,256,12,128],f16>, !torch.vtensor<[2,12,256],f32>, !torch.float, !torch.float, !torch.bool, !torch.vtensor<[2],si64>) -> (!torch.vtensor<[2,256,12,128],f16>, !torch.vtensor<[2,256,12,128],f16>, !torch.vtensor<[2,256,12,128],f16>, !torch.vtensor<[2,12,256],f32>, !torch.vtensor<[2,12,256,128],f32>)") \
            .check("torch.aten.transpose").check("torch.aten.transpose").check("torch.aten.transpose").run(str(compiled_graph))
    return graph


def flash_attn_compile_fx(model: torch.fx.GraphModule, inputs):
    model = replace_flash_attn(model)
    module = aot_module(model, fw_compiler=functools.partial(flash_attn_compile_fx_inner, is_backward=False),
                        bw_compiler=functools.partial(flash_attn_compile_fx_inner, is_backward=True))
    return module


def test_flash_attn():
    model = FlashAttnModel().cuda()
    q = torch.rand(2, 12, 256, 128, requires_grad=True, dtype=torch.half).cuda()
    k = torch.rand(2, 12, 256, 128, requires_grad=True, dtype=torch.half).cuda()
    v = torch.rand(2, 12, 256, 128, requires_grad=True, dtype=torch.half).cuda()
    optimized_model = torch.compile(model, backend=flash_attn_compile_fx)
    output = optimized_model(q, k, v)
    output.sum().backward()
