import torch
from torch import nn
import torch._decomp
from functorch.compile import aot_module, aot_function
from torch._decomp import get_decompositions
import copy
import torchvision.models as models

from torch_frontend import convert_to_mhlo_via_torch_mlir
from torch_frontend import list_decomposed_ops, preprocess_fx_graph

# Wrap the ResNet model to avoid multiple returns problem
class ResNetWrapper(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.resnet = models.resnet18()
    
    def forward(self, data):
        return self.resnet(data)[0]

model = ResNetWrapper()
data = torch.randn(2,3,200,200)

def forward_compile(graph: torch.fx.GraphModule, inputs):
    print("============")
    print("Forward Part")
    print("============\n\n")
    origin_graph = copy.deepcopy(graph)
    graph = preprocess_fx_graph(graph)
    # print(graph.code)
    compiled_graph = convert_to_mhlo_via_torch_mlir(graph, inputs, use_tracing=False)
    # print(compiled_graph)
    with open("resnet_forward.mhlo.mlir", "w+") as fout:
        compiled_graph.operation.print(file=fout,
                                       large_elements_limit=None,
                                       print_generic_op_form=True)
    return origin_graph

def backward_compile(graph: torch.fx.GraphModule, inputs):
    print("============\n\n")
    print("Backward Part")
    print("============\n\n")
    origin_graph = copy.deepcopy(graph)
    graph = preprocess_fx_graph(graph)
    # print(graph.code)
    compiled_graph = convert_to_mhlo_via_torch_mlir(graph, inputs, use_tracing=False)
    # print(compiled_graph)
    with open("resnet_backward.mhlo.mlir", "w+") as fout:
        compiled_graph.operation.print(file=fout,
                                       large_elements_limit=None,
                                       print_generic_op_form=True)
    return origin_graph


# This simulates what inductor does (running the fw + bw decompositions)
decompositions = get_decompositions(list_decomposed_ops())
aot_print_fn = aot_module(model, fw_compiler=forward_compile,
			  bw_compiler=backward_compile, 
			  decompositions=decompositions)
# cloned_data = data.clone().detach().requires_grad_(True)
cloned_data = data.clone().detach()
res = aot_print_fn(cloned_data)
res.sum().backward()
