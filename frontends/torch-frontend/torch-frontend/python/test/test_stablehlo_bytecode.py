import torch
import torch as tu

import torch_frontend
from torch_frontend import compile
from torch_frontend._mlir_libs._stablehlo import deserialize_portable_artifact

def serialize_helper(module, inputs):
    stablehlo_bytecode = compile(module, inputs, "stablehlo+0.16.2")
    deserialize_str = deserialize_portable_artifact(stablehlo_bytecode)
    print(deserialize_str)

# ==============================================================================
class SoftmaxModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.ops.aten._softmax(x, dim=1, half_to_float=False)

def test_softmax():
    inputs = [tu.rand(3, 4)]
    serialize_helper(SoftmaxModule(), inputs)
