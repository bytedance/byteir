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

class CatModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        y = x.bool()
        return torch.cat([x, y], dim=0)

def test_cat():
  module = CatModule()
  # inputs = [tu.randint(0, 2, (10, 20)).float(), tu.randint(0, 2, (10, 20)).float()]
  # inputs = [tu.randint(0, 2, (10, 20)).bool(), tu.randint(0, 2, (10, 20)).long()]
  inputs = [tu.randint(0, 2, (10, 20)).long()]
  
  module = torch.jit.trace(module, inputs)
  # fx_g = make_fx(module)(*inputs)
  # print(fx_g.code)

  module = convert_to_mhlo_via_torch_mlir(module , inputs)
  print(module.operation.get_asm(large_elements_limit=10, enable_debug_info=False))
