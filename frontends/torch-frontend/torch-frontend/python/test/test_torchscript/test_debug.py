import torch
import torch as tu

import torch_frontend
from torch_frontend import compile, DebugType

# ==============================================================================

class AtenCudaModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.cuda() + x.cuda()

def test_aten_cuda():
    module = AtenCudaModule()
    inputs = [tu.randn(3, 4)]
    module = torch.jit.script(module)
    module = compile(module, inputs, "stablehlo", debug=DebugType.PRINT_AFTER_ONLY_CHANGE)
    print(module.operation.get_asm(large_elements_limit=10, enable_debug_info=False))

