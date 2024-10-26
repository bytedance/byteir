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

def test_debug():
    module = AtenCudaModule()
    inputs = [tu.randn(3, 4)]
    module = torch.jit.script(module)
    module = compile(module, inputs, "stablehlo", debug=DebugType.PRINT_AFTER_ONLY_CHANGE)
    print(module.operation.get_asm(large_elements_limit=10, enable_debug_info=False))

# ==============================================================================

class AtenAddModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x + x

def test_rewrite_entry_func_name():
    module = AtenAddModule()
    inputs = [tu.randn(3, 4)]
    module = compile(module, inputs, "stablehlo", entry_func_name="main")
    module_str = module.operation.get_asm(large_elements_limit=10, enable_debug_info=False)
    print(module_str)
    assert "func.func @main" in module_str

if __name__ == "__main__":
    test_debug()
