import torch
import torch_frontend
from torch_frontend import compile_dynamo_model, compile

# ==============================================================================

@torch.library.custom_op("triton::add", mutates_args=())
def custom_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a + b

@torch.library.register_fake("triton::add")
def _(a, b):
    return a + b

class AddMod(torch.nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return custom_add(x, y)

def test_triton_custom_op():
    example_inputs = (torch.randn(10, 10), torch.randn(10, 10))
    prog = torch.export.export(AddMod(), args=example_inputs)
    print(prog)
    module = compile_dynamo_model(prog, "stablehlo", verbose=True, debug=torch_frontend.DebugType(1))
    # module = compile(torch.jit.trace(AddMod(), example_inputs), example_inputs, "stablehlo", verbose=True, debug=torch_frontend.DebugType(1))
    print(module.operation.get_asm())

# ==============================================================================

if __name__ == "__main__":
    test_triton_custom_op()
