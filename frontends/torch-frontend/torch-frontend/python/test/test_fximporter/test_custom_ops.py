import torch
import torch_frontend
from torch_frontend import compile_dynamo_model, compile, GENERIC_CUSTOM_OPS

# ==============================================================================

@torch.library.custom_op("triton::add", mutates_args=())
def triton_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a + b

@torch.library.register_fake("triton::add")
def triton_add_fake_impl(a, b):
    return a + b

class TritonAddMod(torch.nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return triton_add(x, y)

def test_triton_custom_op():
    example_inputs = (torch.randn(10, 10), torch.randn(10, 10))
    prog = torch.export.export(TritonAddMod(), args=example_inputs)
    print(prog)
    module = compile_dynamo_model(prog, "stablehlo")
    # module = compile(torch.jit.trace(TritonAddMod(), example_inputs), example_inputs, "stablehlo", verbose=True, debug=torch_frontend.DebugType(1))
    assert "stablehlo.custom_call @triton.add" in module.operation.get_asm()

# ==============================================================================

@torch.library.custom_op("custom::add", mutates_args=())
def custom_add(a: torch.Tensor, c: int, d: str, b: torch.Tensor) -> torch.Tensor:
    return a + b + c

@torch.library.register_fake("custom::add")
def custom_add_fake_impl(a, c, d, b):
    return a + b

class CustomAddMod(torch.nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return custom_add(x, 2, "add", y)

def test_custom_op():
    example_inputs = (torch.randn(10, 10), torch.randn(10, 10))
    prog = torch.export.export(CustomAddMod(), args=example_inputs)
    print(prog)
    module = compile_dynamo_model(prog, "stablehlo", backend_legal_ops=GENERIC_CUSTOM_OPS+["custom.add"], verbose=True)
    print(module.operation.get_asm())
    assert "stablehlo.custom_call @custom.add" in module.operation.get_asm()
    assert 'byteir_attrs = {custom_attrs = [2, "add"]}' in module.operation.get_asm()

if __name__ == "__main__":
    test_custom_op()
