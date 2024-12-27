import torch
import torch_frontend
from torch_frontend import compile, MATH_CUSTOM_OPS

def custom_test_helper(model, inputs, custom_op_name):
    mlir_module = compile(model, inputs, "stablehlo", backend_legal_ops=MATH_CUSTOM_OPS)
    mlir_str = mlir_module.operation.get_asm()
    print(mlir_str)
    compare_str = "stablehlo.custom_call @{}".format(custom_op_name)
    assert compare_str in mlir_str

# ==============================================================================

class TruncModule(torch.nn.Module):
    def forward(self, x):
        return torch.trunc(x)

def test_trunc():
    custom_test_helper(TruncModule(), [torch.rand(3, 4)], "math.trunc")

# ==============================================================================

class Exp2Module(torch.nn.Module):
    def forward(self, x):
        return torch.exp2(x)

def test_exp2():
    custom_test_helper(Exp2Module(), [torch.rand(3, 4)], "math.exp2")

# ==============================================================================

class CopysignModule(torch.nn.Module):
    def forward(self, x, y):
        return torch.copysign(x, y)

def test_copysign():
    custom_test_helper(CopysignModule(), [torch.rand(3, 4), torch.rand(3, 4)], "math.copysign")

# ==============================================================================

class LdexpModule(torch.nn.Module):
    def forward(self, x, y):
        return torch.ldexp(x, y)

def test_ldexp():
    custom_test_helper(LdexpModule(), [torch.rand(3, 4), torch.rand(3, 4)], "math.ldexp")

# ==============================================================================

class SignbitModule(torch.nn.Module):
    def forward(self, x):
        return torch.signbit(x)

def test_signbit():
    custom_test_helper(SignbitModule(), [torch.rand(3, 4)], "math.signbit")
