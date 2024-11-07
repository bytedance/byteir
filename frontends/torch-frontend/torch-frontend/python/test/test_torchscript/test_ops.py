import torch
import torch as tu

import torch_frontend
from torch_frontend import compile

import pytest

def numerical_test_helper(module, inputs, torch_output, atol=1e-5, rtol=1e-5):
    if isinstance(torch_output, torch.Tensor):
        torch_output = [torch_output]
    np_inputs = [t.numpy() for t in inputs]

    from mhlo_tools.ir_executor import Interpreter
    func_name = module.body.operations[0].name.value
    interp = Interpreter.load_from_string(module.operation.get_asm(), is_stablehlo=True)
    mhlo_outputs = interp.call_function(func_name, np_inputs)
    mhlo_outputs = [torch.tensor(t) for t in mhlo_outputs]

    assert len(torch_output) == len(mhlo_outputs)
    for t, m in zip(torch_output, mhlo_outputs):
        torch.testing.assert_close(m, t, rtol=rtol, atol=atol, equal_nan=True)


# ==============================================================================
# rng testcases

class AtenUniformModule(torch.nn.Module):
    def forward(self, x):
        return x.uniform_(1.0, 10.0)

def test_aten_uniform():
    module = compile(AtenUniformModule(), [tu.zeros(3, 4)], "stablehlo")
    mlir_str = module.operation.get_asm()
    assert "stablehlo.rng" in mlir_str
    assert "UNIFORM" in mlir_str

class AtenRandnModule(torch.nn.Module):
    def forward(self, x):
        return torch.randn(size=x.shape)

def test_aten_randn():
    module = compile(AtenRandnModule(), [tu.zeros(3, 4)], "stablehlo")
    mlir_str = module.operation.get_asm()
    assert "stablehlo.rng" in mlir_str
    assert "NORMAL" in mlir_str

class AtenNormalFunctionalModule(torch.nn.Module):
    def forward(self, x):
        return torch.ops.aten.normal_functional(x, mean=-5.0, std=2.0)

def test_aten_normal_functional():
    module = compile(AtenNormalFunctionalModule(), [tu.zeros(3, 4)], "stablehlo")
    mlir_str = module.operation.get_asm()
    assert "stablehlo.rng" in mlir_str
    assert "NORMAL" in mlir_str

# ==============================================================================
# uint8

class AtenBitwiseAndUint8Module(torch.nn.Module):
    def forward(self, x, y):
        return torch.bitwise_and(x, y)

def test_uint8():
    input_shape = [1024, 2048]
    x = torch.randint(0, 256, input_shape, dtype=torch.uint8)
    y = torch.randint(0, 256, input_shape, dtype=torch.uint8)
    inputs = [x, y]
    module = compile(AtenBitwiseAndUint8Module(), inputs, "stablehlo")
    mlir_str = module.operation.get_asm()
    assert "stablehlo.and" in mlir_str
    assert "tensor<1024x2048xui8>" in mlir_str

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
    module = compile(module, inputs, "stablehlo")
    print(module.operation.get_asm(large_elements_limit=10, enable_debug_info=False))

# ==============================================================================

class AtenBatchNormFp16Module(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.ops.aten.batch_norm(x, y, y, y, y, training=False, momentum=0.1, eps=0.01, cudnn_enabled=False)

def test_aten_batch_norm_fp16():
    module = AtenBatchNormFp16Module()
    inputs = [tu.randn(1, 16, 28, 28).half().cuda(), tu.randn(16).cuda()]
    module = torch.jit.script(module)
    module = compile(module, inputs, "stablehlo")
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
    inputs = [tu.randint(0, 2, (10, 20)).long()]
    module = torch.jit.trace(module, inputs) # x.bool() can't be scripted
    module = compile(module , inputs, "stablehlo")
    print(module.operation.get_asm(large_elements_limit=10, enable_debug_info=False))

# ==============================================================================

class LinalgVectorNormModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return torch.ops.aten.linalg_vector_norm(x, ord=3.0, dim=[0, 1], keepdim=False)

def test_linalg_vector_norm():
    module = LinalgVectorNormModule()
    inputs = [tu.randn(3, 4, 5).to(torch.float16)]
    module = compile(module, inputs, "stablehlo")
    print(module.operation.get_asm(large_elements_limit=10, enable_debug_info=False))

# ==============================================================================

class ScaledDotProductAttentionDifferentModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value):
        return torch.ops.aten.scaled_dot_product_attention(query, key, value)

def test_attention():
    model = ScaledDotProductAttentionDifferentModule()
    query = torch.randn(2, 3, 8, 4, dtype=torch.float32)
    key = torch.randn(2, 3, 16, 4, dtype=torch.float32)
    value = torch.randn(2, 3, 16, 4, dtype=torch.float32)
    inputs = [query, key, value]
    module = compile(model, inputs, "stablehlo")
    numerical_test_helper(module, inputs, model(*inputs))

# ==============================================================================

class NllLossStaticModule(torch.nn.Module):
    # Here the 2nd index is ignored.
    def forward(self, x, y):
        return torch.ops.aten.nll_loss_forward(
            x, target=y, weight=None, reduction=0, ignore_index=2
        )

def test_nll_loss_forward():
    inputs = [tu.rand(2, 3), tu.randint(low=0, high=3, size=(2,))]
    module = compile(NllLossStaticModule(), inputs, "stablehlo")
    numerical_test_helper(module, inputs, model(*inputs))

# ==============================================================================

class VarDimModule(torch.nn.Module):
    def forward(self, x):
        return torch.var(x, dim=1)

@pytest.mark.mhlo_tools
def test_var_dim_f32():
    inputs = [tu.randn(4, 4)]
    module = compile(VarDimModule(), inputs, "stablehlo", verbose=True)
    module_str = module.operation.get_asm()
    assert "f64" not in module_str
    numerical_test_helper(module, inputs, VarDimModule()(*inputs))

# ==============================================================================

class MaxDimModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return torch.max(x, dim=1)[0]

def test_max_dim():
    inputs = [tu.randn(3, 4)]
    module = compile(MaxDimModule(), inputs, "stablehlo")
    mlir_str = module.operation.get_asm()
    assert "stablehlo.reduce" in mlir_str
    module_ = compile(MaxDimModule(), inputs, "stablehlo", backend_legal_ops=[])
    mlir_str_ = module_.operation.get_asm()
    assert "stablehlo.reduce" in mlir_str_

class MinDimModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return torch.min(x, dim=1)[0]

def test_min_dim():
    inputs = [tu.randn(3, 4)]
    module = compile(MinDimModule(), inputs, "stablehlo")
    mlir_str = module.operation.get_asm()
    assert "stablehlo.reduce" in mlir_str
    module_ = compile(MinDimModule(), inputs, "stablehlo", backend_legal_ops=[])
    mlir_str_ = module_.operation.get_asm()
    assert "stablehlo.reduce" in mlir_str_

class MaxDimKeepDimModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return torch.max(x, dim=1, keepdim=True)[0]

def test_max_dim_keepdim():
    inputs = [tu.randn(3, 4)]
    module = compile(MaxDimKeepDimModule(), inputs, "stablehlo")
    mlir_str = module.operation.get_asm()
    assert "stablehlo.reduce" in mlir_str
    module_ = compile(MaxDimKeepDimModule(), inputs, "stablehlo", backend_legal_ops=[])
    mlir_str_ = module_.operation.get_asm()
    assert "stablehlo.reduce" in mlir_str_

class AmaxModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return torch.amax(x, dim=1)

def test_amax():
    inputs = [tu.randn(3, 4)]
    module = compile(AmaxModule(), inputs, "stablehlo", backend_legal_ops=[])
    mlir_str = module.operation.get_asm()
    assert "stablehlo.reduce" in mlir_str

class AmaxDimListModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return torch.amax(x, dim=(0, 1))

def test_amax_dim_list():
    inputs = [tu.randn(3, 4, 5)]
    module = compile(AmaxDimListModule(), inputs, "stablehlo", backend_legal_ops=[])
    mlir_str = module.operation.get_asm()
    assert "stablehlo.reduce" in mlir_str
    assert "across dimensions = [0, 1]" in mlir_str

# ==============================================================================

class ListModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return [x, x, x]

def test_return_list():
    inputs = [tu.randn(3, 4)]
    module = compile(ListModule(), inputs, "stablehlo")
    print(module.operation.get_asm())

# ==============================================================================
# derefine cases

class ArangeModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        end = torch.ops.prim.NumToTensor(x.shape[1])
        return torch.arange(0, end, 1)

def test_arange_derefine():
    inputs = [tu.randn(3, 4)]
    module = compile(ArangeModule(), inputs, "stablehlo")
    print(module.operation.get_asm())

class ClampDerefineModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        shape = torch.ops.prim.NumToTensor(x.shape[1])
        return torch.ops.aten.clamp(x, 0, shape)

def test_clamp_derefine():
    inputs = [tu.randn(3, 4)]
    module = compile(ClampDerefineModule(), inputs, "stablehlo")
    print(module.operation.get_asm())

# ==============================================================================
# tuple cases

class Tuple1Module(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return (x, )

def test_tuple_one_tensor():
    inputs = [tu.randn(3, 4)]
    module = compile(Tuple1Module(), inputs, "stablehlo")
    print(module.operation.get_asm())

class Tuple2Module(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return (x, x)

def test_tuple_one_tensor():
    inputs = [tu.randn(3, 4)]
    module = compile(Tuple2Module(), inputs, "stablehlo")
    print(module.operation.get_asm())

# ==============================================================================
# expand_as

class ExpandAsModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y, z):
        return x.expand_as(y), x.expand_as(z)

class ExpandAsModule1(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y):
        return x.expand_as(y)

def test_expand_as():
    inputs = [tu.randn(3, 1), tu.randn(3, 4), tu.randn(2, 2, 3, 4)]
    module = compile(ExpandAsModule(), inputs, "stablehlo")
    numerical_test_helper(module, inputs, ExpandAsModule()(*inputs))

    inputs = [tu.tensor(1.0), tu.randn(3, 4).to(torch.int64)]
    module = compile(ExpandAsModule1(), inputs, "stablehlo")
    numerical_test_helper(module, inputs, ExpandAsModule1()(*inputs))
