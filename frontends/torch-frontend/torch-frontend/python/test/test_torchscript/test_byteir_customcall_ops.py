import torch
import torch as tu

import torch_frontend
from torch_frontend import compile

def custom_test_helper(module, inputs, custom_op_name):
    mlir_module = compile(module, inputs, "stablehlo")
    mlir_str = mlir_module.operation.get_asm(large_elements_limit=10, enable_debug_info=False)
    compare_str = "stablehlo.custom_call @{}".format(custom_op_name)
    print(mlir_str)
    assert compare_str in mlir_str

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

class SoftmaxModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.ops.aten._softmax(x, dim=1, half_to_float=False)

def test_softmax():
    inputs = [tu.rand(3, 4)]
    custom_test_helper(SoftmaxModule(), inputs, "byteir.softmax")


class LogSoftmaxModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.ops.aten._log_softmax(x, dim=1, half_to_float=False)

def test_log_softmax():
    inputs = [tu.rand(3, 4)]
    custom_test_helper(LogSoftmaxModule(), inputs, "byteir.log_softmax")


# ==============================================================================

class NativeLayerNormModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, weight, bias):
        list = [2, 2, 3]
        return torch.ops.aten.native_layer_norm(
            x, list, weight, bias, eps=0.5)

def test_native_layer_norm():
    inputs = [tu.rand(2, 5, 2, 2, 3).to(torch.float16), tu.rand(2, 2, 3).to(torch.float16), tu.rand(2, 2, 3).to(torch.float16)]
    custom_test_helper(NativeLayerNormModule(), inputs, "byteir.layer_norm")


class LayerNormModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, weight, bias):
        list = [2, 2, 3]
        return torch.ops.aten.layer_norm(x, list, weight, bias, eps=0.5)

def test_layer_norm():
    inputs = [tu.rand(2, 5, 2, 2, 3).to(torch.float16), tu.rand(2, 2, 3).to(torch.float16), tu.rand(2, 2, 3).to(torch.float16)]
    custom_test_helper(LayerNormModule(), inputs, "byteir.layer_norm")


class LayerNormNoneBiasModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        list = [2, 2, 3]
        return torch.ops.aten.layer_norm(x, list, eps=0.5)

def test_layer_norm_none_bias():
    inputs = [tu.rand(2, 5, 2, 2, 3).to(torch.float16)]
    custom_test_helper(LayerNormNoneBiasModule(), inputs, "byteir.layer_norm")

class GroupNormModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, weight, bias):
        return torch.ops.aten.group_norm(x, 2, weight, bias, 0.5)

def test_group_norm():
    inputs = [tu.rand(2, 6, 14), tu.rand(6), tu.rand(6)]
    custom_test_helper(GroupNormModule(), inputs, "byteir.layer_norm")

class GroupNormNoneBiasModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return torch.ops.aten.group_norm(x, 2, weight=None, bias=None, eps=0.5, cudnn_enabled=False)

def test_group_norm_none_bias():
    inputs = [tu.rand(2, 6, 14)]
    custom_test_helper(GroupNormNoneBiasModule(), inputs, "byteir.layer_norm")

# ==============================================================================

class OneHotModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.nn.functional.one_hot(x, num_classes=5)

def test_one_hot():
    inputs = [tu.arange(0, 5).long()]
    custom_test_helper(OneHotModule(), inputs, "byteir.one_hot")

# ==============================================================================

class TopKModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return torch.topk(x, 3, dim=1)

def test_topk():
    inputs = [tu.randn(3, 4)]
    custom_test_helper(TopKModule(), inputs, "byteir.top_k")

# ==============================================================================

class MaxDimModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return torch.max(x, dim=1)

def test_max_dim():
    inputs = [tu.randn(3, 4)]
    custom_test_helper(MaxDimModule(), inputs, "byteir.arg_max")

class MinDimModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return torch.min(x, dim=1)

def test_min_dim():
    inputs = [tu.randn(3, 4)]
    custom_test_helper(MinDimModule(), inputs, "byteir.arg_min")

class MaxDimKeepDimModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return torch.max(x, dim=1, keepdim=True)

def test_max_dim_keepdim():
    inputs = [tu.randn(3, 4)]
    custom_test_helper(MaxDimKeepDimModule(), inputs, "byteir.arg_max")

class MaxDimOnlyIndicesModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return torch.max(x, dim=1)[1]

def test_max_dim_only_indices():
    inputs = [tu.randn(3, 4)]
    custom_test_helper(MaxDimOnlyIndicesModule(), inputs, "byteir.arg_max")

class MaxDimKeepDimOnlyIndicesModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return torch.max(x, dim=1, keepdim=True)[1]

def test_max_dim_keepdim_only_indices():
    inputs = [tu.randn(3, 4)]
    custom_test_helper(MaxDimKeepDimOnlyIndicesModule(), inputs, "byteir.arg_max")

class ArgmaxModule(torch.nn.Module):
    def forward(self, x):
        return torch.argmax(x)

def test_argmax_flatten():
    custom_test_helper(ArgmaxModule(), [tu.randn(4, 5)], "byteir.arg_max")

class ArgmaxDimModule(torch.nn.Module):
    def forward(self, x):
        return torch.argmax(x, dim=1)

def test_argmax_dim():
    custom_test_helper(ArgmaxDimModule(), [tu.randn(4, 5)], "byteir.arg_max")

class ArgminDimModule(torch.nn.Module):
    def forward(self, x):
        return torch.argmin(x, dim=1)

def test_argmin_dim():
    custom_test_helper(ArgminDimModule(), [tu.randn(4, 5)], "byteir.arg_min")

# ==============================================================================

class GeluModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return torch.nn.functional.gelu(x, approximate="tanh")

def test_gelu():
    inputs = [tu.randn(3, 4)]
    custom_test_helper(GeluModule(), inputs, "byteir.gelu")

def test_gelu_without_byteir_custom_op():
    inputs = [tu.randn(3, 4)]
    mlir_module = compile(GeluModule(), inputs, "stablehlo", backend_legal_ops=[])
    mlir_str = mlir_module.operation.get_asm()
    assert "stablehlo.custom_call" not in mlir_str
    assert "byteir.gelu" not in mlir_str

# ==============================================================================

class UpsampleNearest2dModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        #FIXME: use torch.nn.interpolate to avoid torch.jit.trace
        return torch.ops.aten.upsample_nearest2d.vec(x, (11, 25), None)

def test_resize():
    inputs = [tu.randn(3, 3, 10, 20)]
    model = UpsampleNearest2dModule()
    module = compile(torch.jit.trace(model, inputs), inputs, "stablehlo")
    numerical_test_helper(module, inputs, model(*inputs))
