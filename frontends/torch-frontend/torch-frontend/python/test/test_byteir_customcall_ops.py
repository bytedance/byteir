import torch
import torch as tu

import torch_frontend
from torch_frontend import convert_to_mhlo_via_torch_mlir

def custom_test_helper(module, inputs, custom_op_name):
    mlir_module = convert_to_mhlo_via_torch_mlir(module, inputs)
    mlir_str = mlir_module.operation.get_asm(large_elements_limit=10, enable_debug_info=False)
    compare_str = "stablehlo.custom_call @{}".format(custom_op_name)
    # print(mlir_str)
    assert compare_str in mlir_str

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

# ==============================================================================

#TODO(lyq): add more tests