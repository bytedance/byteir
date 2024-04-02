import torch
import torch as tu

import torch_frontend
from torch_frontend import convert_to_mhlo_via_torch_mlir

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
    module = convert_to_mhlo_via_torch_mlir(module, inputs)
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
    inputs = [tu.randint(0, 2, (10, 20)).long()]
    module = torch.jit.trace(module, inputs) # x.bool() can't be scripted
    module = convert_to_mhlo_via_torch_mlir(module , inputs)
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
    module = convert_to_mhlo_via_torch_mlir(module, inputs)
    print(module.operation.get_asm(large_elements_limit=10, enable_debug_info=False))

# ==============================================================================

class MaxDimModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return torch.max(x, dim=1)[0]

def test_max_dim():
    inputs = [tu.randn(3, 4)]
    module = convert_to_mhlo_via_torch_mlir(MaxDimModule(), inputs)
    print(module.operation.get_asm())

class MaxDimKeepDimModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return torch.max(x, dim=1, keepdim=True)[0]

def test_max_dim_keepdim():
    inputs = [tu.randn(3, 4)]
    module = convert_to_mhlo_via_torch_mlir(MaxDimKeepDimModule(), inputs)
    print(module.operation.get_asm())

# ==============================================================================

class ListModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return [x, x, x]

def test_return_list():
    inputs = [tu.randn(3, 4)]
    module = convert_to_mhlo_via_torch_mlir(ListModule(), inputs)
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
    module = convert_to_mhlo_via_torch_mlir(ArangeModule(), inputs)
    print(module.operation.get_asm())

class ClampDerefineModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        shape = torch.ops.prim.NumToTensor(x.shape[1])
        return torch.ops.aten.clamp(x, 0, shape)

def test_clamp_derefine():
    inputs = [tu.randn(3, 4)]
    module = convert_to_mhlo_via_torch_mlir(ClampDerefineModule(), inputs)
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
    module = convert_to_mhlo_via_torch_mlir(Tuple1Module(), inputs)
    print(module.operation.get_asm())

class Tuple2Module(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return (x, x)

def test_tuple_one_tensor():
    inputs = [tu.randn(3, 4)]
    module = convert_to_mhlo_via_torch_mlir(Tuple2Module(), inputs)
    print(module.operation.get_asm())
