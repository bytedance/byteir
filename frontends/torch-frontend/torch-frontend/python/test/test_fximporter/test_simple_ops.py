import torch
import torch_frontend
from torch_frontend import compile_dynamo_model

# ==============================================================================

class AtenSliceModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x[1:]

def test_slice():
    inputs = (torch.randn(4),)
    module = AtenSliceModule()
    prog = torch.export.export(module, inputs)
    module = compile_dynamo_model(prog, "raw")
    print(module.operation.get_asm())

# ==============================================================================

class AtenNonZeroModule(torch.nn.Module):
    def forward(self, x):
        return torch.nonzero(x)

def test_nonzero():
    inputs = (torch.tensor([1, 0, 0, 1, 1]),)
    prog = torch.export.export(AtenNonZeroModule(), inputs)
    module = compile_dynamo_model(prog, "raw")
    print(module.operation.get_asm())

# ==============================================================================

class ViewDtypeModule(torch.nn.Module):
    def forward(self, x):
        return x.view(torch.int8)

def test_view_dtype():
    inputs = (torch.rand(4, 5),)
    prog = torch.export.export(ViewDtypeModule(), inputs)
    print(prog)
    module = compile_dynamo_model(prog, "raw") # note: torch2.1 export has bug on view.dtype 
    print(module.operation.get_asm())

# ==============================================================================

class MLPModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(10, 10)
    def forward(self, x):
        return self.fc(x)

def test_mlp():
    inputs = (torch.randn(10, 10),)
    prog = torch.export.export(MLPModule(), inputs)
    module = compile_dynamo_model(prog, "stablehlo")
    mlir_str = module.operation.get_asm()
    print(mlir_str)
    assert "dense_resource" not in mlir_str

if __name__ == "__main__":
    test_view_dtype()