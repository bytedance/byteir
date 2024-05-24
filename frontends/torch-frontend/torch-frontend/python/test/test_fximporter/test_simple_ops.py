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
    prog = torch.export.export(module, inputs, constraints=None)
    module = compile_dynamo_model(prog, "raw")
    print(module.operation.get_asm())

# ==============================================================================

class AtenNonZeroModule(torch.nn.Module):
    def forward(self, x):
        return torch.nonzero(x)

def test_nonzero():
    inputs = (torch.tensor([1, 0, 0, 1, 1]),)
    prog = torch.export.export(AtenNonZeroModule(), inputs, constraints=None)
    module = compile_dynamo_model(prog, "raw")
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
    prog = torch.export.export(MLPModule(), inputs, constraints=None)
    module = compile_dynamo_model(prog, "stablehlo")
    mlir_str = module.operation.get_asm()
    print(mlir_str)
    assert not "dense_resource" in mlir_str

if __name__ == "__main__":
    test_slice()
    test_nonzero()
    test_mlp()