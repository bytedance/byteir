import torch

import torch_frontend
from torch_mlir import fx

class UnbindIntModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.unbind(x, 1)

def test_unbind_int_op():
    m = fx.export_and_import(
        UnbindIntModule(),
        torch.randn(3, 4),
        func_name="test_unbind_int",
        decomposition_table=False,
    )
    mlir_str = m.operation.get_asm()
    assert "aten.unbind.int" in mlir_str
    assert "aten.__getitem__.t" in mlir_str
