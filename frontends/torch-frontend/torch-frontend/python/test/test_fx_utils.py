import torch
import torch.fx as fx
import torch_frontend
from torch_frontend.fx_utils import _replace_aten_full_arugment

class FullModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        y = torch.ops.aten.full(x.shape, True, dtype=torch.bool)
        return y


def test_full_bool_pattern():
    fx_g = fx.symbolic_trace(FullModule())
    fx_g = _replace_aten_full_arugment(fx_g)
    module = torch.jit.script(fx_g)
