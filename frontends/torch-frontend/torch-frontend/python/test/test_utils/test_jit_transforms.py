import torch
import sys

from torch_frontend.utils import replace_copy_fill_with_slice_scatter


def _test_helper(model_class, inputs):
    # input = torch.randn(4, 4, 4)
    model = model_class()
    golden = model(*inputs)

    ts_model = torch.jit.trace(model, inputs, check_trace=False)
    replace_copy_fill_with_slice_scatter(ts_model.graph)
    print(f"{model_class.__name__}: {ts_model.graph}")

    # validate graph.
    has_slice_scatter = False
    for node in ts_model.graph.nodes():
        assert node.kind() not in [
            "aten::copy_",
            "aten::fill_",
            "aten::select",
        ], ts_model.graph

        if node.kind() == "aten::slice":
            uses = node.output().uses()
            assert len(uses) == 1
            user = uses[0].user
            assert user.kind() == "aten::slice_scatter"

        if node.kind() == "aten::slice_scatter":
            has_slice_scatter = True
    assert has_slice_scatter

    out = ts_model(*inputs)
    torch.testing.assert_close(golden, out)

######################################################################
# fill_ related

class NSliceSliceFill(torch.nn.Module):
    def __init__(self):
        super().__init__()

    # (4, 4, 4)
    def forward(self, x):
        x1 = torch.ops.aten.slice(x, 1, 0, sys.maxsize, 1)
        x2 = torch.ops.aten.slice(x1, 0, 0, sys.maxsize, 1)
        x3 = torch.ops.aten.slice(x2, 2, 0, sys.maxsize, 1)
        x4 = torch.ops.aten.slice(x3, 0, 0, 2, 1)
        _ = torch.ops.aten.fill_(x4, 2.0)
        x = x + 1
        return x

def test_nslice_slice_fill():
    inputs = [torch.randn(4, 4, 4)]
    _test_helper(NSliceSliceFill, inputs)


class NSliceSelectFill(torch.nn.Module):
    def __init__(self):
        super().__init__()

    # (4, 4, 4)
    def forward(self, x):
        x1 = torch.ops.aten.slice(x, 1, 0, sys.maxsize, 1)
        x2 = torch.ops.aten.slice(x1, 0, 0, sys.maxsize, 1)
        x3 = torch.ops.aten.slice(x2, 2, 0, sys.maxsize, 1)
        x4 = torch.ops.aten.select(x3, 0, 2)
        _ = torch.ops.aten.fill_(x4, 2.0)
        x = x + 1
        return x

def test_nslice_select_fill():
    inputs = [torch.randn(4, 4, 4)]
    _test_helper(NSliceSelectFill, inputs)


class SelectSliceFill(torch.nn.Module):
    def __init__(self):
        super().__init__()

    # (4, 4, 4)
    def forward(self, x):
        x1 = torch.select(x, 0, 1)
        x2 = torch.ops.aten.slice(x1, 0, 0, sys.maxsize)
        _ = torch.ops.aten.fill_(x2, -torch.inf)
        x = x + 1
        return x

def test_select_slice_fill():
    inputs = [torch.randn(4, 4, 4)]
    _test_helper(SelectSliceFill, inputs)


######################################################################
# copy_ related

class NSliceSliceCopy(torch.nn.Module):
    def __init__(self):
        super().__init__()

    # (4, 4, 4)
    def forward(self, x):
        x1 = torch.ops.aten.slice(x, 1, 0, sys.maxsize, 1)
        x2 = torch.ops.aten.slice(x1, 0, 0, sys.maxsize, 1)
        x3 = torch.ops.aten.slice(x2, 2, 0, sys.maxsize, 1)
        x4 = torch.ops.aten.slice(x3, 0, 0, 2, 1)
        zeros = torch.zeros((2, 4, 4))
        _ = torch.ops.aten.copy_(x4, zeros)
        x = x + 1
        return x

def test_nslice_slice_copy():
    inputs = [torch.randn(4, 4, 4)]
    _test_helper(NSliceSliceCopy, inputs)


class NSliceSelectCopy(torch.nn.Module):
    def __init__(self):
        super().__init__()

    # (4, 4, 4)
    def forward(self, x):
        x1 = torch.ops.aten.slice(x, 1, 0, sys.maxsize, 1)
        x2 = torch.ops.aten.slice(x1, 0, 0, sys.maxsize, 1)
        x3 = torch.ops.aten.slice(x2, 2, 0, sys.maxsize, 1)
        x4 = torch.ops.aten.select(x3, 0, 2)
        zeros = torch.zeros((4, 4))
        _ = torch.ops.aten.copy_(x4, zeros)
        x = x + 1
        return x

def test_nslice_select_copy():
    inputs = [torch.randn(4, 4, 4)]
    _test_helper(NSliceSelectCopy, inputs)

