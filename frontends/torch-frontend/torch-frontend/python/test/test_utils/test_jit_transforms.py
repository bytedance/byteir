import torch
import sys

from torch_frontend.utils import replace_copy_fill_with_slice_scatter
from torch_frontend import compile


# fill_ related.
class NSliceSliceFill(torch.nn.Module):
    def __init__(self):
        super().__init__()

    # (4,4,4)
    def forward(self, x):
        x1 = torch.ops.aten.slice(x, 1, 0, sys.maxsize, 1)
        x2 = torch.ops.aten.slice(x1, 0, 0, sys.maxsize, 1)
        x3 = torch.ops.aten.slice(x2, 2, 0, sys.maxsize, 1)
        x4 = torch.ops.aten.slice(x3, 0, 0, 2, 1)
        _ = torch.ops.aten.fill_(x4, 0)
        x = x + 1
        return x


class NSliceSelectFill(torch.nn.Module):
    def __init__(self):
        super().__init__()

    # (4,4,4)
    def forward(self, x):
        x1 = torch.ops.aten.slice(x, 1, 0, sys.maxsize, 1)
        x2 = torch.ops.aten.slice(x1, 0, 0, sys.maxsize, 1)
        x3 = torch.ops.aten.slice(x2, 2, 0, sys.maxsize, 1)
        x4 = torch.ops.aten.select(x3, 0, 2)
        _ = torch.ops.aten.fill_(x4, 0)
        x = x + 1
        return x


class SelectSliceFill(torch.nn.Module):
    def __init__(self):
        super().__init__()

    # (4, 4, 4)
    def forward(self, x):
        x1 = torch.select(x, 0, 1)
        x2 = torch.ops.aten.slice(x1, 0, 0, sys.maxsize)
        _ = torch.ops.aten.fill_(x2, 0)
        x = x + 1
        return x


# copy_ related.
class NSliceSliceCopy(torch.nn.Module):
    def __init__(self):
        super().__init__()

    # (4,4,4)
    def forward(self, x):
        x1 = torch.ops.aten.slice(x, 1, 0, sys.maxsize, 1)
        x2 = torch.ops.aten.slice(x1, 0, 0, sys.maxsize, 1)
        x3 = torch.ops.aten.slice(x2, 2, 0, sys.maxsize, 1)
        x4 = torch.ops.aten.slice(x3, 0, 0, 2, 1)
        # import pdb;pdb.set_trace()
        zeros = torch.zeros((2, 4, 4))
        _ = torch.ops.aten.copy_(x4, zeros)
        x = x + 1
        return x


class NSliceSelectCopy(torch.nn.Module):
    def __init__(self):
        super().__init__()

    # (4,4,4)
    def forward(self, x):
        x1 = torch.ops.aten.slice(x, 1, 0, sys.maxsize, 1)
        x2 = torch.ops.aten.slice(x1, 0, 0, sys.maxsize, 1)
        x3 = torch.ops.aten.slice(x2, 2, 0, sys.maxsize, 1)
        x4 = torch.ops.aten.select(x3, 0, 2)
        zeros = torch.zeros((4, 4))
        _ = torch.ops.aten.copy_(x4, zeros)
        x = x + 1
        return x


class SelectSliceCopy(torch.nn.Module):
    def __init__(self):
        super().__init__()

    # (4, 4, 4)
    def forward(self, x):
        x1 = torch.select(x, 0, 1)
        x2 = torch.ops.aten.slice(x1, 0, 0, sys.maxsize)
        _ = torch.ops.aten.fill_(x2, 0)
        x = x + 1
        return x


def _test_model(model_class):
    input = torch.randn(4, 4, 4)
    model = model_class()
    golden = model(input)

    ts_model = torch.jit.trace(model, (input,), check_trace=False)
    replace_copy_fill_with_slice_scatter(ts_model.graph)

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
            # for use in node.uses():
            user = uses[0].user
            assert user.kind() == "aten::slice_scatter"

        if node.kind() == "aten::slice_scatter":
            has_slice_scatter = True
    assert has_slice_scatter

    out = ts_model(input)
    assert torch.allclose(golden, out)


def test_rewrite():
    _test_model(NSliceSliceFill)
    _test_model(NSliceSelectFill)
    _test_model(NSliceSliceCopy)
    _test_model(NSliceSelectCopy)
    _test_model(SelectSliceCopy)


test_rewrite()
