from typing import List, Optional
import torch
from torch._functorch.compile_utils import strip_overloads

def _returns_nothing(fx_g: torch.fx.GraphModule) -> bool:
    for node in fx_g.graph.nodes:
        if node.op == "output":
            assert len(node.args) == 1, "Output node must have a single argument"
            node_arg = node.args[0]
            if isinstance(node_arg, tuple):
                return len(node_arg) == 0
    return False


def _unwrap_single_tuple_return(fx_g: torch.fx.GraphModule) -> bool:
    """
    Replace tuple with tuple element in functions that return one-element tuples.
    Returns true if an unwrapping took place, and false otherwise.
    """
    unwrapped_tuple = False
    for node in fx_g.graph.nodes:
        if node.op == "output":
            assert len(node.args) == 1, "Output node must have a single argument"
            node_arg = node.args[0]
            if isinstance(node_arg, tuple):
                if len(node_arg) == 1:
                    node.args = (node_arg[0],)
                    unwrapped_tuple = True
                    break

    if unwrapped_tuple:
        fx_g.graph.lint()
        fx_g.recompile()
    return unwrapped_tuple


def _list_return_to_tuple_return(fx_g: torch.fx.GraphModule) -> bool:
    replaced_list = False
    for node in fx_g.graph.nodes:
        if node.op == "output":
            assert len(node.args) == 1, "Output node must have a single argument"
            node_arg = node.args[0]
            if isinstance(node_arg, list):
                node.args = (tuple(node_arg),)
                replaced_list = True
                break

    if replaced_list:
        fx_g.graph.lint()
        fx_g.recompile()
    return replaced_list


def _remove_nones(fx_g: torch.fx.GraphModule) -> List[int]:
    removed_indexes = []
    for node in fx_g.graph.nodes:
        if node.op == "output":
            assert len(node.args) == 1, "Output node must have a single argument"
            node_arg = node.args[0]
            if isinstance(node_arg, (list, tuple)):
                node_arg = list(node_arg)
                node_args_len = len(node_arg)
                for i in range(node_args_len):
                    curr_index = node_args_len - (i + 1)
                    if node_arg[curr_index] is None:
                        removed_indexes.append(curr_index)
                        node_arg.pop(curr_index)
                node.args = (tuple(node_arg),)
                break

    if len(removed_indexes) > 0:
        fx_g.graph.lint()
        fx_g.graph.eliminate_dead_code()
        fx_g.recompile()
    removed_indexes.sort()
    return removed_indexes


def threshold_backward_pattern(grad_output, inp, threshold):
    return torch.ops.aten.threshold_backward(grad_output, inp, threshold)

def threshold_backward_replacement(grad_output, inp, threshold):
    true_branch = torch.zeros_like(grad_output)
    cond = torch.le(inp, threshold)
    return torch.where(cond, true_branch, grad_output)

def squeeze_dims_pattern(tensor, dims):
    return torch.ops.aten.squeeze.dims(tensor, dims)

def squeeze_dims_replacement(tensor, dims):
    return  torch.ops.prims.squeeze(tensor, dims)

def unsafe_index_put_pattern(self, indices, values, accumulate):
    return torch.ops.aten._unsafe_index_put(self, indices, values, accumulate)

def unsafe_index_put_replacement(self, indices, values, accumulate):
    return  torch.ops.aten.index_put_.hacked_twin(self, indices, values, accumulate)

def list_decomposed_ops():
    return [
        torch.ops.aten._native_batch_norm_legit_functional,
        torch.ops.aten.native_batch_norm_backward,
        torch.ops.aten.native_layer_norm_backward,
        torch.ops.aten.embedding_dense_backward,
        torch.ops.aten.select_backward,
        torch.ops.aten.slice_backward
    ]

def preprocess_fx_graph(fx_graph: torch.fx.GraphModule):
    if _returns_nothing(fx_graph):
        return fx_graph

    torch.fx.replace_pattern(fx_graph, squeeze_dims_pattern, squeeze_dims_replacement)
    torch.fx.replace_pattern(fx_graph, unsafe_index_put_pattern, unsafe_index_put_replacement)
    was_unwrapped = _unwrap_single_tuple_return(fx_graph)
    was_list_replaced = _list_return_to_tuple_return(fx_graph)
    removed_none_indexes = _remove_nones(fx_graph)
    strip_overloads(fx_graph)
    torch.fx.replace_pattern(fx_graph, threshold_backward_pattern, threshold_backward_replacement)
    return fx_graph
