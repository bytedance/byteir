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


# note: torch.jit.script doesn't support  torch.ops.aten.full([2, 1, 1, 128], True, dtype = torch.bool), replace it with  torch.ops.aten.full([2, 1, 1, 128], 1, dtype = torch.bool)
def _replace_aten_full_arugment(fx_g: torch.fx.GraphModule) -> torch.fx.GraphModule :
    def get_aten_target(node):
        if hasattr(node.target, 'overloadpacket'):
            return node.target.overloadpacket
        return node.target

    nodes = []
    for node in fx_g.graph.nodes:
        if get_aten_target(node) == torch.ops.aten.full:
            if node.args[1] == True or node.args[1] == False:
                nodes.append(node)
    for node in nodes:
        if node.args[1] == True:
            with fx_g.graph.inserting_after(node):
                new_node = fx_g.graph.call_function(torch.ops.aten.full, args=(node.args[0], 1), kwargs=node.kwargs)
                node.replace_all_uses_with(new_node)
                fx_g.graph.erase_node(node)
        if node.args[1] == False:
            with fx_g.graph.inserting_after(node):
                new_node = fx_g.graph.call_function(torch.ops.aten.full, args=(node.args[0], 0), kwargs=node.kwargs)
                node.replace_all_uses_with(new_node)
                fx_g.graph.erase_node(node)
    fx_g.graph.lint()
    fx_g.recompile()
    return fx_g


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

# LLaMA aten attention op pattern
def LLaMAAttnPattern(query, key, value, attn_mask, min_val, inv_scale, batch, num_head, fused_batch, seq_len, head_dim):
    transpose_3 = torch.ops.aten.transpose.int(key, 2, 3)
    expand_2 = torch.ops.aten.expand.default(query, [batch, num_head, seq_len, head_dim])
    clone = torch.ops.aten.clone.default(expand_2, memory_format = torch.contiguous_format)
    _unsafe_view_3 = torch.ops.aten._unsafe_view.default(clone, [fused_batch, seq_len, head_dim])
    expand_3 = torch.ops.aten.expand.default(transpose_3, [batch, num_head, head_dim, seq_len])
    clone_1 = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format)
    _unsafe_view_4 = torch.ops.aten._unsafe_view.default(clone_1, [fused_batch, head_dim, seq_len])
    bmm = torch.ops.aten.bmm.default(_unsafe_view_3, _unsafe_view_4)
    _unsafe_view_5 = torch.ops.aten._unsafe_view.default(bmm, [batch, num_head, seq_len, seq_len])
    div = torch.ops.aten.div.Tensor(_unsafe_view_5, inv_scale)
    add_5 = torch.ops.aten.add.Tensor(div, attn_mask)
    maximum = torch.ops.aten.maximum.default(add_5, min_val)
    _softmax = torch.ops.aten._softmax.default(maximum, -1, False)
    _to_copy_10 = torch.ops.aten._to_copy.default(_softmax, dtype = torch.float16)
    expand_4 = torch.ops.aten.expand.default(_to_copy_10, [batch, num_head, seq_len, seq_len])
    view_8 = torch.ops.aten.view.default(expand_4, [fused_batch, seq_len, seq_len]);  expand_4 = None
    expand_5 = torch.ops.aten.expand.default(value, [batch, num_head, seq_len, head_dim])
    clone_2 = torch.ops.aten.clone.default(expand_5, memory_format = torch.contiguous_format)
    _unsafe_view_6 = torch.ops.aten._unsafe_view.default(clone_2, [fused_batch, seq_len, head_dim])
    bmm_1 = torch.ops.aten.bmm.default(view_8, _unsafe_view_6)
    _unsafe_view_5 = torch.ops.aten._unsafe_view.default(bmm_1, [batch, num_head, seq_len, head_dim])
    return _softmax, _unsafe_view_5


def LLaMAAttnReplacement(query, key, value, attn_mask, min_val, inv_scale, batch, num_head, fused_batch, seq_len, head_dim):
    out, q_pad, k_pad, v_pad, out_pad, softmax_lse, S_dmask, rng_state = torch.ops.byteir.flash_attn_fwd(
        query,
        key,
        value,
        0.0,
        1.0/inv_scale,
        True,
        True
    )
    return S_dmask, out


def get_none_indices(fx_g: torch.fx.GraphModule) -> List[int]:
    none_indices = []
    for node in fx_g.graph.nodes:
        if node.op == "output":
            assert len(node.args) == 1, "Output node must have a single argument"
            node_arg = node.args[0]
            if isinstance(node_arg, (list, tuple)):
                node_arg = list(node_arg)
                node_args_len = len(node_arg)
                for i in range(node_args_len):
                    if node_arg[i] is None:
                        none_indices.append(i)
                break
    return none_indices


def list_decomposed_ops():
    return [
        torch.ops.aten._native_batch_norm_legit_functional,
        torch.ops.aten.native_batch_norm_backward,
        torch.ops.aten.native_layer_norm_backward,
        torch.ops.aten.embedding_dense_backward,
        torch.ops.aten.select_backward,
        torch.ops.aten.slice_backward,
        torch.ops.aten.native_dropout_backward,
        torch.ops.aten.tril
    ]


def preprocess_fx_graph(fx_graph: torch.fx.GraphModule):
    if _returns_nothing(fx_graph):
        return fx_graph

    torch.fx.replace_pattern(fx_graph, squeeze_dims_pattern, squeeze_dims_replacement)
    torch.fx.replace_pattern(fx_graph, unsafe_index_put_pattern, unsafe_index_put_replacement)
    torch.fx.replace_pattern(fx_graph, LLaMAAttnPattern, LLaMAAttnReplacement)
    was_unwrapped = _unwrap_single_tuple_return(fx_graph)
    was_list_replaced = _list_return_to_tuple_return(fx_graph)
    removed_none_indexes = _remove_nones(fx_graph)
    strip_overloads(fx_graph)
    torch.fx.replace_pattern(fx_graph, threshold_backward_pattern, threshold_backward_replacement)
    fx_graph = _replace_aten_full_arugment(fx_graph)
    return fx_graph
