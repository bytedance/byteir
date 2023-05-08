from typing import Optional, Sequence, Union, List
import torch

def decompose_aten_chunk_op(input: torch.Tensor, chunks: int, dim: int):
    total_size = input.shape[dim]
    piece_size = (total_size + chunks - 1) // chunks
    results = []
    piece_count = int((total_size + piece_size - 1) // piece_size)
    for i in range(piece_count):
        results = results + [
            torch.ops.aten.slice(
                input,
                dim,
                i * piece_size,
                min(i * piece_size + piece_size, total_size),
                1,
            )
        ]
    return results


def decompose_aten_split_Tensor_op(input: torch.Tensor, split_size_or_sections, dim: int):
    total_size = input.shape[dim]
    if isinstance(split_size_or_sections, int):
        piece_size = split_size_or_sections
        results = []
        piece_count = int((total_size + piece_size - 1) // piece_size)
        for i in range(piece_count):
            results = results + [
                torch.ops.aten.slice(
                    input,
                    dim,
                    i * piece_size,
                    min(i * piece_size + piece_size, total_size),
                    1,
                )
            ]
        return results
    elif isinstance(split_size_or_sections, list):
        assert sum(split_size_or_sections) == total_size
        begin = 0
        results = []
        for i in split_size_or_sections:
            results = results + [
                torch.ops.aten.slice(
                    input,
                    dim,
                    begin,
                    begin + i,
                    1,
                )
            ]
            begin = begin + i
        return results
    else:
        assert False, "split_size_or_sections should be int or list"


def decompose_aten_unbind_int_op(input: torch.Tensor, dim: int):
    total_size = input.shape[dim]
    results = []
    for i in range(total_size):
        results = results + [
            torch.ops.aten.select(input, dim, i)
        ]
    return results

def decompose_aten_bucketize_Tensor_op(
    input: torch.Tensor, boundaries: torch.Tensor, out_int32: bool, right: bool
):
    bcast_shape = input.shape + boundaries.shape
    bcast_input = torch.broadcast_to(input.unsqueeze(-1), bcast_shape)
    bcast_boundaries = torch.broadcast_to(boundaries, bcast_shape)
    if not right:
        be = bcast_boundaries >= bcast_input
        result = torch.argmax(be.long(), -1)

        if_exceed = input > boundaries[-1]
        result = torch.where(if_exceed, boundaries.shape[0], result)
        if out_int32:
            result = result.int()
        return result
    else:
        b = bcast_boundaries > bcast_input
        result = torch.argmax(b.long(), -1)

        if_exceed = input >= boundaries[-1]
        result = torch.where(if_exceed, boundaries.shape[0], result)
        if out_int32:
            result = result.int()
        return result


def register_decomposition_in_torchscript():
    torch.jit._register_decomposition(
        torch.ops.aten.chunk.default,
        torch.jit.script(decompose_aten_chunk_op).graph,
    )
    torch.jit._register_decomposition(
        torch.ops.aten.split.Tensor,
        torch.jit.script(decompose_aten_split_Tensor_op).graph,
    )
    # torch.jit._register_decomposition(
    #     torch.ops.aten.bucketize.Tensor,
    #     torch.jit.script(decompose_aten_bucketize_Tensor_op).graph,
    # )
    # torch.jit._register_decomposition(
    #     torch.ops.aten.unbind.int,
    #     torch.jit.script(decompose_aten_unbind_int_op).graph,
    # )
