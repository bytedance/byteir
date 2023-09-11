import torch
from torch.library import Library

OPERATORS = []


def op(schema):
    def inner(f):
        # TODO: Refactor the Library API so this is less rage inducing
        # TODO: Perhaps the namespace should be directly based on Python
        # module
        if '::' in schema:
            ns = schema.split('::', 2)[0]
        else:
            ns = 'contrib'
        # TODO: Library doesn't allow FRAGMENT, need to allow it
        lib = Library(ns, 'FRAGMENT')
        name = lib.define(schema)
        if '::' in name:
            name = name.split('::', 2)[1]
        lib.impl(name, f, 'CompositeExplicitAutograd')
        OPERATORS.append(lib)
        return getattr(getattr(torch.ops, ns), name)
    return inner


@op("byteir::flash_attn_fwd(Tensor q, Tensor k, Tensor v, float dropout_p, float softmax_scale, bool casual, bool return_softmax) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)")
def byteir_flash_attn_fwd(q, k, v, dropout_p, softmax_scale, causal, return_softmax):
    sizes = q.shape
    batch_size = sizes[0]
    seqlen_q = sizes[1]
    num_heads = sizes[2]
    seqlen_k = k.shape[1]

    rng = torch.empty((2), dtype=torch.int64, device='meta')
    softmax_lse = torch.empty(
        (batch_size, num_heads, seqlen_q), dtype=torch.float, device='meta')
    p = None
    if (return_softmax):
        p = torch.empty((batch_size, num_heads, seqlen_q,
                        seqlen_k), dtype=torch.float, device='meta')
    q_padded = q
    k_padded = k
    v_padded = v
    out = torch.empty_like(q_padded)
    out_padded = torch.empty_like(out)
    return out, q_padded, k_padded, v_padded, out_padded, softmax_lse, p, rng


@op("byteir::flash_attn_bwd(Tensor dout, Tensor q, Tensor k, Tensor v, Tensor out, Tensor softmax_lse, float dropout_p, float softmax_scale, bool casual, Tensor rng) -> (Tensor, Tensor, Tensor, Tensor, Tensor)")
def byteir_flash_attn_bwd(dout, q, k, v, out, softmax_lse, dropout_p, softmax_scale, causal, rng_state):
    sizes = q.shape
    batch_size = sizes[0]
    seqlen_q = sizes[1]
    num_heads = sizes[2]
    seqlen_q_rounded = ((seqlen_q+127)//128)*128
    head_size = sizes[3]
    head_size_rounded = ((head_size+31)//32)*32
    dq_accum = torch.empty((batch_size, num_heads, seqlen_q_rounded, head_size_rounded), dtype=torch.float, device='meta')
    softmax_d = torch.empty((batch_size, num_heads, seqlen_q_rounded), dtype=torch.float, device='meta')
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    return dq, dk, dv, softmax_d, dq_accum


class CustomFlashAttnFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, dropout_p, softmax_scale, causal, return_softmax):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        assert q.dtype == torch.float16 or q.dtype == torch.bfloat16
        # Save rng_state because the backward pass will regenerate the dropout mask
        out, q_pad, k_pad, v_pad, out_pad, softmax_lse, S_dmask, rng_state = torch.ops.byteir.flash_attn_fwd(
            q, k, v, dropout_p, softmax_scale,
            causal, (return_softmax and dropout_p > 0)
        )
        out = out.transpose(1, 2)
        ctx.save_for_backward(q_pad, k_pad, v_pad, out_pad, softmax_lse, rng_state)
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        return out if not return_softmax else (out, softmax_lse, S_dmask)

    @staticmethod
    def backward(ctx, dout, *args):
        dout = dout.transpose(1, 2)
        q_pad, k_pad, v_pad, out_pad, softmax_lse, rng_state = ctx.saved_tensors
        sizes = q_pad.shape

        dq, dk, dv, d_softmax, dq_accum = torch.ops.byteir.flash_attn_bwd(
            dout, q_pad, k_pad, v_pad, out_pad, softmax_lse, ctx.dropout_p, ctx.softmax_scale, ctx.causal, rng_state
        )
        dq = dq[..., :dout.shape[-1]]  # We could have padded the head dimension
        dk = dk[..., :dout.shape[-1]]
        dv = dv[..., :dout.shape[-1]]
        return dq, dk, dv, None, None, None, None


def flash_attn_func(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=1.0):
    """dropout_p should be set to 0.0 during evaluation
    Supports multi-query and grouped-query attention (MQA/GQA) by passing in KV with fewer heads
    than Q. Note that the number of heads in KV must be divisible by the number of heads in Q.
    For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
    0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.

    Arguments:
        q: (batch_size, seqlen, nheads, headdim)
        k: (batch_size, seqlen, nheads_k, headdim)
        v: (batch_size, seqlen, nheads_k, headdim)
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).

    Return:
        out: (batch_size, seqlen, nheads, headdim).
        softmax_lse [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
        S_dmask [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen, seqlen).
            The output of softmax (possibly with different scaling). It also encodes the dropout
            pattern (negative means that location was dropped, nonnegative means it was kept).
    """
    return CustomFlashAttnFunc.apply(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), dropout_p, scale, is_causal, False)


def replace_flash_attn(gm: torch.fx.GraphModule) -> torch.nn.Module:
    for node in gm.graph.nodes:
        if node.op == 'call_function' and node.target == torch.ops.aten.scaled_dot_product_attention:
            node.target = flash_attn_func

    gm.graph.lint()
    gm.recompile()
    return gm
