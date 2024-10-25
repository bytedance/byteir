import torch
import math

@torch.library.custom_op("byteir::flash_attn_fwd", mutates_args=())
def byteir_flash_attn_fwd(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, dropout_p: float, softmax_scale: float, causal: bool, return_softmax: bool
) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
    sizes = q.shape
    batch_size = sizes[0]
    seqlen_q = sizes[1]
    num_heads = sizes[2]
    seqlen_k = k.shape[1]

    rng = torch.empty((2), dtype=torch.int64, device="meta")
    softmax_lse = torch.empty(
        (batch_size, num_heads, seqlen_q), dtype=torch.float, device="meta"
    )
    p = None
    if return_softmax:
        p = torch.empty(
            (batch_size, num_heads, seqlen_q, seqlen_k),
            dtype=torch.float,
            device="meta",
        )
    q_padded = q
    k_padded = k
    v_padded = v
    out = torch.empty_like(q_padded)
    out_padded = torch.empty_like(out)
    return out, q_padded, k_padded, v_padded, out_padded, softmax_lse, p, rng

@torch.library.register_fake("byteir::flash_attn_fwd")
def byteir_flash_attn_fwd(q, k, v, dropout_p, softmax_scale, causal, return_softmax):
    sizes = q.shape
    batch_size = sizes[0]
    seqlen_q = sizes[1]
    num_heads = sizes[2]
    seqlen_k = k.shape[1]

    rng = torch.empty((2), dtype=torch.int64, device="meta")
    softmax_lse = torch.empty(
        (batch_size, num_heads, seqlen_q), dtype=torch.float, device="meta"
    )
    p = None
    if return_softmax:
        p = torch.empty(
            (batch_size, num_heads, seqlen_q, seqlen_k),
            dtype=torch.float,
            device="meta",
        )
    q_padded = q
    k_padded = k
    v_padded = v
    out = torch.empty_like(q_padded)
    out_padded = torch.empty_like(out)
    return out, q_padded, k_padded, v_padded, out_padded, softmax_lse, p, rng


@torch.library.custom_op("byteir::flash_attn_bwd", mutates_args=())
def byteir_flash_attn_bwd(
    dout: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, out: torch.Tensor, softmax_lse: torch.Tensor, dropout_p: float, softmax_scale: float, causal: bool, rng_state: torch.Tensor
) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
    sizes = q.shape
    batch_size = sizes[0]
    seqlen_q = sizes[1]
    num_heads = sizes[2]
    seqlen_q_rounded = ((seqlen_q + 127) // 128) * 128
    head_size = sizes[3]
    head_size_rounded = ((head_size + 31) // 32) * 32
    dq_accum = torch.empty(
        (batch_size, num_heads, seqlen_q_rounded, head_size_rounded),
        dtype=torch.float,
        device="meta",
    )
    softmax_d = torch.empty(
        (batch_size, num_heads, seqlen_q_rounded), dtype=torch.float, device="meta"
    )
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    return dq, dk, dv, softmax_d, dq_accum


@torch.library.register_fake("byteir::byteir_flash_attn_bwd")
def byteir_flash_attn_bwd(
    dout, q, k, v, out, softmax_lse, dropout_p, softmax_scale, causal, rng_state
):
    sizes = q.shape
    batch_size = sizes[0]
    seqlen_q = sizes[1]
    num_heads = sizes[2]
    seqlen_q_rounded = ((seqlen_q + 127) // 128) * 128
    head_size = sizes[3]
    head_size_rounded = ((head_size + 31) // 32) * 32
    dq_accum = torch.empty(
        (batch_size, num_heads, seqlen_q_rounded, head_size_rounded),
        dtype=torch.float,
        device="meta",
    )
    softmax_d = torch.empty(
        (batch_size, num_heads, seqlen_q_rounded), dtype=torch.float, device="meta"
    )
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    return dq, dk, dv, softmax_d, dq_accum


@torch.library.custom_op("byteir::flash_attn_kvcache", mutates_args())
def byteir_flash_attn_kvcache(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, kcache: torch.Tensor, vcache: torch.Tensor, seqlen_k: torch.Tensor, softmax_scale: float, causal: bool
) -> (torch.Tensor, torch.Tensor):
    sizes = q.shape
    batch_size = sizes[0]
    seqlen_q = sizes[1]
    num_heads = sizes[2]

    softmax_lse = torch.empty(
        (batch_size, num_heads, seqlen_q), dtype=torch.float, device="meta"
    )
    out = torch.empty_like(q)
    return out, softmax_lse


@torch.library.register_fake("byteir::flash_attn_kvcache")
def byteir_flash_attn_kvcache(q, k, v, kcache, vcache, seqlen_k, softmax_scale, causal):
    sizes = q.shape
    batch_size = sizes[0]
    seqlen_q = sizes[1]
    num_heads = sizes[2]

    softmax_lse = torch.empty(
        (batch_size, num_heads, seqlen_q), dtype=torch.float, device="meta"
    )
    out = torch.empty_like(q)
    return out, softmax_lse


class CustomFlashAttnFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, dropout_p, softmax_scale, causal, return_softmax):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        assert q.dtype == torch.float16 or q.dtype == torch.bfloat16
        # Save rng_state because the backward pass will regenerate the dropout mask
        out, q_pad, k_pad, v_pad, out_pad, softmax_lse, S_dmask, rng_state = (
            torch.ops.byteir.flash_attn_fwd(
                q,
                k,
                v,
                dropout_p,
                softmax_scale,
                causal,
                (return_softmax and dropout_p > 0),
            )
        )
        # output also needs to be transposed
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
            dout,
            q_pad,
            k_pad,
            v_pad,
            out_pad,
            softmax_lse,
            ctx.dropout_p,
            ctx.softmax_scale,
            ctx.causal,
            rng_state,
        )
        dq = dq[..., : dout.shape[-1]]  # We could have padded the head dimension
        dk = dk[..., : dout.shape[-1]]
        dv = dv[..., : dout.shape[-1]]
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
    # q, k, v needs to be transposed for flash attn v2
    if attn_mask == None and is_causal:
        return CustomFlashAttnFunc.apply(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            dropout_p,
            scale,
            is_causal,
            False,
        )
    else:
        return torch.ops.aten.scaled_dot_product_attention


def flash_attn_functional_func(
    q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None
):
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
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).

    Return:
        out: (batch_size, seqlen, nheads, headdim).
        softmax_lse [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
        S_dmask [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen, seqlen).
            The output of softmax (possibly with different scaling). It also encodes the dropout
            pattern (negative means that location was dropped, nonnegative means it was kept).
    """
    if attn_mask == None and is_causal:
        # q, k, v needs to be transposed for flash attn v2
        scale_factor = 1 / math.sqrt(q.size(-1)) if scale is None else scale
        return CustomFlashAttnFunc.apply(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            dropout_p,
            scale_factor,
            is_causal,
            False,
        )
    else:
        return torch._C._nn.scaled_dot_product_attention


def replace_flash_attn(gm: torch.fx.GraphModule) -> torch.nn.Module:
    for node in gm.graph.nodes:
        if (
            node.op == "call_function"
            and node.target == torch.ops.aten.scaled_dot_product_attention
        ):
            node.target = flash_attn_func
        if (
            node.op == "call_function"
            and node.target == torch._C._nn.scaled_dot_product_attention
        ):
            node.target = flash_attn_functional_func

    gm.graph.lint()
    gm.recompile()
    return gm
