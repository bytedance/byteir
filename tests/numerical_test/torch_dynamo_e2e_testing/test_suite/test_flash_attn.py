import torch
from ..backend import byteir_compile_fx


class FlashAttnModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v):
        return torch.ops.aten.scaled_dot_product_attention(
            q, k, v, attn_mask=None, is_causal=True, scale=0.2, dropout_p=0.0
        )


def test_flash_attn_unit():
    model = FlashAttnModel().to("cuda")
    model_golden = FlashAttnModel().to("cuda")
    model_golden.load_state_dict(model.state_dict())

    q = torch.rand(
        2, 12, 256, 128, dtype=torch.half, device="cuda:0", requires_grad=True
    )
    k = torch.rand(
        2, 12, 256, 128, dtype=torch.half, device="cuda:0", requires_grad=True
    )
    v = torch.rand(
        2, 12, 256, 128, dtype=torch.half, device="cuda:0", requires_grad=True
    )
    q_clone = q.clone().detach().requires_grad_()
    k_clone = k.clone().detach().requires_grad_()
    v_clone = v.clone().detach().requires_grad_()
    model.zero_grad(set_to_none=True)
    model_golden.zero_grad(set_to_none=True)
    optimized_model = torch.compile(model, backend=byteir_compile_fx)
    output = optimized_model(q, k, v)
    golden_output = model_golden(q_clone, k_clone, v_clone)
    torch.testing.assert_close(output, golden_output)
    output.mean().backward()
    golden_output.mean().backward()
    torch.testing.assert_close(q.grad, q_clone.grad)
    torch.testing.assert_close(k.grad, k_clone.grad)
    torch.testing.assert_close(v.grad, v_clone.grad)


class FlashAttnFunctionalModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v):
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)


def test_flash_attn_functional_unit():
    model = FlashAttnFunctionalModel().to("cuda")
    model_golden = FlashAttnFunctionalModel().to("cuda")
    model_golden.load_state_dict(model.state_dict())

    q = torch.rand(
        2, 12, 256, 128, dtype=torch.half, device="cuda:0", requires_grad=True
    )
    k = torch.rand(
        2, 12, 256, 128, dtype=torch.half, device="cuda:0", requires_grad=True
    )
    v = torch.rand(
        2, 12, 256, 128, dtype=torch.half, device="cuda:0", requires_grad=True
    )
    q_clone = q.clone().detach().requires_grad_()
    k_clone = k.clone().detach().requires_grad_()
    v_clone = v.clone().detach().requires_grad_()
    model.zero_grad(set_to_none=True)
    model_golden.zero_grad(set_to_none=True)
    optimized_model = torch.compile(model, backend=byteir_compile_fx)
    output = optimized_model(q, k, v)
    golden_output = model_golden(q_clone, k_clone, v_clone)
    torch.testing.assert_close(output, golden_output)
    output.mean().backward()
    golden_output.mean().backward()
    torch.testing.assert_close(q.grad, q_clone.grad)
    torch.testing.assert_close(k.grad, k_clone.grad)
    torch.testing.assert_close(v.grad, v_clone.grad)


import flash_attn
from flash_attn.flash_attn_interface import flash_attn_with_kvcache


class FlashAttnKVCacheModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, kcache, vcache, seqlen):
        out, _ = torch.ops.byteir.flash_attn_kvcache(
            q, k, v, kcache, vcache, seqlen, 0.5, True
        )
        return out


def test_flash_attn_kvcache():
    model = FlashAttnKVCacheModel().to("cuda")
    b = 2
    seq_len = 128
    seq_len_q = 1
    num_heads = 3
    head_dims = 32
    causal = True
    input_len = b * seq_len_q * num_heads * head_dims
    q = (
        torch.arange(input_len, dtype=torch.float16, device="cuda").reshape(
            (b, seq_len_q, num_heads, head_dims)
        )
        / 30000
    )
    k = (
        torch.arange(input_len, dtype=torch.float16, device="cuda").reshape(
            (b, seq_len_q, num_heads, head_dims)
        )
        / 40000
    )
    v = (
        torch.arange(input_len, dtype=torch.float16, device="cuda").reshape(
            (b, seq_len_q, num_heads, head_dims)
        )
        / 50000
    )

    cache_len = b * seq_len * num_heads * head_dims
    kcache = (
        torch.arange(cache_len, dtype=torch.float16, device="cuda").reshape(
            (b, seq_len, num_heads, head_dims)
        )
        / 40000
    )
    vcache = (
        torch.arange(cache_len, dtype=torch.float16, device="cuda").reshape(
            (b, seq_len, num_heads, head_dims)
        )
        / 50000
    )
    kcache_clone = kcache.clone()
    vcache_clone = vcache.clone()
    cache_seqlens = torch.ones((b,), dtype=torch.int32, device="cuda") * 64
    window_size = (-1, -1)
    cos = None
    sin = None
    cache_batch_idx = None
    out_golden = flash_attn_with_kvcache(
        q,
        kcache,
        vcache,
        k,
        v,
        cos,
        sin,
        cache_seqlens,
        cache_batch_idx,
        softmax_scale=0.5,
        causal=causal,
        window_size=window_size,
        rotary_interleaved=False,
        num_splits=1,
    )
    optimized_model = torch.compile(model, backend=byteir_compile_fx)
    out = optimized_model(q, k, v, kcache_clone, vcache_clone, cache_seqlens)

    torch.testing.assert_close(out, out_golden)
    torch.testing.assert_close(kcache, kcache_clone)
    torch.testing.assert_close(vcache, vcache_clone)
