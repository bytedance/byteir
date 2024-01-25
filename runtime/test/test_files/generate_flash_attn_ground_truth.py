import torch
import flash_attn
from flash_attn.flash_attn_interface import _flash_attn_forward, _flash_attn_backward, flash_attn_with_kvcache

assert flash_attn.__version__ == "2.4.2"

def outputs(tensor, filename):
    to_write = torch.flatten(tensor)
    of = open(filename, "w")
    for num in to_write:
        of.write(f"{num} ")
    of.close()

def generate_flash_attn_fwd_backward_data():
    b = 1
    seq_len = 128
    num_heads = 3
    head_dims = 32
    causal = True
    input_len = b * seq_len * num_heads * head_dims
    q = torch.arange(input_len, dtype=torch.float16, device='cuda').reshape((b, seq_len, num_heads, head_dims))/30000
    k = torch.arange(input_len, dtype=torch.float16, device='cuda').reshape((b, seq_len, num_heads, head_dims))/40000
    v = torch.arange(input_len, dtype=torch.float16, device='cuda').reshape((b, seq_len, num_heads, head_dims))/50000

    window_size = (-1, -1)
    out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state = _flash_attn_forward(q, k, v, 0.0, 0.5, causal, window_size, None, False)

    print("generating flash_attn fwd inputs...")
    outputs(q, "flash_attn_inputs_q.data")
    outputs(k, "flash_attn_inputs_k.data")
    outputs(v, "flash_attn_inputs_v.data")


    print("generating ground truth for flash_attn fwd output...")
    outputs(out, "flash_attn_fwd_outputs.data")


    print("generating flash_attn bwd inputs...")
    dout = torch.ones(out.shape, dtype=torch.float16, device='cuda')/32
    outputs(dout, "flash_attn_inputs_dout.data")


    dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)

    _flash_attn_backward(
        dout, q, k, v, out, softmax_lse,
        dq, dk, dv, 0, 0.5, causal, window_size, None, deterministic=True
    )

    print("generating ground truth for flash_attn bwd output...")
    outputs(dq, "flash_attn_bwd_outputs_dq.data")
    outputs(dk, "flash_attn_bwd_outputs_dk.data")
    outputs(dv, "flash_attn_bwd_outputs_dv.data")


def generate_flash_attn_kvcache_data():
    b = 2
    seq_len = 128
    seq_len_q = 1
    num_heads = 3
    head_dims = 32
    causal = True
    input_len = b * seq_len_q * num_heads * head_dims
    q = torch.arange(input_len, dtype=torch.float16, device='cuda').reshape((b, seq_len_q, num_heads, head_dims))/30000
    k = torch.arange(input_len, dtype=torch.float16, device='cuda').reshape((b, seq_len_q, num_heads, head_dims))/40000
    v = torch.arange(input_len, dtype=torch.float16, device='cuda').reshape((b, seq_len_q, num_heads, head_dims))/50000

    cache_len = b * seq_len * num_heads * head_dims
    kcache = torch.arange(cache_len, dtype=torch.float16, device='cuda').reshape((b, seq_len, num_heads, head_dims))/40000
    vcache = torch.arange(cache_len, dtype=torch.float16, device='cuda').reshape((b, seq_len, num_heads, head_dims))/50000

    outputs(kcache, "flash_attn_kvcache_inputs_kcache.data")
    outputs(vcache, "flash_attn_kvcache_inputs_vcache.data")
    cache_seqlens = torch.ones((b,), dtype=torch.int32, device='cuda')*64
    window_size = (-1, -1)
    cos = None
    sin = None
    cache_batch_idx = None
    out = flash_attn_with_kvcache(
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

    print("generating flash_attn kvcache inputs...")
    outputs(q, "flash_attn_kvcache_inputs_q.data")
    outputs(k, "flash_attn_kvcache_inputs_k.data")
    outputs(v, "flash_attn_kvcache_inputs_v.data")
    outputs(cache_seqlens, "flash_attn_kvcache_inputs_cache_seqlens.data")

    print("generating ground truth for flash_attn kvcache output...")
    outputs(out, "flash_attn_kvcache_outputs.data")
    outputs(kcache, "flash_attn_kvcache_outputs_kcache.data")
    outputs(vcache, "flash_attn_kvcache_outputs_vcache.data")

if __name__ == "__main__":
    generate_flash_attn_fwd_backward_data()
    generate_flash_attn_kvcache_data()
