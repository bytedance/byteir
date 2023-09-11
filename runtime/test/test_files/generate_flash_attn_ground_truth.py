import torch
import flash_attn
from flash_attn.flash_attn_interface import _flash_attn_forward, _flash_attn_backward, flash_attn_func

assert flash_attn.__version__ == "2.0.9"

b = 1
seq_len = 128
num_heads = 3
head_dims = 32
causal = True


input_len = b * seq_len * num_heads * head_dims
q = torch.arange(input_len, dtype=torch.float16, device='cuda').reshape((b, seq_len, num_heads, head_dims))/30000
k = torch.arange(input_len, dtype=torch.float16, device='cuda').reshape((b, seq_len, num_heads, head_dims))/40000
v = torch.arange(input_len, dtype=torch.float16, device='cuda').reshape((b, seq_len, num_heads, head_dims))/50000


out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state = _flash_attn_forward(q, k, v, 0.0, 0.5, causal, False)


def outputs(tensor, filename):
    to_write = torch.flatten(tensor)
    of = open(filename, "w")
    for num in to_write:
        of.write(f"{num} ")
    of.close()

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
    dq, dk, dv, 0, 0.5, causal
)

print("generating ground truth for flash_attn bwd output...")
outputs(dq, "flash_attn_bwd_outputs_dq.data")
outputs(dk, "flash_attn_bwd_outputs_dk.data")
outputs(dv, "flash_attn_bwd_outputs_dv.data")
