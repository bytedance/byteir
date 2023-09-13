import torch
from ..backend import byteir_compile_fx

class FlashAttnModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v):
        return torch.ops.aten.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True, scale=0.2, dropout_p=0.0)

def test_flash_attn_unit():
    model = FlashAttnModel().to('cuda')
    model_golden = FlashAttnModel().to('cuda')
    model_golden.load_state_dict(model.state_dict())

    q = torch.rand(2, 12, 256, 128, dtype=torch.half, device="cuda:0", requires_grad=True)
    k = torch.rand(2, 12, 256, 128, dtype=torch.half, device="cuda:0", requires_grad=True)
    v = torch.rand(2, 12, 256, 128, dtype=torch.half, device="cuda:0", requires_grad=True)
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
