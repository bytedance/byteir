import torch
import pytest

from tritontemplate.compiler.base import IntImm,Tensor
from tritontemplate.compiler.ops.gemm import Gemm
from tritontemplate.compiler.compiler import compile_kernel

def gen_gemm_rcr_bias_relu(M, N, K):
    A = Tensor(name='A', dtype='float16', shape=[IntImm(M), IntImm(K)])
    B = Tensor(name='B', dtype='float16', shape=[IntImm(N), IntImm(K)])
    Bias = Tensor(name='Bias', dtype='float16', shape=[IntImm(N)])
    C = Tensor(name='C', dtype='float16', shape=[IntImm(M), IntImm(N)])
    
    gemm_op = Gemm(
        inputs=[A, B, Bias],
        outputs=[C],
        layout='rcr',
        is_bias=True,
        is_transpose=False,
        activation='relu',
    )

    kernel = compile_kernel(gemm_op, target_name='cuda')
    return kernel

def gemm_rcr_bias_relu(a, b, bias):
    M, K = a.shape
    N, K_b = b.shape 
    assert K == K_b, "K dimension mismatch"

    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    kernel = gen_gemm_rcr_bias_relu(M, N, K)
    kernel(a, b, c, bias)
    return c

@pytest.mark.parametrize(
    'M, N, K',
    [
        (1024, 1024, 1024),
        (1024, 1024, 512),
        (1024, 1024, 256),
        (1023,1023,1023),
        (1025,1025,511),
    ],
)
def test_gemm_rcr_bias_relu(M, N, K):
    A = torch.randn((M, K), dtype=torch.float16, device='cuda')
    B = torch.randn((N, K), dtype=torch.float16, device='cuda')
    Bias = torch.randn((N,), dtype=torch.float16, device='cuda')

    # Triton and PyTorch outputs
    c_triton = gemm_rcr_bias_relu(A, B, Bias)
    y_torch = torch.relu(torch.nn.functional.linear(A, B, bias=Bias))

    if not torch.allclose(c_triton, y_torch, atol=1e-2, rtol=1e-2):
        print("Outputs mismatch!")
        diff = torch.abs(c_triton - y_torch)
        print("Max diff:", torch.max(diff), "Mean diff:", torch.mean(diff))



