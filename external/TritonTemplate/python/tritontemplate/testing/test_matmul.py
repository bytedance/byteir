import torch
import pytest

import triton
from tritontemplate.compiler.base import IntImm,Tensor
from tritontemplate.compiler.ops.gemm import Gemm
from tritontemplate.compiler.compiler import compile_kernel
from tritontemplate.backend.cuda.gemm.gemm_rcr import gemm_rcr_bias as gemm_rcr_bias_kernel

def gen_gemm_rcr_bias_relu(M, N, K):
    A = Tensor(name='A', dtype='float32', shape=[IntImm(M), IntImm(K)])
    B = Tensor(name='B', dtype='float32', shape=[IntImm(N), IntImm(K)])
    Bias = Tensor(name='Bias', dtype='float32', shape=[IntImm(N)])
    C = Tensor(name='C', dtype='float32', shape=[IntImm(M), IntImm(N)])
    
    gemm_op = Gemm(
        inputs=[A, B, Bias],
        outputs=[C],
        layout='rcr',
        is_bias=True,
        activation='relu',
    )

    kernel = compile_kernel(gemm_op, device='cuda')
    return kernel

def gemm_rcr_bias_relu(a, b, bias):
    M, K = a.shape
    N, K_b = b.shape 

    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    a=a.contiguous()
    b=b.contiguous()
    bias=bias.contiguous()
    c=c.contiguous()
    kernel = gen_gemm_rcr_bias_relu(M, N, K)
    kernel(a, b, bias, c)
    return c

@pytest.mark.parametrize(
    'M, N, K',
    [
        (128, 256,512),
        (128, 512, 256),
        (128, 256, 256),
        (128, 512, 512),
        (256,128,256),
        (256,256,256),
    ],
)
def test_gemm_rcr_bias_relu(M, N, K):
    torch.backends.cuda.matmul.allow_tf32=False
    A = torch.randn((M, K), dtype=torch.float32, device='cuda')
    B = torch.randn((N, K), dtype=torch.float32, device='cuda')
    Bias = torch.randn((N,), dtype=torch.float32, device='cuda')

    # Triton and PyTorch outputs
    c_triton = torch.empty((M, N), device=A.device, dtype=A.dtype)
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    triton_aot=gemm_rcr_bias_relu(A,B,Bias)
    gemm_rcr_bias_kernel[grid](A,B,Bias,c_triton, M, N, K,A.stride(0),A.stride(1),B.stride(0),B.stride(1),c_triton.stride(0),c_triton.stride(1),Bias.stride(0),64,64,64,'relu',enable_tf32=False)
    
    assert torch.allclose(c_triton, triton_aot, atol=1e-2, rtol=1e-2), \
        f"Outputs mismatch between aot and jit for M={M}, N={N}, K={K}\n"
    c=torch.nn.functional.relu(torch.nn.functional.linear(A,B,bias=Bias))
    assert torch.allclose(c, triton_aot, atol=1e-2, rtol=1e-2), \
        f"Outputs mismatch standard for M={M}, N={N}, K={K}\n"

if __name__ == '__main__':
    test_gemm_rcr_bias_relu(128, 256, 512)