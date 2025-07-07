import torch
import pytest
import triton

from tritontemplate.backend.cuda.gemm.gemm_rrr import gemm_rrr_bias as gemm_rrr_bias_kernel
from tritontemplate.compiler.base import IntImm, Tensor
from tritontemplate.compiler.ops.gemm import Gemm
from tritontemplate.compiler.compiler import compile_kernel

def gen_gemm_rrr_bias_relu(M, N, K, stype):
    """
    Generates an AOT (Ahead-of-Time) compiled kernel for GEMM RRR + Bias + ReLU.
    """
    A = Tensor(name='A', dtype=stype, shape=[IntImm(M), IntImm(K)])
    
    B = Tensor(name='B', dtype=stype, shape=[IntImm(K), IntImm(N)])
    Bias = Tensor(name='Bias', dtype=stype, shape=[IntImm(N)])
    C = Tensor(name='C', dtype=stype, shape=[IntImm(M), IntImm(N)])

    gemm_op = Gemm(
        inputs=[A, B, Bias],
        outputs=[C],
        layout='rrr',
        is_bias=True,
        activation='relu',
    )

    kernel = compile_kernel(gemm_op, device='cuda')
    return kernel

def gemm_rrr_bias_relu_aot(a, b, bias, stype):
    """
    Wrapper function to execute the AOT compiled kernel.
    """
    M, K = a.shape
    K_b, N = b.shape
    assert K == K_b, "K dimension mismatch between A and B"
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    a = a.contiguous()
    b = b.contiguous()
    bias = bias.contiguous()
    c = c.contiguous()
    kernel = gen_gemm_rrr_bias_relu(M, N, K, stype)
    kernel(a, b, bias, c)
    return c

@pytest.mark.parametrize(
    'M, N, K, stype',
    [
        (2, 128, 31, 'float32'),
        (128, 2, 31, 'float16'),
        (128, 128, 31, 'float32'),
        (31, 128, 2, 'float16'),
        (129, 128, 128, 'float32'),
        (128, 257, 512, 'float16'),
        (128, 512, 257, 'float32'),
        (127, 256, 256, 'float16'),
        (128, 511, 512, 'float32'),
        (256, 128, 255, 'float16'),
        (1, 256, 256, 'float32'),
    ],
)
def test_gemm_rrr_bias_relu(M, N, K, stype):
    """
    Tests the RRR GEMM kernel against a reference PyTorch implementation and an AOT compiled version.
    """
    if stype == 'float32':
        torch.backends.cuda.matmul.allow_tf32 = False
        dtype = torch.float32
    else:
        dtype = torch.float16

    
    A = torch.randn((M, K), dtype=dtype, device='cuda')
    B = torch.randn((K, N), dtype=dtype, device='cuda')
    Bias = torch.randn((N,), dtype=dtype, device='cuda')

    
    triton_aot_result = gemm_rrr_bias_relu_aot(A, B, Bias, stype)

    
    c_triton_jit = torch.empty((M, N), device=A.device, dtype=A.dtype)
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    
    gemm_rrr_bias_kernel[grid](
        A, B, Bias, c_triton_jit,
        M, N, K,
        A.stride(0), A.stride(1),  
        B.stride(0), B.stride(1),  
        c_triton_jit.stride(0), c_triton_jit.stride(1),  
        Bias.stride(0),     
        BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=64,
        ACTIVATION='relu', enable_tf32=False
    )
    
    
    
    pytorch_result = torch.nn.functional.relu(A @ B + Bias)

    assert torch.allclose(c_triton_jit, triton_aot_result, atol=1e-2, rtol=1e-2), \
        f"Outputs mismatch between AOT and JIT for M={M}, N={N}, K={K}\n"
    assert torch.allclose(pytorch_result, triton_aot_result, atol=1e-2, rtol=1e-2), \
        f"Outputs mismatch between AOT and PyTorch for M={M}, N={N}, K={K}\n"

