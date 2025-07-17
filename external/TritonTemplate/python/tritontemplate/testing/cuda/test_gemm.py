import torch
import pytest
import triton

from tritontemplate.backend.cuda.gemm import gemm_bias as gemm_bias_kernel
from tritontemplate.backend.cuda.gemm import gemm as gemm_kernel
from tritontemplate.compiler.base import IntImm, Tensor
from tritontemplate.compiler.ops.gemm import Gemm
from tritontemplate.compiler.compiler import compile_kernel

def gen_gemm_bias(format, M, N, K, stype):
    if format[0]=='r':
        A=Tensor(name='A',dtype=stype,shape=[M,K])
    else:
        A=Tensor(name='A',dtype=stype,shape=[K,M])
    if format[1]=='r':
        B=Tensor(name='B',dtype=stype,shape=[K,N])
    else:
        B=Tensor(name='B',dtype=stype,shape=[N,K])  
    Bias=Tensor(name='Bias',dtype=stype,shape=[M,N])
    C=Tensor(name='C',dtype=stype,shape=[M,N])
    gemm_op=Gemm(
        inputs=[A,B,Bias],
        outputs=[C],
        layout=format,
        is_bias=True, 
        activation='relu',
    )
    kernel = compile_kernel(gemm_op,device='cuda')
    return kernel

def gen_gemm(format, M, N, K, stype):
    if format[0]=='r':
        A=Tensor(name='A',dtype=stype,shape=[M,K])
    else:
        A=Tensor(name='A',dtype=stype,shape=[K,M])
    if format[1]=='r':
        B=Tensor(name='B',dtype=stype,shape=[K,N])
    else:
        B=Tensor(name='B',dtype=stype,shape=[N,K])
    C=Tensor(name='C',dtype=stype,shape=[M,N])
    gemm_op=Gemm(
        inputs=[A,B],
        outputs=[C],
        layout=format,
        is_bias=False, 
        activation='relu',
    )
    kernel = compile_kernel(gemm_op,device='cuda')
    return kernel



FORMATS = ['rcr','rrr','ccr','crr']
MATRIX_PARAMS = [
    (2, 128, 31, 'float32'),
    (128, 2, 31, 'float16'),
    (128, 128, 31, 'float32'),
    (128, 31, 2, 'float16'),
    (128, 128, 128, 'float32'),
    (128, 257, 512, 'float16'),
    (128, 512, 257, 'float32'),
    (127, 256, 256, 'float16'),
    (128, 511, 512, 'float32'),
    (256, 128, 255, 'float16'),
    (1, 256, 256, 'float32'),
]

@pytest.mark.parametrize('format', FORMATS)
@pytest.mark.parametrize(
    'M, N, K, stype',
    MATRIX_PARAMS
)
def test_gemm_bias_relu(format, M, N, K, stype):

    if stype == 'float32':
        torch.backends.cuda.matmul.allow_tf32 = False
        dtype = torch.float32
    else:
        dtype = torch.float16

    
    A = torch.randn((M, K), dtype=dtype, device='cuda')
    B = torch.randn((K, N), dtype=dtype, device='cuda')
    Bias = torch.randn((N,), dtype=dtype, device='cuda')
    c_triton_jit = torch.empty((M, N), device=A.device, dtype=A.dtype)
    c_triton_aot = torch.empty((M, N), device=A.device, dtype=A.dtype)

    pytorch_result = torch.nn.functional.relu(A @ B + Bias)

    is_trans_a=False
    is_trans_b=False

    if format[0] == 'c':
        is_trans_a = True
        A = A.transpose(1,0).contiguous()

    if format[1] == 'c':
        is_trans_b = True
        B = B.transpose(1,0).contiguous()

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    
    gemm_bias_kernel[grid](
        A, B, Bias, c_triton_jit,
        M, N, K,
        is_trans_a, *A.stride(),  
        is_trans_b, *B.stride(),  
        *c_triton_jit.stride(),  
        *Bias.stride(),     
        BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=64,
        ACTIVATION='relu', enable_tf32=False
    )
    kernel = gen_gemm_bias(format, M, N, K, stype)
    kernel(A, B, Bias, c_triton_aot)

    assert torch.allclose(c_triton_aot, c_triton_jit, atol=1e-2, rtol=1e-2)
    assert torch.allclose(pytorch_result, c_triton_jit, atol=1e-2, rtol=1e-2)

FORMATS = ['rcr','rrr','ccr','crr']
MATRIX_PARAMS = [
    (2, 128, 31, 'float32'),
    (128, 2, 31, 'float16'),
    (128, 128, 31, 'float32'),
    (128, 31, 2, 'float16'),
    (128, 128, 128, 'float32'),
    (128, 257, 512, 'float16'),
    (128, 512, 257, 'float32'),
    (127, 256, 256, 'float16'),
    (128, 511, 512, 'float32'),
    (256, 128, 255, 'float16'),
    (1, 256, 256, 'float32'),
]

@pytest.mark.parametrize('format', FORMATS)
@pytest.mark.parametrize(
    'M, N, K, stype',
    MATRIX_PARAMS
)
def test_gemm_relu(format, M, N, K, stype):

    if stype == 'float32':
        torch.backends.cuda.matmul.allow_tf32 = False
        dtype = torch.float32
    else:
        dtype = torch.float16

    A = torch.randn((M, K), dtype=dtype, device='cuda')
    B = torch.randn((K, N), dtype=dtype, device='cuda')
    c_triton_jit = torch.empty((M, N), device=A.device, dtype=A.dtype)
    c_triton_aot = torch.empty((M, N), device=A.device, dtype=A.dtype)

    pytorch_result = torch.nn.functional.relu(A @ B)

    is_trans_a=False
    is_trans_b=False

    if format[0] == 'c':
        is_trans_a = True
        A = A.transpose(1,0).contiguous()

    if format[1] == 'c':
        is_trans_b = True
        B = B.transpose(1,0).contiguous()

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    
    gemm_kernel[grid](
        A, B, c_triton_jit,
        M, N, K,
        is_trans_a, *A.stride(),  
        is_trans_b, *B.stride(),  
        *c_triton_jit.stride(),      
        BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=64,
        ACTIVATION='relu', enable_tf32=False
    )
    kernel = gen_gemm(format, M, N, K, stype)
    kernel(A, B, c_triton_aot)

    atol = 1e-2 if dtype == torch.float16 else 1e-4
    rtol = 1e-2 if dtype == torch.float16 else 1e-4
    assert torch.allclose(c_triton_aot, c_triton_jit, atol=atol, rtol=rtol)
    assert torch.allclose(pytorch_result, c_triton_jit, atol=atol, rtol=rtol)

