import torch
import pytest

import triton
from tritontemplate.compiler.base import IntImm,Tensor
from tritontemplate.compiler.ops.bmm import Bmm
from tritontemplate.compiler.compiler import compile_kernel

from tritontemplate.backend.cuda.bmm.bmm import bmm_bias as bmm_bias_kernel
from tritontemplate.backend.cuda.bmm.bmm import bmm as bmm_kernel

def gen_bmm_bias(format, batch_size, M, N, K, stype):
    if format[0]=='r':
        A=Tensor(name='A',dtype=stype,shape=[batch_size,M,K])
    else:
        A=Tensor(name='A',dtype=stype,shape=[batch_size,K,M])
    if format[1]=='r':
        B=Tensor(name='B',dtype=stype,shape=[batch_size,K,N])
    else:
        B=Tensor(name='B',dtype=stype,shape=[batch_size,N,K])
    Bias=Tensor(name='Bias',dtype=stype,shape=[batch_size,M,N])
    C=Tensor(name='C',dtype=stype,shape=[batch_size,M,N])
    bmm_op=Bmm(
        inputs=[A,B,Bias],
        outputs=[C],
        layout=format,
        is_bias=True
    )
    kernel = compile_kernel(bmm_op,device='cuda')
    return kernel

def gen_bmm(format, batch_size, M, N, K, stype):
    if format[0]=='r':
        A=Tensor(name='A',dtype=stype,shape=[batch_size,M,K])
    else:
        A=Tensor(name='A',dtype=stype,shape=[batch_size,K,M])
    if format[1]=='r':
        B=Tensor(name='B',dtype=stype,shape=[batch_size,K,N])
    else:
        B=Tensor(name='B',dtype=stype,shape=[batch_size,N,K])
    C=Tensor(name='C',dtype=stype,shape=[batch_size,M,N])
    bmm_op=Bmm(
        inputs=[A,B],
        outputs=[C],
        layout=format,
        is_bias=False
    )
    kernel = compile_kernel(bmm_op,device='cuda')
    return kernel


FORMATS = ['rcr','rrr','ccr','crr']
MATRIX_PARAMS = [
    (2, 2, 128, 31, 'float32'),
    (2, 128, 2, 31, 'float16'),
    (2, 128, 128, 31, 'float32'),
    (2, 31, 128, 2, 'float16'),
    (2, 129, 128, 128, 'float32'),
    (2, 128, 257, 512, 'float16'),
    (2, 128, 512, 257, 'float32'),
    (2, 127, 256, 256, 'float16'),
    (2, 128, 511, 512, 'float32'),
    (2, 256, 128, 255, 'float16'),
    (2, 1, 256, 256, 'float32'),
]

@pytest.mark.parametrize('format', FORMATS)
@pytest.mark.parametrize(
    'batch_size, M, N, K, stype',
    MATRIX_PARAMS
)
def test_bmm_bias(format, batch_size, M, N, K, stype):
    torch.manual_seed(0)
    if stype == 'float32':
        torch.backends.cuda.matmul.allow_tf32=False
        dtype=torch.float32
    else:
        dtype=torch.float16

    a = torch.randn(batch_size, M, K, dtype=dtype, device='cuda')
    b = torch.randn(batch_size, K, N, dtype=dtype, device='cuda')
    bias = torch.randn(batch_size,M, N, dtype=dtype, device='cuda')
    c_triton = torch.empty(batch_size, M, N, dtype=dtype, device='cuda')
    c_ttemplate = torch.empty(batch_size, M, N, dtype=dtype, device='cuda')
      
    c_torch= torch.bmm(a, b)+bias

    grid = lambda META: (
        META['BATCH_SIZE'],triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    is_trans_a=False
    is_trans_b=False
    if format[0]=='c':
        a=a.transpose(1,2).contiguous()
        is_trans_a=True
    if format[1]=='c':
        b=b.transpose(1,2).contiguous()
        is_trans_b=True
    test_kernel=gen_bmm_bias(format,batch_size,M,N,K,stype)
    test_kernel(a,b,bias,c_ttemplate)
    bmm_bias_kernel[grid](a,b,bias,c_triton,batch_size,M,N,K,is_trans_a,*a.stride(),is_trans_b,*b.stride(),*bias.stride(),*c_triton.stride(),64,64,64,False)
    print(*b.stride())
    torch.testing.assert_close(c_ttemplate,c_triton,atol=1e-2,rtol=1e-2)  
    torch.testing.assert_close(c_ttemplate,c_torch,atol=1e-2,rtol=1e-2)

@pytest.mark.parametrize('format', FORMATS)
@pytest.mark.parametrize(
    'batch_size, M, N, K, stype',
    MATRIX_PARAMS
)
def test_bmm(format, batch_size, M, N, K, stype):
    torch.manual_seed(0)
    if stype == 'float32':
        torch.backends.cuda.matmul.allow_tf32=False
        dtype=torch.float32
    else:
        dtype=torch.float16

    a = torch.randn(batch_size, M, K, dtype=dtype, device='cuda')
    b = torch.randn(batch_size, K, N, dtype=dtype, device='cuda')
    c_triton = torch.randn(batch_size, M, N, dtype=dtype, device='cuda')
    c_ttemplate = torch.randn(batch_size, M, N, dtype=dtype, device='cuda')

    c_torch= torch.bmm(a, b)
    grid = lambda META: (
        META['BATCH_SIZE'],triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    is_trans_a=False
    is_trans_b=False
    if format[0]=='c':
        a=a.transpose(1,2).contiguous()
        is_trans_a=True
    if format[1]=='c':
        b=b.transpose(1,2).contiguous()
        is_trans_b=True


    bmm_kernel[grid](a,b,c_triton,batch_size,M,N,K,is_trans_a,*a.stride(),is_trans_b,*b.stride(),*c_triton.stride(),64,64,64,False)
    kernel=gen_bmm(format,batch_size,M,N,K,stype)
    kernel(a,b,c_ttemplate)
    torch.testing.assert_close(c_ttemplate,c_triton,atol=1e-2,rtol=1e-2)
    torch.testing.assert_close(c_ttemplate,c_torch,atol=1e-2,rtol=1e-2)

