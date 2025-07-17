import torch
import pytest
import triton

from tritontemplate.backend.cuda.softmax import softmax as kernel_softmax
from tritontemplate.backend.cuda.softmax import online_softmax as kernel_online_softmax
from tritontemplate.compiler.base import IntImm, Tensor
from tritontemplate.compiler.ops.softmax import Softmax
from tritontemplate.compiler.compiler import compile_kernel

def gen_softmax(is_online,M,N,stype):
    A=Tensor(name='A',dtype=stype,shape=[M,N])
    B=Tensor(name='B',dtype=stype,shape=[M,N])
    softmax_op=Softmax(
        inputs=[A],
        dim=1,
        enable_online=is_online,
        outputs=[B],
    )
    kernel = compile_kernel(softmax_op,device='cuda')
    return kernel

FORMATS = [
    'softmax',
    'online_softmax',
]
MATRIX_PARAMS = [
    (128, 31, 'float32'),
    (2, 31, 'float16'),
    (128, 31, 'float32'),
    (31, 2, 'float16'),
    (128, 128, 'float32'),
    (257, 512, 'float16'),
    (512, 257, 'float32'),
    (256, 256, 'float16'),
    (511, 512, 'float32'),
    (128, 255, 'float16'),
    (256, 256, 'float32'),
]

@pytest.mark.parametrize('M, N, stype', MATRIX_PARAMS)
@pytest.mark.parametrize('format', FORMATS)
def test_softmax(M, N, stype, format):
    if stype == 'float32':
        torch.backends.cuda.matmul.allow_tf32=False
        dtype=torch.float32
    else:
        dtype=torch.float16
    a = torch.randn(M, N, dtype=dtype, device='cuda')
    b_triton_jit = torch.empty_like(a)
    b_triton_aot = torch.empty_like(a)
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']),   
    )

    if format == 'online_softmax':
        kernel_online_softmax[grid](a,b_triton_jit,M,N,*a.stride(),*b_triton_jit.stride(),64,64)
        kernel = gen_softmax(True,M,N,stype)
        kernel(a,b_triton_aot)
    else:
        kernel_softmax[grid](a,b_triton_jit,M,N,*a.stride(),*b_triton_jit.stride(),64,64)
        kernel = gen_softmax(False,M,N,stype)
        kernel(a,b_triton_aot)
    
    b_torch = torch.softmax(a, dim=-1)
    torch.testing.assert_close(b_triton_jit, b_torch)
    torch.testing.assert_close(b_triton_aot, b_torch)

