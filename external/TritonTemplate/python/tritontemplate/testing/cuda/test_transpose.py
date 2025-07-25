import torch
import pytest
import triton
import os

from tritontemplate.backend.cuda.transpose import transpose_10 as kernel_transpose_10
from tritontemplate.backend.cuda.transpose import transpose_0213 as kernel_transpose_0213
from tritontemplate.backend.cuda.transpose import gen_grid_transpose_10, gen_grid_transpose_0213
from tritontemplate.compiler.base import IntImm, Tensor
from tritontemplate.compiler.ops.transpose import Transpose
from tritontemplate.compiler.compiler import compile_kernel

FORMATS = [
    'transpose_10',
    'transpose_0213',
]

if os.environ.get('GITHUB_CI_TEST'):
    MATRIX_PARAMS = [
        (256,128,'float16'),
        (255,257,'float32'),
    ]
else:
    MATRIX_PARAMS = [
        (256,128,'float16'),
        (255,257,'float32'),
        (127,129,'float16'),
        (2304,768,'float16'),
    ]

if os.environ.get('GITHUB_CI_TEST'):
    TENSOR4D_PARAMS = [
        (16, 64, 128, 32, 'float16'),
        (4, 127, 255, 63, 'float32'),
    ]
else:
    TENSOR4D_PARAMS = [
        (16, 64, 128, 32, 'float16'),
        (4, 127, 255, 63, 'float32'),
        (8, 256, 512, 64, 'float16'),
        (8, 1024,8,96, 'float16'),
    ]

def gen_transpose_10(M,N,stype):
    X = Tensor([M, N], stype)
    Y = Tensor([N, M], stype)
    op = Transpose([X], '10', outputs=None)
    return op

@pytest.mark.parametrize(
    'M, N, stype',
    MATRIX_PARAMS
)
def test_transpose10(M, N, stype):
    if stype == 'float32':
        torch.backends.cuda.matmul.allow_tf32 = False
        dtype = torch.float32
    else:
        dtype = torch.float16
    x = torch.randn(M, N, dtype=dtype, device='cuda')
    y_triton_jit = torch.empty(N, M, dtype=dtype, device='cuda')
    y_triton_aot = torch.empty(N, M, dtype=dtype, device='cuda')
    BLOCK_M=64
    BLOCK_N=64
    y=x.transpose(0,1).contiguous()
    grid = gen_grid_transpose_10(M, N, BLOCK_M,BLOCK_N)
    kernel_transpose_10[grid](x, y_triton_jit, M, N, *x.stride(),*y_triton_jit.stride(), BLOCK_M, BLOCK_N)
    
    kernel = gen_transpose_10(M,N,stype)
    kernel_aot = compile_kernel(kernel)
    kernel_aot(x, y_triton_aot)

    torch.testing.assert_close(y_triton_jit, y)
    torch.testing.assert_close(y_triton_aot, y)

def gen_transpose_0213(D0,D1,D2,D3,stype):
    X = Tensor([D0, D1, D2, D3], stype)
    Y = Tensor([D0, D2, D1, D3], stype)
    op = Transpose([X], '0213', None)
    return op

@pytest.mark.parametrize(
    'D0, D1, D2, D3, stype',
    TENSOR4D_PARAMS
)
def test_transpose0213(D0,D1,D2,D3, stype):
    if stype == 'float32':
        torch.backends.cuda.matmul.allow_tf32 = False
        dtype = torch.float32
    else:
        dtype = torch.float16
    x = torch.randn(D0,D1,D2,D3, dtype=dtype, device='cuda')
    y_triton_jit = torch.empty(D0,D2,D1,D3, dtype=dtype, device='cuda')
    y_triton_aot = torch.empty(D0,D2,D1,D3, dtype=dtype, device='cuda')
    
    BLOCK_D1=32
    BLOCK_D2=32
    
    y = x.permute(0,2,1,3).contiguous()
    grid = gen_grid_transpose_0213(D0, D1, D2, D3, BLOCK_D1, BLOCK_D2)
    kernel_transpose_0213[grid](x, y_triton_jit,
                                D0, D1, D2, D3,
                                *x.stride(), *y_triton_jit.stride(),
                                BLOCK_D1, BLOCK_D2)
    
    kernel = gen_transpose_0213(D0,D1,D2,D3,stype)
    kernel_aot = compile_kernel(kernel)
    kernel_aot(x, y_triton_aot)

    torch.testing.assert_close(y_triton_jit, y)    
    torch.testing.assert_close(y_triton_aot, y)
    
