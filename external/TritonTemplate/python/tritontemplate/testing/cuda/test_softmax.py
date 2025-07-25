import torch
import pytest
import triton
import os

from tritontemplate.backend.cuda.softmax import softmax as kernel_softmax
from tritontemplate.backend.cuda.softmax import online_softmax as kernel_online_softmax
from tritontemplate.compiler.base import IntImm, Tensor
from tritontemplate.compiler.ops.softmax import Softmax
from tritontemplate.compiler.compiler import compile_kernel

def gen_softmax(is_online,batch,num_heads,seqlen,hidden_dim):
    A=Tensor(name='A',dtype='float32',shape=[batch,num_heads,seqlen,hidden_dim])
    B=Tensor(name='B',dtype='float32',shape=[batch,num_heads,seqlen,hidden_dim])
    softmax_op=Softmax(
        inputs=[A],
        dim=3,
        enable_online=is_online,
        outputs=[B],
    )
    kernel = compile_kernel(softmax_op,device='cuda')
    return kernel

FORMATS = [
    'softmax',
    'online_softmax',
]
if os.environ.get('GITHUB_CI_TEST'):
    MATRIX_PARAMS = [
        (64, 8, 8, 255), 
        (64, 16, 2, 66), 
    ]
else:
    MATRIX_PARAMS = [
        (128, 16, 8, 255), 
        (64, 8, 8, 255), 
        (64, 16, 2, 66), 
        (128, 16, 8, 257), 
        (128, 8, 4, 127), 
        (128, 8, 8, 63), 
        (64, 8, 4, 129), 
        (128, 8, 4, 255), 
        (64, 8, 4, 63), 
        (64, 8, 2, 255)
        ]

@pytest.mark.parametrize('hidden_dim, num_heads, batch, seqlen', MATRIX_PARAMS)
@pytest.mark.parametrize('format', FORMATS)
def test_softmax(batch,num_heads,seqlen,hidden_dim, format):

    a = torch.randn(batch,num_heads, seqlen, hidden_dim, dtype=torch.float32, device='cuda')
    M=batch*seqlen*num_heads
    N=hidden_dim
    b_triton_jit = torch.empty(batch,num_heads, seqlen, hidden_dim, dtype=torch.float32, device='cuda')
    b_triton_aot = torch.empty(batch,num_heads, seqlen, hidden_dim, dtype=torch.float32, device='cuda')
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']),   
    )

    if format == 'online_softmax':
        kernel_online_softmax[grid](a,b_triton_jit,M,N,N,1,N,1,128,128)
        kernel = gen_softmax(True,batch,num_heads,seqlen,hidden_dim)
        kernel(a,b_triton_aot)
    else:
        kernel_softmax[grid](a,b_triton_jit,M,N,N,1,N,1,64,64)
        kernel = gen_softmax(False,batch,num_heads,seqlen,hidden_dim)
        kernel(a,b_triton_aot)
    
    b_torch = torch.softmax(a, dim=-1).to(torch.float32)
    torch.testing.assert_close(b_triton_jit, b_triton_aot,atol=1e-2,rtol=1e-2)
    torch.testing.assert_close(b_triton_aot, b_torch,atol=1e-2,rtol=1e-2)

test_softmax(16,256,256,128,'online_softmax')