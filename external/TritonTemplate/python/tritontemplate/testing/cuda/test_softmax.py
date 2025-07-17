import torch
import pytest
import triton

from tritontemplate.backend.cuda.softmax import softmax as kernel_softmax
from tritontemplate.backend.cuda.softmax import online_softmax as kernel_online_softmax
from tritontemplate.compiler.base import IntImm, Tensor
from tritontemplate.compiler.ops.softmax import Softmax
from tritontemplate.compiler.compiler import compile_kernel

def gen_softmax(is_online,batch,num_heads,seqlen,hidden_dim,stype):
    A=Tensor(name='A',dtype=stype,shape=[batch,num_heads,seqlen,hidden_dim])
    B=Tensor(name='B',dtype=stype,shape=[batch,num_heads,seqlen,hidden_dim])
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
hidden_dim = [64, 128,]
num_heads = [8, 16,]
batch = [2, 4, 8]
seqlen = [63,66,127,129,255,257]
stype = ['float16', 'float32']

# Generate 10 random combinations
import random
test_cases = []
for _ in range(10):
    test_cases.append((
        random.choice(hidden_dim),
        random.choice(num_heads),
        random.choice(batch),
        random.choice(seqlen),
        random.choice(stype)
    ))

@pytest.mark.parametrize('hidden_dim, num_heads, batch, seqlen, stype', test_cases)
@pytest.mark.parametrize('format', FORMATS)
def test_softmax(batch,num_heads,seqlen,hidden_dim, stype, format):
    if stype == 'float32':
        torch.backends.cuda.matmul.allow_tf32=False
        dtype=torch.float32
    else:
        dtype=torch.float16
    a = torch.randn(batch,num_heads, seqlen, hidden_dim, dtype=dtype, device='cuda')
    M=batch*seqlen*num_heads
    N=hidden_dim
    b_triton_jit = torch.empty_like(a)
    b_triton_aot = torch.empty_like(a)
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']),   
    )

    if format == 'online_softmax':
        kernel_online_softmax[grid](a,b_triton_jit,M,N,N,1,N,1,64,64)
        kernel = gen_softmax(True,batch,num_heads,seqlen,hidden_dim, stype)
        kernel(a,b_triton_aot)
    else:
        kernel_softmax[grid](a,b_triton_jit,M,N,N,1,N,1,64,64)
        kernel = gen_softmax(False,batch,num_heads,seqlen,hidden_dim,stype)
        kernel(a,b_triton_aot)
    
    b_torch = torch.softmax(a, dim=-1)
    torch.testing.assert_close(b_triton_jit, b_torch)
    torch.testing.assert_close(b_triton_aot, b_torch)

test_softmax(12,8,144,128,'float32','online_softmax')