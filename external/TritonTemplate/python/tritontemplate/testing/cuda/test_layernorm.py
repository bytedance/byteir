import torch
import pytest
import triton

from tritontemplate.backend.cuda.layernorm import layernorm as kernel_layernorm
from tritontemplate.backend.cuda.layernorm import layernorm_weight_bias as kernel_layernorm_weight_bias
from tritontemplate.compiler.base import IntImm, Tensor
from tritontemplate.compiler.ops.layernorm import Layernorm
from tritontemplate.compiler.compiler import compile_kernel

def gen_layernorm(with_weight_bias,batch,seq_len,hidden_size,stype):
    X=Tensor(name='X',shape=(batch,seq_len,hidden_size),dtype=stype)
    Y=Tensor(name='Y',shape=(batch,seq_len,hidden_size),dtype=stype)
    if with_weight_bias:
        W=Tensor(name='W',shape=(hidden_size,),dtype=stype)
        B=Tensor(name='B',shape=(hidden_size,),dtype=stype)
        op=Layernorm(
            inputs=[X,W,B],
            outputs=[Y],
            axis=2,
            eps=1e-5)
    else:
        op=Layernorm(
            inputs=[X],
            outputs=[Y],
            axis=2,
            eps=1e-5)
    return op

MATRIX_PARAMS = [
    (2, 128, 31, 'float32'),
    (128, 2, 31, 'float16'),
    (128, 128, 31, 'float32'),
    (128, 31, 32, 'float16'),
    (128, 128, 128, 'float32'),
    (128, 257, 512, 'float16'),
    (128, 512, 257, 'float32'),
    (127, 256, 256, 'float16'),
    (128, 511, 512, 'float32'),
    (256, 128, 255, 'float16'),
    (1, 256, 256, 'float32'),
]
FORMATS = ['layernorm','layernorm_weight_bias']

@pytest.mark.parametrize('batch,seq_len,hidden_size,stype',MATRIX_PARAMS)
@pytest.mark.parametrize('format',FORMATS)
def test_layernorm(batch,seq_len,hidden_size,stype,format):
    if stype == 'float32':
        torch.backends.cuda.matmul.allow_tf32 = False
        dtype = torch.float32
    else:
        dtype = torch.float16
    x = torch.randn(batch,seq_len,hidden_size,dtype=dtype,device='cuda')
    y_triton_jit = torch.empty(batch,seq_len,hidden_size,dtype=dtype,device='cuda')
    y_triton_aot = torch.empty(batch,seq_len,hidden_size,dtype=dtype,device='cuda')
    
    M= batch*seq_len
    N = hidden_size

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']),   
    )
    
    if format == 'layernorm':
        y_torch=torch.nn.functional.layer_norm(x,(N,),eps=1e-5)
        kernel_layernorm[grid](x,y_triton_jit,M,N,N,1,N,1,64,64,1e-5)
        kernel = gen_layernorm(False,batch,seq_len,hidden_size,stype)
        kernel_aot = compile_kernel(kernel)
        kernel_aot(x,y_triton_aot)

    else:
        weight = torch.randn(hidden_size,dtype=dtype,device='cuda')
        bias = torch.randn(hidden_size,dtype=dtype,device='cuda')
        y_torch = torch.nn.functional.layer_norm(x,(N,),weight,bias,eps=1e-5)
        kernel_layernorm_weight_bias[grid](x,bias,weight,y_triton_jit,M,N,N,1,N,1,1,1,64,64,1e-5)
        kernel = gen_layernorm(True,batch,seq_len,hidden_size,stype)
        kernel_aot = compile_kernel(kernel)
        kernel_aot(x,bias,weight,y_triton_aot)

    atol = 1e-2 if dtype == torch.float16 else 1e-4
    rtol = 1e-2 if dtype == torch.float16 else 1e-4
    torch.testing.assert_close(y_triton_jit,y_torch,atol=atol,rtol=rtol)
    torch.testing.assert_close(y_triton_aot,y_torch,atol=atol,rtol=rtol)

