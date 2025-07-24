import triton
import triton.language as tl

def gen_grid_layernorm(M:int,BLOCK_SIZE_M:int):
    grid = (triton.cdiv(M, BLOCK_SIZE_M),1,1)
    return grid

def gen_smem_size_layernorm(BLOCK_SIZE_M:int, BLOCK_SIZE_N:int,num_stages:int):
    return (BLOCK_SIZE_N*BLOCK_SIZE_M)*num_stages

@triton.jit
def layernorm(x_ptr,y_ptr,M:tl.constexpr,N:tl.constexpr,stride_x0:tl.constexpr,stride_x1:tl.constexpr,stride_y0:tl.constexpr,stride_y1:tl.constexpr,BLOCK_SIZE_M:tl.constexpr,BLOCK_SIZE_N:tl.constexpr,eps:tl.constexpr=1e-5):
    '''
    layernorm function for last dimension.
    '''
    NUM_BLOCK_N:tl.constexpr = (N+BLOCK_SIZE_N-1)//BLOCK_SIZE_N

    pid = tl.program_id(axis=0)
    block_start = pid*BLOCK_SIZE_M
    offsets_M = tl.arange(0,BLOCK_SIZE_M) + block_start
    mask_M = offsets_M < M
    ex = tl.zeros((BLOCK_SIZE_M, ), dtype=tl.float32)
    varx = tl.zeros((BLOCK_SIZE_M, ), dtype=tl.float32)

    for i in tl.static_range(NUM_BLOCK_N):
        offsets_N = tl.arange(0, BLOCK_SIZE_N) + i*BLOCK_SIZE_N
        mask_N = offsets_N < N
        x_offsets = offsets_M[:, None] * stride_x0 + offsets_N[None, :]* stride_x1
        mask = mask_M[:, None] & mask_N[None, :]
        x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
        ex += tl.sum(x, axis=1)
        varx += tl.sum(x * x, axis=1)

    ex = ex / N
    varx = varx / N
    varx -= (ex * ex)
    normized = tl.sqrt(varx + eps)
    for i in tl.static_range(NUM_BLOCK_N):
        offsets_N = tl.arange(0, BLOCK_SIZE_N) + i*BLOCK_SIZE_N
        mask_N = offsets_N < N
        x_offsets = offsets_M[:, None] * stride_x0 + offsets_N[None, :]* stride_x1
        y_offsets = offsets_M[:, None] * stride_y0 + offsets_N[None, :]* stride_y1
        mask = mask_M[:, None] & mask_N[None, :]
        x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
        y = (x - ex[:, None]) / normized[:, None]
        tl.store(y_ptr + y_offsets, y, mask=mask)
    

@triton.jit
def layernorm_weight_bias(
    x_ptr,
    weight_ptr,
    bias_ptr,
    y_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    stride_x0: tl.constexpr,
    stride_x1: tl.constexpr,
    stride_y0: tl.constexpr,
    stride_y1: tl.constexpr,
    stride_weight: tl.constexpr,
    stride_bias: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    eps: tl.constexpr = 1e-5,
):
    '''
    layernorm function with weight and bias for last dimension.
    '''
    NUM_BLOCK_N:tl.constexpr = (N+BLOCK_SIZE_N-1)//BLOCK_SIZE_N
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE_M
    offsets_M = tl.arange(0, BLOCK_SIZE_M) + block_start
    mask_M = offsets_M < M
    ex = tl.zeros((BLOCK_SIZE_M, ), dtype=tl.float32)
    varx = tl.zeros((BLOCK_SIZE_M, ), dtype=tl.float32)

    for i in tl.static_range(NUM_BLOCK_N):
        offsets_N = tl.arange(0, BLOCK_SIZE_N) + i*BLOCK_SIZE_N
        mask_N = offsets_N < N
        x_offsets = offsets_M[:, None] * stride_x0 + offsets_N[None, :] * stride_x1
        mask = mask_M[:, None] & mask_N[None, :]
        x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0)
        ex += tl.sum(x, axis=1) 
        varx += tl.sum(x * x, axis=1) 

    ex = ex / N
    varx = varx / N
    varx -= (ex * ex)
    normized = tl.sqrt(varx + eps)
    for i in tl.static_range(NUM_BLOCK_N):
        offsets_N = tl.arange(0, BLOCK_SIZE_N) + i*BLOCK_SIZE_N
        mask_N = offsets_N < N
        x_offsets = offsets_M[:, None] * stride_x0 + offsets_N[None, :] * stride_x1
        y_offsets = offsets_M[:, None] * stride_y0 + offsets_N[None, :] * stride_y1
        mask = mask_M[:, None] & mask_N[None, :]
        x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0)
        weight = tl.load(weight_ptr + offsets_N*stride_weight, mask=mask_N, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + offsets_N*stride_bias, mask=mask_N, other=0.0).to(tl.float32)
        y = (x - ex[:, None]) * weight[None, :] / normized[:, None] + bias[None, :]
        tl.store(y_ptr + y_offsets, y, mask=mask)
