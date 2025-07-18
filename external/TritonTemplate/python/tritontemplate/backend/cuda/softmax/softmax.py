import triton
import triton.language as tl

def gen_grid_softmax(M, BLOCK_SIZE_M):
    """
    Generates the grid for a softmax kernel.
    """
    return (
        triton.cdiv(M, BLOCK_SIZE_M),1,1)

@triton.jit
def softmax(x_ptr, y_ptr, M: tl.constexpr, N: tl.constexpr,stride_x0:tl.constexpr,stride_x1:tl.constexpr, stride_y0:tl.constexpr, stride_y1:tl.constexpr, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    '''
    softmax function for last dimension.
    '''
    NUM_BLOCK_N:tl.constexpr = (N+BLOCK_SIZE_N-1)//BLOCK_SIZE_N
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE_M
    offsets_M = tl.arange(0, BLOCK_SIZE_M) + block_start
    
    mask_M = offsets_M < M

    max_ele = tl.zeros((BLOCK_SIZE_M, ), dtype=tl.float32)
    
    for i in  tl.static_range(NUM_BLOCK_N):
        offsets_N = tl.arange(0, BLOCK_SIZE_N) + i*BLOCK_SIZE_N
        mask_N = offsets_N < N
        elem = tl.load(x_ptr + offsets_M[:, None] * stride_x0 + offsets_N[None, :] * stride_x1, mask=mask_M[:, None] & mask_N[None, :], other=-float('inf'))
        max_ele = tl.maximum(max_ele,tl.max(elem, axis=1))

    y_sum = tl.zeros((BLOCK_SIZE_M, ), dtype=tl.float32)

    for i in tl.static_range(NUM_BLOCK_N):
        offsets_N = tl.arange(0, BLOCK_SIZE_N) + i*BLOCK_SIZE_N
        mask_N = offsets_N < N
        elem = tl.load(x_ptr + offsets_M[:, None] * stride_x0 + offsets_N[None, :] * stride_x1, mask=mask_M[:, None] & mask_N[None, :], other=-float('inf'))
        y = tl.exp(elem - max_ele[:, None])
        y_sum += tl.sum(y, axis=1)
        tl.store(y_ptr + offsets_M[:, None] * stride_y0 + offsets_N[None, :] * stride_y1, y,mask=mask_M[:, None] & mask_N[None, :])


    for i in tl.static_range(NUM_BLOCK_N):
        offsets_N = tl.arange(0, BLOCK_SIZE_N) + i*BLOCK_SIZE_N
        mask_N = offsets_N < N
        y = tl.load(y_ptr + offsets_M[:, None] * stride_y0 + offsets_N[None, :] * stride_y1, mask=mask_M[:, None] & mask_N[None, :], other=0)
        y = y / y_sum[:,None]
        tl.store(y_ptr + offsets_M[:, None] * stride_y0 + offsets_N[None, :] * stride_y1, y, mask=mask_M[:, None] & mask_N[None, :])


@triton.jit
def online_softmax(x_ptr, y_ptr, M: tl.constexpr, N: tl.constexpr,stride_x0:tl.constexpr,stride_x1:tl.constexpr, stride_y0:tl.constexpr, stride_y1:tl.constexpr, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    '''
    online softmax function for last dimension.
    '''
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE_M
    offsets_M = tl.arange(0, BLOCK_SIZE_M) + block_start
    mask_M = offsets_M < M

    m_i = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32) - float('inf')
    s_i = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    
    NUM_BLOCK_N:tl.constexpr = (N+BLOCK_SIZE_N-1)//BLOCK_SIZE_N

    for i in tl.static_range(NUM_BLOCK_N):
        offsets_N = tl.arange(0, BLOCK_SIZE_N) + i * BLOCK_SIZE_N
        mask_N = offsets_N < N
        
        x = tl.load(x_ptr + offsets_M[:, None] * stride_x0 + offsets_N[None, :] * stride_x1, 
                    mask=mask_M[:, None] & mask_N[None, :], other=-float('inf'))
        
        m_block = tl.max(x, axis=1)
        m_new = tl.maximum(m_i, m_block)
        
        s_i *= tl.exp(m_i - m_new)
        s_i += tl.sum(tl.exp(x - m_new[:, None]), axis=1)
        
        m_i = m_new

    for i in tl.static_range(NUM_BLOCK_N):
        offsets_N = tl.arange(0, BLOCK_SIZE_N) + i * BLOCK_SIZE_N
        mask_N = offsets_N < N
        
        x = tl.load(x_ptr + offsets_M[:, None] * stride_x0 + offsets_N[None, :] * stride_x1, 
                    mask=mask_M[:, None] & mask_N[None, :], other=0.)
        
        y = tl.exp(x - m_i[:, None]) / s_i[:, None]
        
        tl.store(y_ptr + offsets_M[:, None] * stride_y0 + offsets_N[None, :] * stride_y1, 
                 y, mask=mask_M[:, None] & mask_N[None, :])