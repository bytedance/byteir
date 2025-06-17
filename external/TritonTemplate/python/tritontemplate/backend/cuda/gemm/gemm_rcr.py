import triton
import triton.language as tl

def gen_grid_gemm_rcr(M, N, BLOCK_SIZE_M, BLOCK_SIZE_N):
    
    return (
        triton.cdiv(M, BLOCK_SIZE_M)*triton.cdiv(N, BLOCK_SIZE_N),1,1)

# smem_size=val*dtype_size*num_stage
smem_demand_per_stage ={
    'gemm_rcr_bias': 128*128*2,
    'gemm_rcr': 128*128*2,
} 

@triton.jit
def gemm_rcr_bias(
    a_ptr, b_ptr,  bias_ptr, c_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    stride_a0: tl.constexpr, stride_a1: tl.constexpr,
    stride_b0: tl.constexpr, stride_b1: tl.constexpr, 
    stride_c0: tl.constexpr, stride_c1: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    ACTIVATION: tl.constexpr 
):
    """
    Kernel for GEMM RCR (Row-Col-Row) + Bias + ReLU.
    A (M, K) @ B (N, K)^T + Bias (N) -> C (M, N)
    B is stored as (N, K) but accessed as if it's (K, N) for the matmul.
    """

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_a0 + offs_k[None, :] * stride_a1)
    b_ptrs = b_ptr + (offs_bn[None, :] * stride_b0 + offs_k[:, None] * stride_b1) # K is outer, N is inner

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=(offs_k[None, :] + k * BLOCK_SIZE_K < K) & (offs_am[:, None] < M), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] + k * BLOCK_SIZE_K < K) & (offs_bn[None, :] < N), other=0.0)


        accumulator += tl.dot(a, b) 
        a_ptrs += BLOCK_SIZE_K * stride_a1
        b_ptrs += BLOCK_SIZE_K * stride_b1 

    if bias_ptr is not None:
        offs_bias_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
        bias_ptrs = bias_ptr + offs_bias_n 
        bias_vals = tl.load(bias_ptrs, mask=offs_bias_n < N, other=0.0)
        accumulator = accumulator + bias_vals[None, :] # Broadcast bias across M

    # Apply activation function
    if ACTIVATION == 'relu':
        accumulator = tl.maximum(accumulator, 0)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_c0 * offs_cm[:, None] + stride_c1 * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


@triton.jit
def gemm_rcr(
    a_ptr, b_ptr, c_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    stride_a0: tl.constexpr, stride_a1: tl.constexpr,
    stride_b0: tl.constexpr, stride_b1: tl.constexpr, 
    stride_c0: tl.constexpr, stride_c1: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    ACTIVATION: tl.constexpr 
):
    """
    Kernel for GEMM RCR (Row-Col-Row) (+ReLU).
    A (M, K) @ B (N, K)^T  -> C (M, N)
    B is stored as (N, K) but accessed as if it's (K, N) for the matmul.
    """

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_a0 + offs_k[None, :] * stride_a1)
    b_ptrs = b_ptr + (offs_bn[None, :] * stride_b0 + offs_k[:, None] * stride_b1) # K is outer, N is inner

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=(offs_k[None, :] + k * BLOCK_SIZE_K < K) & (offs_am[:, None] < M), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] + k * BLOCK_SIZE_K < K) & (offs_bn[None, :] < N), other=0.0)


        accumulator += tl.dot(a, b) 
        a_ptrs += BLOCK_SIZE_K * stride_a1
        b_ptrs += BLOCK_SIZE_K * stride_b1 

    # Apply activation function
    if ACTIVATION == 'relu':
        accumulator = tl.maximum(accumulator, 0)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_c0 * offs_cm[:, None] + stride_c1 * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)

