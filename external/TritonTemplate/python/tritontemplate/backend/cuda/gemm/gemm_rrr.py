import triton
import triton.language as tl
from tritontemplate.backend.cuda.utils.activation import *

def gen_grid_gemm_rrr(M, N, BLOCK_SIZE_M, BLOCK_SIZE_N):
    """
    Generates the grid for a GEMM kernel.
    """
    return (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N), 1, 1)

smem_demand_per_stage = {
    'gemm_rrr_bias': 128 * 128 * 2,
    'gemm_rrr': 128 * 128 * 2,
}

@triton.jit
def gemm_rrr_bias(
    # Pointers to matrices
    a_ptr, b_ptr, bias_ptr, c_ptr,
    # Matrix dimensions
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    # Strides for matrices
    stride_am: tl.constexpr, stride_ak: tl.constexpr,
    stride_bk: tl.constexpr, stride_bn: tl.constexpr,
    stride_cm: tl.constexpr, stride_cn: tl.constexpr,
    stride_biasn: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    ACTIVATION: tl.constexpr,  # 'relu' or None
    enable_tf32: tl.constexpr,
):
    """
    Kernel for GEMM RRR (Row-Row-Row) + Bias + ReLU.
    A (M, K) @ B (K, N) + Bias (N) -> C (M, N)
    B is stored as (N, K) but accessed as if it's (K, N) for the matmul.
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=(offs_am[:, None] < M) & (offs_k[None, :] + k * BLOCK_SIZE_K < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] + k * BLOCK_SIZE_K < K) & (offs_bn[None, :] < N), other=0.0)

        accumulator += tl.dot(a, b, allow_tf32=enable_tf32)
        
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk


    if bias_ptr is not None:
        offs_bias_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        bias_ptrs = bias_ptr + offs_bias_n * stride_biasn
        bias_vals = tl.load(bias_ptrs, mask=offs_bias_n < N, other=0.0)

        accumulator = accumulator + bias_vals[None, :]


    if ACTIVATION == 'relu':
        accumulator = relu(accumulator)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


@triton.jit
def gemm_rrr(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    # Strides for matrices
    stride_am: tl.constexpr, stride_ak: tl.constexpr,
    stride_bk: tl.constexpr, stride_bn: tl.constexpr,
    stride_cm: tl.constexpr, stride_cn: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    ACTIVATION: tl.constexpr,  # 'relu' or None
    enable_tf32: tl.constexpr,
):
    """
    Kernel for GEMM RRR (Row-Row-Row) + ReLU.
    A (M, K) @ B (K,N)  -> C (M, N)
    """

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)


    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in tl.static_range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=(offs_am[:, None] < M) & (offs_k[None, :] + k * BLOCK_SIZE_K < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] + k * BLOCK_SIZE_K < K) & (offs_bn[None, :] < N), other=0.0)
        
        accumulator += tl.dot(a, b, allow_tf32=enable_tf32)
        
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if ACTIVATION == 'relu':
        accumulator = relu(accumulator)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)
