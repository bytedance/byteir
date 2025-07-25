import triton
import triton.language as tl
from tritontemplate.backend.cuda.utils.activation import *

def gen_grid_gemm(M:int, N:int, BLOCK_SIZE_M:int, BLOCK_SIZE_N:int):
    """
    Generates the grid for a GEMM kernel.
    """
    return (
        triton.cdiv(M, BLOCK_SIZE_M)*triton.cdiv(N, BLOCK_SIZE_N),1,1)

def gen_smem_size_gemm(BLOCK_SIZE_M:int, BLOCK_SIZE_K:int, BLOCK_SIZE_N:int,num_stages:int,size_dtype:int):
    # not exactly predict, just an upper bound
    return max((BLOCK_SIZE_N*BLOCK_SIZE_K+BLOCK_SIZE_K*BLOCK_SIZE_M)*num_stages*size_dtype,(BLOCK_SIZE_M*BLOCK_SIZE_N)*4)

@triton.jit
def gemm_bias(
    # Pointers to matrices
    a_ptr, b_ptr, bias_ptr, c_ptr, 
    # Matrix dimensions
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    # Strides for matrices
    is_transpose_a: tl.constexpr, 
    stride_a0: tl.constexpr, stride_a1: tl.constexpr,
    is_transpose_b: tl.constexpr,
    stride_b0: tl.constexpr, stride_b1: tl.constexpr, 
    stride_c0: tl.constexpr, stride_c1: tl.constexpr,
    stride_bias0: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    ACTIVATION: tl.constexpr, # 'relu' or None
    enable_tf32: tl.constexpr,
):
    """
    Kernel for GEMM + Bias + ReLU.
    A(M,K) @ B(K,N) + Bias -> C(M,N)
    """
    # _TF32_ASM: tl.constexpr = "cvt.rna.tf32.f32 $0, $1;"
    # DTYPE: tl.constexpr = k.dtype.element_ty
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    if is_transpose_a:
        # A(K,M)
        a_ptrs = a_ptr + offs_k[None,:] * stride_a0 + offs_am[:,None] * stride_a1
        stride_ak = stride_a0
    else:
        # A(M,K)
        a_ptrs = a_ptr + offs_am[:, None] * stride_a0 + offs_k[None, :] * stride_a1
        stride_ak = stride_a1
    
    if is_transpose_b:
        # B(N,K)
        b_ptrs = b_ptr + offs_bn[None,:]* stride_b0 + offs_k[:,None] * stride_b1
        stride_bk = stride_b1
    else:
        # B(K,N)
        b_ptrs = b_ptr + offs_k[:, None] * stride_b0 + offs_bn[None, :] * stride_b1
        stride_bk = stride_b0
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)


    num_k_block:tl.constexpr = (K+BLOCK_SIZE_K-1)//BLOCK_SIZE_K

    for k in range(0, num_k_block):
        if is_transpose_a:
            a_mask = (offs_k[None, :] + BLOCK_SIZE_K * k < K) & (offs_am[:, None] < M)
        else:
            a_mask = (offs_am[:, None] < M) & (offs_k[None, :] + BLOCK_SIZE_K * k < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        if is_transpose_b:
            b_mask = (offs_bn[None,:] < N) & (offs_k[:,None] + BLOCK_SIZE_K * k < K)
        else:
            b_mask = (offs_k[:, None] + BLOCK_SIZE_K * k < K) & (offs_bn[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        accumulator += tl.dot(a, b, allow_tf32=enable_tf32)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk 

    if bias_ptr is not None:
        offs_bias_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
        bias_ptrs = bias_ptr + offs_bias_n * stride_bias0
        bias_vals = tl.load(bias_ptrs, mask=offs_bias_n < N, other=0.0)
        accumulator = accumulator + bias_vals[None, :] 

    if ACTIVATION == 'relu':
        accumulator = relu(accumulator)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_c0 * offs_cm[:, None] + stride_c1 * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


@triton.jit
def gemm(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    # Strides for matrices
    is_transpose_a: tl.constexpr, stride_a0: tl.constexpr, stride_a1: tl.constexpr,
    is_transpose_b: tl.constexpr, stride_b0: tl.constexpr, stride_b1: tl.constexpr, 
    stride_c0: tl.constexpr, stride_c1: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    ACTIVATION: tl.constexpr, # 'relu' or None
    enable_tf32: tl.constexpr,
):
    """
    Kernel for GEMM + ReLU.
    A @ B -> C
    This kernel can handle transpositions for A and B.
    """
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    num_k_block:tl.constexpr = (K+BLOCK_SIZE_K-1)//BLOCK_SIZE_K

    if is_transpose_a:
        # A(K,M)
        a_ptrs = a_ptr + offs_k[None,:] * stride_a0 + offs_am[:,None] * stride_a1
        stride_ak = stride_a0
    else:
        # A(M,K)
        a_ptrs = a_ptr + offs_am[:, None] * stride_a0 + offs_k[None, :] * stride_a1
        stride_ak = stride_a1
    
    if is_transpose_b:
        # B(N,K)
        b_ptrs = b_ptr + offs_bn[None,:] * stride_b0 + offs_k[:,None] * stride_b1
        stride_bk = stride_b1
    else:
        # B(K,N)
        b_ptrs = b_ptr + offs_k[:, None] * stride_b0 + offs_bn[None, :] * stride_b1
        stride_bk = stride_b0
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, num_k_block):
        if is_transpose_a:
            a_mask = (offs_k[None,:] + BLOCK_SIZE_K * k < K) & (offs_am[:,None] < M)
        else:
            a_mask = (offs_am[:, None] < M) & (offs_k[None, :] + BLOCK_SIZE_K * k < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        if is_transpose_b:
            b_mask = (offs_bn[None,:] < N) & (offs_k[:,None] + BLOCK_SIZE_K * k < K)
        else:
            b_mask = (offs_k[:, None] + BLOCK_SIZE_K * k < K) & (offs_bn[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        accumulator += tl.dot(a, b, allow_tf32=enable_tf32)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk 

    if ACTIVATION == 'relu':
        accumulator = relu(accumulator)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_c0 * offs_cm[:, None] + stride_c1 * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)