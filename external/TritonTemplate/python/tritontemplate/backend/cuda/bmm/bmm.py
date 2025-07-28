import triton
import triton.language as tl

def gen_grid_bmm(BATCH_SIZE:int,M:int, N:int, BLOCK_SIZE_M:int, BLOCK_SIZE_N:int):
    """
    Generates the grid for a Batch GEMM kernel.
    """
    return (BATCH_SIZE,triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N), 1)

def gen_smem_size_bmm(BLOCK_SIZE_M:int, BLOCK_SIZE_K:int, BLOCK_SIZE_N:int,num_stages:int,size_dtype:int):
    return max((BLOCK_SIZE_N*BLOCK_SIZE_K+BLOCK_SIZE_K*BLOCK_SIZE_M)*num_stages*size_dtype,(BLOCK_SIZE_M*BLOCK_SIZE_N)*4)

@triton.jit
def bmm_bias(
    # Pointers to matrices
    a_ptr, b_ptr, bias_ptr, c_ptr,
    # Matrix dimensions
    BATCH_SIZE: tl.constexpr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    # Strides for matrices
    is_transpose_a: tl.constexpr,stride_a0: tl.constexpr, stride_a1: tl.constexpr, stride_a2: tl.constexpr,
    is_transpose_b: tl.constexpr,stride_b0: tl.constexpr, stride_b1: tl.constexpr, stride_b2: tl.constexpr,
    stride_bias0: tl.constexpr,stride_bias1: tl.constexpr,stride_bias2: tl.constexpr,
    stride_c0: tl.constexpr, stride_c1: tl.constexpr, stride_c2: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    enable_tf32: tl.constexpr,
    ):
    """
    Kernel for BMM + Bias.
    if !is_transpose_a and !is_transpose_b :
        A (B, M, K) @ B (B, K, N) + Bias (B, M, N) -> C (B, M, N)
    """
    batch_id=tl.program_id(0)
    pid=tl.program_id(1)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_am = pid_m*BLOCK_SIZE_M+tl.arange(0,BLOCK_SIZE_M)
    offs_bn = pid_n*BLOCK_SIZE_N+tl.arange(0,BLOCK_SIZE_N)
    offs_k = tl.arange(0,BLOCK_SIZE_K)
    
    if is_transpose_a:
        # A(B,K,M)
        a_ptrs = a_ptr+stride_a0*batch_id+stride_a1*offs_k[None,:]+stride_a2*offs_am[:,None]
        stride_ak=stride_a1
    else:
        # A(B,M,K)
        a_ptrs = a_ptr+stride_a0*batch_id+stride_a1*offs_am[:,None]+stride_a2*offs_k[None,:]
        stride_ak=stride_a2
    
    if is_transpose_b:
        # B(B,N,K)
        b_ptrs = b_ptr+stride_b0*batch_id+stride_b1*offs_bn[None,:]+stride_b2*offs_k[:,None]
        stride_bk=stride_b2
    else:
        # B(B,K,N)
        b_ptrs = b_ptr+stride_b0*batch_id+stride_b1*offs_k[:,None]+stride_b2*offs_bn[None,:]
        stride_bk=stride_b1

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        if is_transpose_a:
            a_mask= (offs_k[None,:]+BLOCK_SIZE_K*k<K) & (offs_am[:,None]<M)
        else:
            a_mask= (offs_am[:,None]<M) & (offs_k[None,:]+BLOCK_SIZE_K*k<K)
        a=tl.load(a_ptrs,mask=a_mask)
        if is_transpose_b:
            b_mask= (offs_bn[None,:]<N) & (offs_k[:,None]+BLOCK_SIZE_K*k<K)
        else:
            b_mask= (offs_k[:,None]+BLOCK_SIZE_K*k<K) & (offs_bn[None,:]<N)
        b=tl.load(b_ptrs,mask=b_mask)

        accumulator += tl.dot(a, b, allow_tf32=enable_tf32)

        a_ptrs+=BLOCK_SIZE_K*stride_ak
        b_ptrs+=BLOCK_SIZE_K*stride_bk

    offs_cm = pid_m*BLOCK_SIZE_M+ tl.arange(0,BLOCK_SIZE_M)
    offs_cn = pid_n*BLOCK_SIZE_N+ tl.arange(0,BLOCK_SIZE_N)

    c_ptrs_offs = batch_id* stride_c0+ offs_cm[:,None]*stride_c1+ offs_cn[None,:]*stride_c2
    c_mask = (offs_cm[:,None]<M) & (offs_cn[None,:]<N)
    bias=tl.load(bias_ptr+c_ptrs_offs,mask=c_mask,other=0.0)
    accumulator+=bias

    tl.store(c_ptr+c_ptrs_offs,accumulator,mask=c_mask)


@triton.jit
def bmm(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    BATCH_SIZE: tl.constexpr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    # Strides for matrices
    is_transpose_a: tl.constexpr,stride_a0: tl.constexpr, stride_a1: tl.constexpr, stride_a2: tl.constexpr,
    is_transpose_b: tl.constexpr,stride_b0: tl.constexpr, stride_b1: tl.constexpr, stride_b2: tl.constexpr,
    stride_c0: tl.constexpr, stride_c1: tl.constexpr, stride_c2: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    enable_tf32: tl.constexpr,
    ):
    """
    Kernel for BMM.
    if !is_transpose_a and !is_transpose_b :
        A (B, M, K) @ B (B, K, N) -> C (B, M, N)
    """
    batch_id=tl.program_id(0)
    pid=tl.program_id(1)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_am = pid_m*BLOCK_SIZE_M+tl.arange(0,BLOCK_SIZE_M)
    offs_bn = pid_n*BLOCK_SIZE_N+tl.arange(0,BLOCK_SIZE_N)
    offs_k = tl.arange(0,BLOCK_SIZE_K)
    
    if is_transpose_a:
        # A(B,K,M)
        a_ptrs = a_ptr+stride_a0*batch_id+stride_a1*offs_k[None,:]+stride_a2*offs_am[:,None]
        stride_ak=stride_a1
    else:
        # A(B,M,K)
        a_ptrs = a_ptr+stride_a0*batch_id+stride_a1*offs_am[:,None]+stride_a2*offs_k[None,:]
        stride_ak=stride_a2
    
    if is_transpose_b:
        # B(B,N,K)
        b_ptrs = b_ptr+stride_b0*batch_id+stride_b1*offs_bn[None,:]+stride_b2*offs_k[:,None]
        stride_bk=stride_b2
    else:
        # B(B,K,N)
        b_ptrs = b_ptr+stride_b0*batch_id+stride_b1*offs_k[:,None]+stride_b2*offs_bn[None,:]
        stride_bk=stride_b1

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        if is_transpose_a:
            a_mask= (offs_k[None,:]+BLOCK_SIZE_K*k<K) & (offs_am[:,None]<M)
        else:
            a_mask= (offs_am[:,None]<M) & (offs_k[None,:]+BLOCK_SIZE_K*k<K)
        a=tl.load(a_ptrs,mask=a_mask)
        if is_transpose_b:
            b_mask= (offs_bn[None,:]<N) & (offs_k[:,None]+BLOCK_SIZE_K*k<K)
        else:
            b_mask= (offs_k[:,None]+BLOCK_SIZE_K*k<K) & (offs_bn[None,:]<N)
        b=tl.load(b_ptrs,mask=b_mask)


        accumulator += tl.dot(a, b, allow_tf32=enable_tf32)

        a_ptrs+=BLOCK_SIZE_K*stride_ak
        b_ptrs+=BLOCK_SIZE_K*stride_bk

    offs_cm = pid_m*BLOCK_SIZE_M+ tl.arange(0,BLOCK_SIZE_M)
    offs_cn = pid_n*BLOCK_SIZE_N+ tl.arange(0,BLOCK_SIZE_N)

    c_ptrs_offs = batch_id* stride_c0+ offs_cm[:,None]*stride_c1+ offs_cn[None,:]*stride_c2
    c_mask = (offs_cm[:,None]<M) & (offs_cn[None,:]<N)

    tl.store(c_ptr+c_ptrs_offs,accumulator,mask=c_mask)