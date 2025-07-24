import triton  
import triton.language as tl  
import torch  

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,  # Pointers to matrices
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,  # Matrix dimensions
    stride_am: tl.constexpr, stride_ak: tl.constexpr,  # A matrix strides
    stride_bk: tl.constexpr, stride_bn: tl.constexpr,  # B matrix strides
    stride_cm: tl.constexpr, stride_cn: tl.constexpr,  # C matrix strides
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  # Tile sizes
):
    # Get program IDs
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    
    # Create offsets for the block
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    rk = tl.arange(0, BLOCK_SIZE_K)
    
    # Create masks to avoid out-of-bounds accesses
    a_mask = (rm[:, None] < M) & (rk[None, :] < K)
    b_mask = (rk[:, None] < K) & (rn[None, :] < N)
    c_mask = (rm[:, None] < M) & (rn[None, :] < N)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, K, BLOCK_SIZE_K):
        # Load tiles from A and B
        a = tl.load(a_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak, mask=a_mask)
        b = tl.load(b_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn, mask=b_mask)
        
        # Compute matrix multiplication
        acc += tl.dot(a, b)
    
    # Store result
    tl.store(c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn, acc, mask=c_mask)

def compile_matmul_kernel_to_ptx(filename="matmul_kernel.ptx"):
    # Define the signature of the kernel
    signature = {
        0: '*fp32',  # a_ptr
        1: '*fp32',  # b_ptr
        2: '*fp32',  # c_ptr
    }
    
    # Define compile-time constants
    constants = {
        'M': 1024, 'N': 1024, 'K': 1024,  # Example matrix dimensions
        'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32,  # Tile sizes
        'stride_am': 1024, 'stride_ak': 1,  # Assuming row-major layout
        'stride_bk': 1024, 'stride_bn': 1,  # Assuming row-major layout
        'stride_cm': 1024, 'stride_cn': 1,  # Assuming row-major layout
    }
    # AOT compile the kernel
    compiled_kernel = triton.compile(
        matmul_kernel,
        signature=signature,
        constants=constants,
        num_warps=16, 
    )
    # Get the PTX assembly code
    ptx_code = compiled_kernel.asm['ptx']
    print(f"Number of warps: {compiled_kernel.num_warps}")


    # Save the PTX code to a file
    with open(filename, "w") as f:
        f.write(ptx_code)

if __name__ == "__main__":
    # AOT compile to PTX
    compile_matmul_kernel_to_ptx("matmul_kernel.ptx")