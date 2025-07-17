import triton
import triton.language as tl

def gen_grid_transpose_10(M, N, BLOCK_SIZE_M, BLOCK_SIZE_N):
    """
    Generates the grid for a GEMM kernel.
    """
    return (
        triton.cdiv(M, BLOCK_SIZE_M),
        triton.cdiv(N, BLOCK_SIZE_N),
        1)

@triton.jit
def transpose_10(
    x_ptr, y_ptr, 
    M: tl.constexpr, N: tl.constexpr,
    stride_x0:tl.constexpr,stride_x1:tl.constexpr,
    stride_y0:tl.constexpr,stride_y1:tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    block_start_m = pid_m * BLOCK_SIZE_M
    block_start_n = pid_n * BLOCK_SIZE_N
    offset_M = tl.arange(0, BLOCK_SIZE_M) + block_start_m
    offset_N = tl.arange(0, BLOCK_SIZE_N) + block_start_n

    input_offsets = (offset_M[:, None] * stride_x0 + offset_N[None, :]) * stride_x1
    mask_M = offset_M < M
    mask_N = offset_N < N
    mask = mask_M[:, None] & mask_N[None, :]

    x = tl.load(x_ptr + input_offsets, mask=mask)
    output_offsets = (offset_N[:, None] * stride_y0 + offset_M[None, :]) * stride_y1
    output_mask = mask_M[None, :] & mask_N[:, None]
    x = tl.trans(x)
    tl.store(y_ptr + output_offsets, x, mask=output_mask)