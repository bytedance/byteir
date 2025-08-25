import triton
import triton.language as tl

def gen_grid_transpose_0213(D0:int, D1:int, D2:int, D3:int, BLOCK_SIZE_D1:int, BLOCK_SIZE_D2:int):
    """
    Generates the grid for a transpose kernel.
    """
    return (D0 * D3, triton.cdiv(D2, BLOCK_SIZE_D2), triton.cdiv(D1, BLOCK_SIZE_D1))

def gen_smem_size_transpose_0213(BLOCK_SIZE_D1:int, BLOCK_SIZE_D2:int,num_stages:int,size_dtype:int):
    return (BLOCK_SIZE_D1*BLOCK_SIZE_D2)*num_stages*size_dtype

#TODO: rewrite for contiguous D3 reading and storing
@triton.jit
def transpose_0213(x, y, 
                   D0:tl.constexpr, D1:tl.constexpr, D2:tl.constexpr, D3:tl.constexpr,
                   stride_x0:tl.constexpr, stride_x1:tl.constexpr, stride_x2:tl.constexpr, stride_x3:tl.constexpr,
                   stride_y0:tl.constexpr, stride_y1:tl.constexpr, stride_y2:tl.constexpr, stride_y3:tl.constexpr,
                   BLOCK_SIZE_D1:tl.constexpr,BLOCK_SIZE_D2:tl.constexpr):
    """
    Transpose a matrix in the 0213 layout.
    """
    pid_d0d3 = tl.program_id(0)
    pid_d2_chunk = tl.program_id(1)
    pid_d1_chunk = tl.program_id(2)

    pid_d0 = pid_d0d3 // D3
    pid_d3 = pid_d0d3 % D3

    offs_d1 = pid_d1_chunk * BLOCK_SIZE_D1 + tl.arange(0, BLOCK_SIZE_D1)
    offs_d2 = pid_d2_chunk * BLOCK_SIZE_D2 + tl.arange(0, BLOCK_SIZE_D2)

    x_ptr = x + pid_d0 * stride_x0 + \
            offs_d1[:, None] * stride_x1 + \
            offs_d2[None, :] * stride_x2 + \
            pid_d3 * stride_x3
    
    y_ptr = y + pid_d0 * stride_y0 + \
            offs_d2[:, None] * stride_y1 + \
            offs_d1[None, :] * stride_y2 + \
            pid_d3 * stride_y3


    mask_x = (offs_d1[:, None] < D1) & (offs_d2[None, :] < D2)
    mask_y = (offs_d2[:, None] < D2) & (offs_d1[None, :] < D1)

    tile = tl.load(x_ptr, mask=mask_x, other=0.0)
    transposed_tile = tl.trans(tile)
    tl.store(y_ptr, transposed_tile, mask=mask_y)