import triton
import triton.language as tl

__all__ = ['relu']

@triton.jit
def relu(x):
    return tl.maximum(x, 0)

