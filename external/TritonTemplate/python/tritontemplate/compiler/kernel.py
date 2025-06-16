from typing import Sequence
import triton


class TritonExecutor:
    def __init__(self,triton_kernel:triton.compiler.compiler.CompiledKernel,grid_size:Sequence[int],warp_size:int=32):
        self.triton_kernel = triton_kernel
        self.gridsize = grid_size
        self.blocksize = triton_kernel.num_warps * warp_size
        self.warpsize = warp_size
        self.name = triton_kernel.metadata['name']

    def __call__(self, *args, **kwds):
        return self.triton_kernel[self.gridsize](*args, **kwds)
    
    def kernel_ptx(self,func_name:str):
        ptx = self.triton_kernel.asm['ptx']
        ptx = ptx.replace(self.name, func_name)
        return ptx
    