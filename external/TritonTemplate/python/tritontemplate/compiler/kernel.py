from typing import Sequence
import triton

from tritontemplate.compiler.utils import get_cuda_device_max_shared_memory,get_cuda_device_name

class TritonExecutor:
    def __init__(self,triton_kernel:triton.compiler.compiler.CompiledKernel,grid_size:Sequence[int],warp_size:int=32,constants:dict=None):
        self.call_constants = constants
        self.triton_kernel = triton_kernel
        self.gridsize = grid_size
        self.blocksize = triton_kernel.num_warps * warp_size
        self.warpsize = warp_size
        self.name = triton_kernel.metadata['name']
        self.smemsize = triton_kernel.shared

        self.device_name = get_cuda_device_name()
        assert self.smemsize <= get_cuda_device_max_shared_memory(), \
            f'kernel {self.name} smem size {self.smemsize} exceeds device {self.device_name} max smem size {get_cuda_device_max_shared_memory()}'


    def __call__(self, *args, **kwds):
        return self.triton_kernel[self.gridsize](*args, **kwds)
    
    def kernel_ptx(self,func_name:str):
        ptx = self.triton_kernel.asm['ptx']
        ptx = ptx.replace(self.name, func_name)
        return ptx
    