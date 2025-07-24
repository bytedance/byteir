from typing import List,Optional
import importlib

import triton

from tritontemplate.compiler.base import IntImm, Tensor, Operation
from tritontemplate.compiler.dtype import dtype_str_to_triton_signature
from tritontemplate.compiler.kernel import TritonExecutor
from tritontemplate.compiler.utils import get_warpsize,get_cuda_device_max_shared_memory
from tritontemplate.backend.cuda.utils.utils import shape2stride

_supported_permutations = ['10','0213']

_exec_metadata = {
    'num_warps': 4,
    'num_stages': 1,
}

class Transpose(Operation):
    def __init__(self,
                inputs: List[Tensor],
                permutation: str,
                outputs: Optional[List[Tensor]] = None,
                name: Optional[str] = None):
        super().__init__(inputs, outputs, name)
        assert permutation in _supported_permutations, f"Unsupported permutation {permutation}"
        self._attrs['permutation'] = permutation

        self._deduce_output_shape()

    def _deduce_output_shape(self):
        input_shape = self._attrs['inputs'][0].shape
        output_shape = []
        for i in self._attrs['permutation']:
            output_shape.append(input_shape[int(i)])
        if self._attrs['outputs'] is None:
            self._attrs['outputs'] = [Tensor(output_shape, self._attrs['inputs'][0].dtype)]
        else:
            assert self._attrs['outputs'][0].shape == output_shape, f"Transpose op output shape {self._attrs['outputs'][0].shape} does not match expected shape {output_shape}"
    
    def _gen_constants_10(self,num_stages,func_gen_smem_size):
        const_metadata={}
        M,N=self._attrs['inputs'][0].shape

        const_metadata['M'] = M
        const_metadata['N'] = N
        const_metadata['stride_x0'] = N
        const_metadata['stride_x1'] = 1
        const_metadata['stride_y0'] = M
        const_metadata['stride_y1'] = 1

        const_metadata['BLOCK_SIZE_M'] = self._block_size(M)
        const_metadata['BLOCK_SIZE_N'] = self._block_size(N)
        self._shrink_shared_mem(func_gen_smem_size,const_metadata,get_cuda_device_max_shared_memory(),num_stages)

        return const_metadata
    
    def _gen_grid_10(self,target_name,const_metadata):
        gen_grid = getattr(importlib.import_module(f'tritontemplate.backend.{target_name}.transpose'),'gen_grid_transpose_10')
        return gen_grid(const_metadata['M'],const_metadata['N'],const_metadata['BLOCK_SIZE_M'],const_metadata['BLOCK_SIZE_N'])
    
    def _gen_constants_0213(self,num_stages,func_gen_smem_size):
        const_metadata={}
        D0,D1,D2,D3=self._attrs['inputs'][0].shape
        const_metadata['D0'] = D0
        const_metadata['D1'] = D1
        const_metadata['D2'] = D2
        const_metadata['D3'] = D3
        
        const_metadata['stride_x0'],const_metadata['stride_x1'],const_metadata['stride_x2'],const_metadata['stride_x3'] = shape2stride(self._attrs['inputs'][0].shape)
        const_metadata['stride_y0'],const_metadata['stride_y1'],const_metadata['stride_y2'],const_metadata['stride_y3']= shape2stride(self._attrs['outputs'][0].shape)

        const_metadata['BLOCK_SIZE_D1'] = self._block_size(D1)
        const_metadata['BLOCK_SIZE_D2'] = self._block_size(D2)

        self._shrink_shared_mem(func_gen_smem_size,const_metadata,get_cuda_device_max_shared_memory(),num_stages)

        return const_metadata

    def _gen_grid_0213(self,target_name,const_metadata):
        gen_grid = getattr(importlib.import_module(f'tritontemplate.backend.{target_name}.transpose'),'gen_grid_transpose_0213')
        return gen_grid(const_metadata['D0'],const_metadata['D1'],const_metadata['D2'],const_metadata['D3'],const_metadata['BLOCK_SIZE_D1'],const_metadata['BLOCK_SIZE_D2'])


    def _gen_exec_metadata(self):
        return _exec_metadata.copy()
    
    
    def compile(self, target_name, workdir, enable_tf32)->TritonExecutor:
        triton_kernel_name= 'transpose_' + self._attrs['permutation']
        triton_kernel=getattr(importlib.import_module(f'tritontemplate.backend.{target_name}.transpose'),triton_kernel_name)
        func_gen_smem_size=getattr(importlib.import_module(f'tritontemplate.backend.{target_name}.transpose'),f'gen_smem_size_transpose_{self._attrs["permutation"]}')
        signature,divisiability=self._gen_tensor_signature_divisiability(['inputs','outputs'])
        exec_metadata=self._gen_exec_metadata()
        num_warps=exec_metadata['num_warps']
        num_stages=exec_metadata['num_stages']

        if self._attrs['permutation'] == '10':
            constants=self._gen_constants_10(num_stages,func_gen_smem_size)
            exec_grid = self._gen_grid_10(target_name,constants)
        elif self._attrs['permutation'] == '0213':
            constants=self._gen_constants_0213(num_stages,func_gen_smem_size)
            exec_grid = self._gen_grid_0213(target_name,constants)
        else:
            raise ValueError(f"Unsupported permutation {self._attrs['permutation']}")
        
        config = triton.compiler.instance_descriptor(divisible_by_16=divisiability[16], equal_to_1=divisiability[1])

        triton_compiled_kernel=triton.compile(fn=triton_kernel,signature=signature,constants=constants,num_warps=num_warps,num_stages=num_stages,configs=[config],debug=False)

        
        return TritonExecutor(triton_compiled_kernel,exec_grid,get_warpsize(target_name),constants)