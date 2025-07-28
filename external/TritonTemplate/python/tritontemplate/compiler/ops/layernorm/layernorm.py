from typing import List,Optional
import importlib
from math import prod

import triton

from tritontemplate.compiler.base import IntImm, Tensor, Operation
from tritontemplate.compiler.dtype import get_dtype_size
from tritontemplate.compiler.kernel import TritonExecutor
from tritontemplate.compiler.utils import get_warpsize, get_cuda_device_max_shared_memory
from tritontemplate.backend.cuda.utils.utils import shape2stride

_exec_metadata = {
    'num_warps': 4,
    'num_stages': 2,
}

class Layernorm(Operation):
    def __init__(
        self,
        inputs: List[Tensor],# [x,bias(beta),weight(gamma)]
        axises:List[int],
        eps:float = 1e-5,
        outputs: Optional[List[Tensor]] = None,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(inputs, outputs, name)
        assert len(axises)==1 and axises[0] == len(inputs[0].shape)-1, f'Only last axis normalization is supported (axis={axises}, input shape={inputs[0].shape})'
        self._attrs['axis'] = axises[0]
        self._attrs['eps'] = eps

        self._deduce_output_shape()
    
    def _deduce_output_shape(self):
        M = prod(self._attrs['inputs'][0].shape[:-1])
        N = self._attrs['inputs'][0].shape[-1]
        self._attrs['M'] = M
        self._attrs['N'] = N
        self._attrs['with_weight_bias']=len(self._attrs['inputs']) == 3
        if self._attrs['outputs'] is None:
            self._attrs['outputs'] = [Tensor(shape=self._attrs['inputs'][0].shape,dtype=self._attrs['inputs'][0].dtype)]

    def _gen_constants(self,num_stages,func_gen_smem_size):
        const_metadata={}
        const_metadata['M']= self._attrs['M']
        const_metadata['N']= self._attrs['N']
        const_metadata['stride_x0'] = self._attrs['N']
        const_metadata['stride_x1'] = 1
        const_metadata['stride_y0'] = self._attrs['N']  
        const_metadata['stride_y1'] = 1
        const_metadata['eps'] = self._attrs['eps']

        const_metadata['BLOCK_SIZE_M'] = self._block_size(self._attrs['M'])
        const_metadata['BLOCK_SIZE_N'] = self._block_size(self._attrs['N'])

        self._shrink_shared_mem(func_gen_smem_size,const_metadata,get_cuda_device_max_shared_memory(),num_stages,get_dtype_size(self._attrs['inputs'][0].dtype))

        if self._attrs['with_weight_bias']:
            const_metadata['stride_weight'] = 1
            const_metadata['stride_bias'] = 1
        
        return const_metadata
    
    def _gen_exec_metadata(self):
        return _exec_metadata.copy()
    
    def compile(self, target_name, workdir,enable_tf32)->TritonExecutor:
        triton_kernel_name= 'layernorm_weight_bias' if self._attrs['with_weight_bias'] else 'layernorm'
        triton_kernel=getattr(importlib.import_module(f'tritontemplate.backend.{target_name}.layernorm'),triton_kernel_name)
        gen_grid=getattr(importlib.import_module(f'tritontemplate.backend.{target_name}.layernorm'),f'gen_grid_layernorm')
        func_gen_smem_size=getattr(importlib.import_module(f'tritontemplate.backend.{target_name}.layernorm'),f'gen_smem_size_layernorm')

        signature,divisiability=self._gen_tensor_signature_divisiability(['inputs','outputs'])

        exec_metadata=self._gen_exec_metadata()

        num_warps=exec_metadata['num_warps']
        num_stages=exec_metadata['num_stages']

        constants=self._gen_constants(num_stages,func_gen_smem_size)
        config = config = triton.compiler.instance_descriptor(divisible_by_16=divisiability[16], equal_to_1=divisiability[1])
        triton_compiled_kernel=triton.compile(fn=triton_kernel,signature=signature,constants=constants,num_warps=num_warps,num_stages=num_stages,configs=[config],debug=False)

        exec_grid=gen_grid(constants['M'],constants['BLOCK_SIZE_M'])
        return TritonExecutor(triton_compiled_kernel,exec_grid,get_warpsize(target_name),constants)
            
        