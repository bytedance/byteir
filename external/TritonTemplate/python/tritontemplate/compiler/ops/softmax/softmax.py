from typing import List,Optional
import importlib
from math import prod

import triton

from tritontemplate.compiler.base import IntImm, Tensor, Operation
from tritontemplate.compiler.dtype import dtype_str_to_triton_signature
from tritontemplate.compiler.kernel import TritonExecutor
from tritontemplate.compiler.utils import get_warpsize
from tritontemplate.backend.cuda.utils.utils import shape2stride

_exec_metadata = {
    'num_warps': 4,
    'num_stages': 1,
}

class Softmax(Operation):
    def __init__(self, inputs: List[Tensor], dim: int,enable_online:bool=True, outputs: Optional[List[Tensor]] = None, name: Optional[str] = None):
        super().__init__(inputs, outputs, name)
        self._attrs['inputs'] = inputs
        self._attrs['outputs'] = outputs 
        self._attrs['enable_online'] = enable_online

        assert dim == len(inputs[0].shape)-1
        self._deduce_output_shape()

    def _deduce_output_shape(self):
        M = prod(self._attrs['inputs'][0].shape[:-1])
        N = self._attrs['inputs'][0].shape[-1]

        self._attrs['M']= M
        self._attrs['N']= N

        if self._attrs['outputs'] is None:
            self._attrs['outputs'] = [Tensor(shape=self._attrs['inputs'][0].shape,dtype=self._attrs['inputs'][0].dtype)]

    def _gen_constants(self):
        const_metadata={}
        const_metadata['M']= self._attrs['M']
        const_metadata['N']= self._attrs['N']
        const_metadata['stride_x0'] = self._attrs['N']
        const_metadata['stride_x1'] = 1
        const_metadata['stride_y0'] = self._attrs['N']
        const_metadata['stride_y1'] = 1

        const_metadata['BLOCK_SIZE_M'] = self._block_size(self._attrs['M'])
        const_metadata['BLOCK_SIZE_N'] = self._block_size(self._attrs['N'])
        
        return const_metadata
    
    def _gen_exec_metadata(self):
        return _exec_metadata.copy()
    
    def compile(self, target_name, workdir, enable_tf32)->TritonExecutor:
        triton_kernel_name= 'online_softmax' if self._attrs['enable_online'] else 'softmax'
        triton_kernel=getattr(importlib.import_module(f'tritontemplate.backend.{target_name}.softmax'),triton_kernel_name)
        gen_grid=getattr(importlib.import_module(f'tritontemplate.backend.{target_name}.softmax'),f'gen_grid_softmax')
        signature,divisiability=self._gen_tensor_signature_divisiability(['inputs','outputs'])
        constants=self._gen_constants()
        exec_metadata=self._gen_exec_metadata()

        num_warps=exec_metadata['num_warps']
        num_stages=exec_metadata['num_stages']
        config = triton.compiler.instance_descriptor(divisible_by_16=divisiability[16], equal_to_1=divisiability[1])
        triton_compiled_kernel=triton.compile(fn=triton_kernel,signature=signature,constants=constants,num_warps=num_warps,num_stages=num_stages,configs=[config],debug=False)

        exec_grid=gen_grid(constants['M'],constants['BLOCK_SIZE_M'])
        return TritonExecutor(triton_compiled_kernel,exec_grid,get_warpsize(target_name),constants)