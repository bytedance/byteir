from typing import List,Optional
import importlib
from math import prod

import triton

from tritontemplate.compiler.base import IntImm, Tensor, Operation
from tritontemplate.compiler.dtype import get_dtype_size
from tritontemplate.compiler.kernel import TritonExecutor
from tritontemplate.compiler.utils import get_cuda_device_max_shared_memory


class Softmax(Operation):
    def __init__(self, inputs: List[Tensor], dim: int,enable_online:bool=True, outputs: Optional[List[Tensor]] = None, name: Optional[str] = None):
        super().__init__(inputs, outputs, name)
        assert dim == len(inputs[0].shape)-1, f'only support last axis now'
        self._attrs['dim'] = dim
        self._attrs['enable_online'] = enable_online
        self._deduce_output_shape()
        self._backend_module_name = 'softmax'
        self._kernel_name = 'online_softmax' if self._attrs['enable_online'] else 'softmax'

    def _deduce_output_shape(self):
        M = prod(self._attrs['inputs'][0].shape[:-1])
        N = self._attrs['inputs'][0].shape[-1]

        self._attrs['M']= M
        self._attrs['N']= N

        if self._attrs['outputs'] is None:
            # Return float32
            self._attrs['outputs'] = [Tensor(shape=self._attrs['inputs'][0].shape,dtype='float32')]
            # self._attrs['outputs'] = [Tensor(shape=self._attrs['inputs'][0].shape,dtype=self._attrs['inputs'][0].dtype)]

    def _gen_constants(self,enable_tf32, num_stages,func_gen_smem_size):
        const_metadata={}
        const_metadata['M']= self._attrs['M']
        const_metadata['N']= self._attrs['N']
        const_metadata['stride_x0'] = self._attrs['N']
        const_metadata['stride_x1'] = 1
        const_metadata['stride_y0'] = self._attrs['N']
        const_metadata['stride_y1'] = 1

        const_metadata['BLOCK_SIZE_M'] = self._block_size(self._attrs['M'])
        const_metadata['BLOCK_SIZE_N'] = self._block_size(self._attrs['N'])
        self._shrink_shared_mem(func_gen_smem_size,const_metadata,get_cuda_device_max_shared_memory(),num_stages,get_dtype_size(self._attrs['inputs'][0].dtype))
        
        return const_metadata
    
    def _gen_exec_metadata(self):
        return {
            'num_warps': 4,
            'num_stages': 2,
        }
    
    def compile(self, target_name, workdir, enable_tf32)->TritonExecutor:
        return super().compile(target_name,workdir,enable_tf32)