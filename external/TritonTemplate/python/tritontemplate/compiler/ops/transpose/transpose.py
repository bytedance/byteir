from typing import List,Optional
import importlib

import triton

from tritontemplate.compiler.base import IntImm, Tensor, Operation
from tritontemplate.compiler.dtype import get_dtype_size
from tritontemplate.compiler.kernel import TritonExecutor
from tritontemplate.compiler.utils import get_warpsize,get_cuda_device_max_shared_memory
from tritontemplate.backend.cuda.utils.utils import shape2stride

_supported_permutations = ['10','0213']


class Transpose(Operation):
    def __init__(self,
                inputs: List[Tensor],
                permutation: str,
                outputs: Optional[List[Tensor]] = None,
                name: Optional[str] = None):
        super().__init__(inputs, outputs, name)
        assert permutation in _supported_permutations, f"Unsupported permutation {permutation}"
        self._attrs['permutation'] = permutation
        self._backend_module_name = 'transpose'
        self._kernel_name = self._backend_module_name + '_' + self._attrs['permutation']

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
    
    def _gen_constants(self, enable_tf32, num_stages, func_gen_smem_size):
        const_metadata={}
        if self._attrs['permutation'] == '10':
            M,N=self._attrs['inputs'][0].shape

            const_metadata['M'] = M
            const_metadata['N'] = N
            const_metadata['stride_x0'] = N
            const_metadata['stride_x1'] = 1
            const_metadata['stride_y0'] = M
            const_metadata['stride_y1'] = 1

            const_metadata['BLOCK_SIZE_M'] = self._block_size(M)
            const_metadata['BLOCK_SIZE_N'] = self._block_size(N)
        elif self._attrs['permutation'] == '0213':
            D0,D1,D2,D3=self._attrs['inputs'][0].shape
            const_metadata['D0'] = D0
            const_metadata['D1'] = D1
            const_metadata['D2'] = D2
            const_metadata['D3'] = D3
            
            const_metadata['stride_x0'],const_metadata['stride_x1'],const_metadata['stride_x2'],const_metadata['stride_x3'] = shape2stride(self._attrs['inputs'][0].shape)
            const_metadata['stride_y0'],const_metadata['stride_y1'],const_metadata['stride_y2'],const_metadata['stride_y3']= shape2stride(self._attrs['outputs'][0].shape)

            const_metadata['BLOCK_SIZE_D1'] = self._block_size(D1)
            const_metadata['BLOCK_SIZE_D2'] = self._block_size(D2)

        self._shrink_shared_mem(func_gen_smem_size,const_metadata,get_cuda_device_max_shared_memory(),num_stages,get_dtype_size(self._attrs['inputs'][0].dtype))

        return const_metadata

    def _gen_exec_metadata(self):
        return  {
            'num_warps': 4,
            'num_stages': 2,
        }
    
    
    def compile(self, target_name, workdir, enable_tf32)->TritonExecutor:
        return super().compile(target_name, workdir, enable_tf32)