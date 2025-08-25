from typing import List,Optional
import importlib

import triton

from tritontemplate.compiler.base import IntImm, Tensor, Operation
from tritontemplate.compiler.dtype import get_dtype_size
from tritontemplate.compiler.kernel import TritonExecutor
from tritontemplate.compiler.utils import get_warpsize,get_cuda_device_max_shared_memory
from tritontemplate.backend.cuda.utils import shape2stride

_supported_layouts = ['rcr','rrr','ccr','crr']
_supported_activations = ['relu',None]


class Gemm(Operation):
    def __init__(
        self,
        inputs: List[Tensor],
        layout: str,
        is_bias: bool = False,
        outputs: Optional[List[Tensor]] = None,
        activation: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None:
        assert layout in _supported_layouts, f'layout {layout} not supported'
        assert activation in _supported_activations, f'activation {activation} not supported'

        super().__init__(inputs, outputs, name)
        self.layout = layout
        self.is_bias= is_bias
        self._attrs['activation'] = activation
        self._deduce_output_shape()
        self._backend_module_name = 'gemm'
        self._kernel_name = self._backend_module_name + ('' if not self.is_bias else '_bias')

    
    def _deduce_output_shape(self):

        is_transpose_a=self.layout[0]=='c'
        is_transpose_b=self.layout[1]=='c'
        M=self._attrs['inputs'][0].shape[1] if is_transpose_a else self._attrs['inputs'][0].shape[0]
        K=self._attrs['inputs'][0].shape[0] if is_transpose_a else self._attrs['inputs'][0].shape[1]
        N=self._attrs['inputs'][1].shape[0] if is_transpose_b else self._attrs['inputs'][1].shape[1]


        self._attrs['M'] = M
        self._attrs['K'] = K
        self._attrs['N'] = N
        self._attrs['is_transpose_a'] = is_transpose_a
        self._attrs['is_transpose_b'] = is_transpose_b

        res_shape=[M,N] if self.layout[2]=='r' else [N,M]
        if self._attrs['outputs'] is None:
            self._attrs['outputs'] = [Tensor(shape=res_shape,dtype=self._attrs['inputs'][0].dtype)]
        else:
            assert self._attrs['outputs'][0].shape == res_shape, f"output shape {self._attrs['outputs'][0].shape} not match {res_shape}"

    def _gen_constants(self,enable_tf32,num_stages, func_gen_smem_size):
        const_metadata={}
        const_metadata['ACTIVATION'] = self._attrs['activation']

        any_float32=False
        for input in self._attrs['inputs']:
            if input.dtype == 'float32':
                any_float32=True
                break

        const_metadata['enable_tf32'] = True if (enable_tf32 and any_float32) else False
                
        const_metadata['BLOCK_SIZE_M']= self._block_size(self._attrs['M'])
        const_metadata['BLOCK_SIZE_N']= self._block_size(self._attrs['N'])
        const_metadata['BLOCK_SIZE_K']= self._block_size(self._attrs['K'])
        self._shrink_shared_mem(func_gen_smem_size,const_metadata,get_cuda_device_max_shared_memory(),num_stages,get_dtype_size(self._attrs['inputs'][0].dtype))

        input=self._attrs['inputs']
        output=self._attrs['outputs']
        const_metadata['M']=self._attrs['M']
        const_metadata['N']=self._attrs['N']
        const_metadata['K']=self._attrs['K']
        
        const_metadata['is_transpose_a']=self._attrs['is_transpose_a']
        const_metadata['is_transpose_b']=self._attrs['is_transpose_b']
        const_metadata['stride_a0'],const_metadata['stride_a1']=shape2stride(input[0].shape)
        const_metadata['stride_b0'],const_metadata['stride_b1']=shape2stride(input[1].shape)
        
        if self.is_bias:
            const_metadata['stride_bias0']=1

        const_metadata['stride_c0'],const_metadata['stride_c1']=shape2stride(output[0].shape)
        return const_metadata
    
    def _gen_exec_metadata(self):
        return {
            'num_warps': 4,
            'num_stages': 2,
        }

    #TODO:enable_tf32 https://github.com/triton-lang/triton/issues/4574
    def compile(self,target_name,workdir,enable_tf32: bool = False,)->TritonExecutor:
        return super().compile(target_name,workdir,enable_tf32)
