from typing import List,Optional
import importlib

import triton

from tritontemplate.compiler.base import IntImm, Tensor, Operation
from tritontemplate.compiler.dtype import dtype_str_to_triton_signature
from tritontemplate.compiler.kernel import TritonExecutor
from tritontemplate.compiler.utils import get_warpsize

_supported_layouts = ['rcr','rrr']
_supported_activations = ['relu',None]


_exec_metadata = {
    'num_warps': 4,
    'num_stages': 1,
}

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
        self._attrs['inputs'] = inputs
        self._attrs['outputs'] = outputs if outputs is not None else self._induce_output_shape()
        
    
    def _induce_output_shape(self):
        # TODO: support transpose, by swap A,B
        if self.layout == 'rcr':
            M,N,K = self._attrs['inputs'][0].shape[0],self._attrs['inputs'][1].shape[0],self._attrs['inputs'][0].shape[1]
        elif self.layout == 'rrr':
            M,K,N = self._attrs['inputs'][0].shape[0],self._attrs['inputs'][1].shape[0],self._attrs['inputs'][0].shape[1]
        else:
            raise NotImplementedError(f'layout {self.layout} not supported')
        return [Tensor(shape=[M,N],dtype=self._attrs['inputs'][0].dtype)]
    
    def _gen_signature_divisiability(self):
        signature_metadata={}
        divisiability={1:[],16:[]}
        for i,input in enumerate(self._attrs['inputs']+self._attrs['outputs']):
            if isinstance(input,Tensor):
                try:
                    sptype='*'+dtype_str_to_triton_signature(input.dtype)
                except KeyError:
                    raise KeyError(f'dtype {input.dtype} not supported')
                signature_metadata[i]=sptype
                # the ptr from torch is 16-byte aligned
                if divisiability.get(16,None) is None:
                    divisiability[16]=[i]
                else:
                    divisiability[16].append(i)
            else:
                raise NotImplementedError(f'input {input} not supported')

        return signature_metadata,divisiability
    
    @staticmethod
    def _block_size(x):
        if x>=128:
            return 128
        elif x<=32:
            return 32
        return triton.next_power_of_2(x)
    
    def _gen_constants(self,enable_tf32):
        const_metadata={}
        const_metadata['ACTIVATION'] = self._attrs['activation']

        any_float32=False
        for input in self._attrs['inputs']:
            if input.dtype == 'float32':
                any_float32=True
                break

        const_metadata['enable_tf32'] = True if (enable_tf32 and any_float32) else False
        if self.layout == 'rcr':
            input=self._attrs['inputs']
            M,N,K=input[0].shape[0],input[1].shape[0],input[0].shape[1]
            const_metadata['M']=M
            const_metadata['N']=N
            const_metadata['K']=K
            const_metadata['stride_am']=K
            const_metadata['stride_ak']=1
            const_metadata['stride_bn']=K
            const_metadata['stride_bk']=1
            const_metadata['stride_cm']=N
            const_metadata['stride_cn']=1
            if self.is_bias:
                const_metadata['stride_biasn']=1
        else:
            raise NotImplementedError(f'layout {self.layout} not supported')
        
        const_metadata['BLOCK_SIZE_M']= self._block_size(M)
        const_metadata['BLOCK_SIZE_N']= self._block_size(N)
        const_metadata['BLOCK_SIZE_K']= self._block_size(K)
        return const_metadata
    
    def _gen_exec_metadata(self):
        return _exec_metadata.copy()

    #TODO:enable_tf32 https://github.com/triton-lang/triton/issues/4574
    def compile(self,target_name,workdir,enable_tf32: bool = False,)->TritonExecutor:
        triton_kernel_name=f'gemm_{self.layout}'+ ('' if not self.is_bias else '_bias')
        triton_kernel=getattr(importlib.import_module(f'tritontemplate.backend.{target_name}.gemm'),triton_kernel_name)
        gen_grid=getattr(importlib.import_module(f'tritontemplate.backend.{target_name}.gemm'),f'gen_grid_gemm_{self.layout}')

        signature,divisiability=self._gen_signature_divisiability()
        constants=self._gen_constants(enable_tf32)
        exec_metadata=self._gen_exec_metadata()

        num_warps=exec_metadata['num_warps']
        num_stages=exec_metadata['num_stages']
        config = triton.compiler.instance_descriptor(divisible_by_16=divisiability[16], equal_to_1=divisiability[1])
        triton_compiled_kernel=triton.compile(fn=triton_kernel,signature=signature,constants=constants,num_warps=num_warps,num_stages=num_stages,configs=[config],debug=False)

        exec_grid=gen_grid(constants['M'],constants['N'],constants['BLOCK_SIZE_M'],constants['BLOCK_SIZE_N'])
        return TritonExecutor(triton_compiled_kernel,exec_grid,get_warpsize(target_name),constants)


