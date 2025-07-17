from abc import ABC,abstractmethod
from pprint import pformat
from typing import Any, Dict, Iterable, List, Optional, Set, Union

from tritontemplate.compiler.dtype import dtype_str_to_triton_signature

class BaseType(ABC):
    def __init__(self) -> None:
        super().__init__()
        self._attrs: Dict[str, Any] = {"name": None, "nop": False}

    def __str__(self) -> str:
        return pformat(self._attrs, indent=2, depth=2)

    def __repr__(self) -> str:
        return self.__str__()


class IntImm(int, BaseType): 
    def __new__(cls, val: int, divisibility: Optional[int] = None, name: Optional[str] = None):
        instance = super().__new__(cls, val)
        return instance

    def __init__(
        self,
        val: int, 
        divisibility: Optional[int] = None,
        name: Optional[str] = None,
    ) -> None:
        BaseType.__init__(self)
        self._attrs['val'] = int(self)
        if name is not None:
            self._attrs['name'] = name
        self._attrs['divisibility'] = divisibility

    @property
    def name(self) -> Optional[str]:
        return self._attrs.get('name')

    @property
    def divisibility(self) -> Optional[int]:
        return self._attrs.get('divisibility')
    
    # set divisibility:
    # @divisibility.setter
    # def divisibility(self, value: Optional[int]):
    #     self._attrs['divisibility'] = value
    
    @property
    def val(self) -> int:
        return int(self)


class Tensor(BaseType):
    """
    """
    def __init__(
        self,
        shape: List[IntImm],
        dtype: str = "float16",
        name: Optional[str] = None,
    ) -> None:
        super().__init__()
        if name is not None:
            self._attrs['name'] = name
        self._attrs['dtype'] = dtype
        self._attrs['shape'] = shape

    @property
    def name(self) -> Optional[str]:
        return self._attrs.get('name')

    @property
    def dtype(self) -> str:
        return self._attrs['dtype']

    @property
    def shape(self) -> List[IntImm]:
        return self._attrs['shape']


class Operation(BaseType):
    """
    """
    def __init__(
        self,
        inputs: List[BaseType],
        outputs: Optional[List[BaseType]] = None,
        name: Optional[str] = None,
    ) -> None:
        super().__init__()
        self._attrs['inputs'] = inputs
        self._attrs['outputs'] = outputs
        self._attrs['name'] = name

    @property
    def name(self) -> Optional[str]:
        return self._attrs.get('name')

    @property
    def inputs(self) -> List[Tensor]:
        return self._attrs['inputs']

    @property
    def outputs(self) -> Optional[List[Tensor]]:
        return self._attrs['outputs']
    
    @abstractmethod
    def compile(self,target_name,workdir):
        raise NotImplementedError

 
    def _gen_tensor_signature_divisiability(self,tensors_names:List[str]):
        signature_metadata={}
        divisiability={1:[],16:[]}
        tensor_obj=[]
        for tensor_name in tensors_names:
            tensor_obj+=self._attrs[tensor_name]
        for i,input in enumerate(tensor_obj):
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
        if x<=32:
            return 32
        elif x<=64:
            return 64
        else:
            return 128
