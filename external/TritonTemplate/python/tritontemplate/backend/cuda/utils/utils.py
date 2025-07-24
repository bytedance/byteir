from typing import Union, List, Tuple

def shape2stride(s: Union[List[int], Tuple[int, ...]]) -> Tuple[int, ...]:
    slen=len(s)
    stride=[1]*slen
    for i in range(1,slen):
        stride[slen-i-1]=stride[slen-i]*s[slen-i]
    return tuple(stride)
