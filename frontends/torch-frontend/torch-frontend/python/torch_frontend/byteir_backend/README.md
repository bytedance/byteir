*Requires torch version 2.4 or higher*

- Usage

```python
import torch

import torch_frontend
from torch_frontend import byteir_backend as byteir_backend
from torch_frontend.byteir_backend.utils import *

class NaiveModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x0, x1, x2):
        r0 = torch.ops.aten.mul(x0, x1)
        r1 = torch.ops.aten.div(r0, x2)
        x0 = torch.ops.aten.mul(r1, r1) - x0
        r2 = torch.ops.aten.slice(x0, 1, 1, 3, 1)
        return r1, r2

model = NaiveModel()
opt_mod = torch.compile(model, backend="byteir")

x0 = torch.rand(32, 64).to('cuda')
x1 = torch.rand(32, 64).to('cuda')
x2 = torch.rand(32, 64).to('cuda')

x0 = x0.as_strided(size=(32,16), stride=(64,2), storage_offset=16)
x1 = x1.as_strided(size=(32,16), stride=(64,1), storage_offset=8)
x2 = x2.as_strided(size=(32,16), stride=(32,1), storage_offset=32)

golden = model(x0, x1, x2)
outs = opt_mod(x0, x1, x2)
torch.cuda.synchronize()

torch.testing.assert_close(golden, outs)
```
