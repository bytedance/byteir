import torch
import torchvision.models as models

import torch_frontend
from torch_frontend import convert_to_mhlo_via_torch_mlir


def test_resnet18_compile():
    resnet18 = models.resnet18(pretrained=True)
    resnet18.train(False)
    inputs = torch.ones(1, 3, 224, 224)
    
    module = convert_to_mhlo_via_torch_mlir(resnet18, inputs)

    print(module.operation.get_asm(large_elements_limit=10, enable_debug_info=False))
