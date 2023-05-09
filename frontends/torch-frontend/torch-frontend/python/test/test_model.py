import torch

import torch_frontend
from torch_frontend import convert_to_mhlo_via_torch_mlir

def test_resnet18_compile():
    import torchvision.models as models
    resnet18 = models.resnet18(pretrained=True)
    resnet18.train(False)
    inputs = torch.ones(1, 3, 224, 224)

    module = convert_to_mhlo_via_torch_mlir(resnet18, inputs)
    print(module.operation.get_asm(large_elements_limit=10, enable_debug_info=False))

# def test_berttiny_compile():
#     from functorch import make_fx
#     from transformers import BertForMaskedLM
#     bert = BertForMaskedLM.from_pretrained("prajjwal1/bert-tiny",
# 					    return_dict=False)
#     bert.train(False)
#     inputs = [torch.randint(100, (1, 128))]
#     bert = torch.jit.trace(bert, inputs)

#     # TODO: support unpack tuple in torch-mlir
#     # FX rewrite to unpack return tuple
#     torch._C._jit_set_nvfuser_enabled(False)
#     fx_g = make_fx(bert)(*inputs)
#     fx_g = torch_frontend.preprocess_fx_graph(fx_g)
#     fx_g.graph.lint()
#     fx_g.recompile()
#     bert = torch.jit.trace(fx_g, inputs)

#     module = convert_to_mhlo_via_torch_mlir(bert, inputs)
#     print(module.operation.get_asm(large_elements_limit=10, enable_debug_info=False))
