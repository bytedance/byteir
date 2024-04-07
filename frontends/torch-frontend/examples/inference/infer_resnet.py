import torch
import torchvision.models as models

from torch_frontend import compile, register_decomposition_in_torchscript
from torch_frontend import torch_mlir
from torch_mlir.passmanager import PassManager

model = models.resnet18(pretrained=True)
model.eval()
# For torchscript model, please execute the following commands to decompose complex ops first
# torch_frontend.register_decomposition_in_torchscript()
# torch._C._jit_pass_inline(model.graph)
# torch._C._jit_pass_run_decompositions(model.graph)
data = torch.randn(2,3,200,200)

module = compile(model, data, "stablehlo", use_tracing=False)

out_mhlo_mlir_path = "./resnet18.stablehlo.mlir"

with open(out_mhlo_mlir_path, "w", encoding="utf-8") as outf:
    module_str = module.operation.get_asm(enable_debug_info=False, print_generic_op_form=True) # large_elements_limit=10
    print(module_str, file=outf)

print(f"Stablehlo IR of resent18 successfully written into {out_mhlo_mlir_path}")
