import torch
import torch.fx
from functorch import make_fx
import torch_frontend
from transformers import BertForMaskedLM

sample_inputs = [torch.randint(100, (1, 128))]
bert = BertForMaskedLM.from_pretrained("prajjwal1/bert-tiny",
					            return_dict=False)   

print(bert(*sample_inputs))

bert = torch.jit.trace(bert, sample_inputs)
bert.train(False)

torch_frontend.register_decomposition_in_torchscript()
torch._C._jit_pass_inline(bert.graph)
torch._C._jit_pass_run_decompositions(bert.graph)

# FX rewrite
torch._C._jit_set_nvfuser_enabled(False)
fx_g = make_fx(bert)(*sample_inputs)
fx_g = torch_frontend.preprocess_fx_graph(fx_g)
fx_g.graph.lint()
fx_g.recompile()
# print(fx_g.code)
bert = torch.jit.trace(fx_g, sample_inputs)

mlir_module = torch_frontend.compile(bert, sample_inputs, "stablehlo")
with open("./bert.stablehlo.mlir", "w") as f:
  print(mlir_module.operation.get_asm(
    enable_debug_info=False, print_generic_op_form=True), file=f)
