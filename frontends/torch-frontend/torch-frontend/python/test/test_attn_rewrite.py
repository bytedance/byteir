import torch
import torch_frontend
from functorch.compile import aot_module
from torch.testing import FileCheck
import transformers
import torch.nn.functional as F
import copy
from typing import Dict, List

def make_data(model, device):
    batch_size = 2
    seq_len = 128
    input = torch.randint(
        low=0, high=model.config.vocab_size, size=(batch_size, seq_len), device=device
    )

    label = torch.randint(low=0, high=model.config.vocab_size, size=(batch_size, seq_len),
                          device=device)
    return input, label


def copy_nested_tensor(nested_tensor):
    """
    Args:
        nested_tensor is a nested list/dict of Pytorch tensors
    """
    if isinstance(nested_tensor, torch.Tensor):
        output = nested_tensor.clone().detach()
        if nested_tensor.requires_grad:
            output.requires_grad_()
        return output

    elif isinstance(nested_tensor, List):
        return [copy_nested_tensor(tensor) for tensor in nested_tensor]
    elif isinstance(nested_tensor, Dict):
        return {
            key: copy_nested_tensor(nested_tensor) for key, tensor in nested_tensor.items()
        }
    elif isinstance(nested_tensor, tuple):
        return tuple([copy_nested_tensor(tensor) for tensor in nested_tensor])
    elif nested_tensor is None:
        return nested_tensor
    elif isinstance(nested_tensor, (bool, int, str, float)):
        return copy.deepcopy(nested_tensor)
    else:
        raise TypeError(f'Unsupported type {type(nested_tensor)}. {nested_tensor}')

def trival_compile_fx_inner(gm, inputs):
    return gm

def trival_compile_fx(model_: torch.fx.GraphModule, inputs):
    model_ = torch_frontend.fx_replace_attn_pattern(model_)
    all_formatted = "\n".join([n.format_node() for n in model_.graph.nodes])
    FileCheck().check("call_function").check(
            "torch.ops.aten.scaled_dot_product_attention").run(all_formatted)
    module = aot_module(model_, fw_compiler=trival_compile_fx_inner)
    return module

def test_flash_attn_gpt2_pattern():
    torch.manual_seed(0)
    config = transformers.GPT2Config.from_pretrained('gpt2')
    config.num_labels = config.vocab_size
    config.attn_pdrop = 0.0
    config.resid_pdrop = 0.0
    config.embd_pdrop = 0.0
    config.classifier_dropout = 0.0
    model = transformers.GPT2ForTokenClassification(config=config).to("cuda")

    flash_model = transformers.GPT2ForTokenClassification(config=config).to("cuda")
    flash_model.load_state_dict(model.state_dict())

    data = make_data(model, "cuda")
    flash_data = copy_nested_tensor(data)

    model.zero_grad(set_to_none=True)
    flash_model.zero_grad(set_to_none=True)
    flash_attn_gm = torch.compile(flash_model, backend=trival_compile_fx)
    with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
        input_data, label = data
        output = flash_model(input_data)
        golden_logits = output.logits
        golden_loss = F.cross_entropy(golden_logits.view(-1, model.config.vocab_size), label.view(-1))

        flash_input_data, flash_label = flash_data
        flash_output = flash_attn_gm(flash_input_data)
        flash_logits = flash_output["logits"]
        flash_loss = F.cross_entropy(flash_logits.view(-1, model.config.vocab_size), flash_label.view(-1))

    torch.testing.assert_close(golden_loss, flash_loss, atol=1e-4, rtol=1e-6)
    torch.testing.assert_close(golden_logits, flash_logits, atol=4e-3, rtol=1e-5)


def test_flash_attn_llama_pattern():
    torch.manual_seed(0)
    config = transformers.LlamaConfig(num_hidden_layers=2)
    config.hidden_size=512
    model = transformers.LlamaForCausalLM(config=config).to("cuda")

    flash_model = transformers.LlamaForCausalLM(config=config).to("cuda")
    flash_model.load_state_dict(model.state_dict())

    data = make_data(model, "cuda")
    flash_data = copy_nested_tensor(data)

    model.zero_grad(set_to_none=True)
    flash_model.zero_grad(set_to_none=True)
    flash_attn_gm = torch.compile(flash_model, backend=trival_compile_fx)
    with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
        input_data, label = data
        output = flash_model(input_data)
        golden_logits = output.logits
        golden_loss = F.cross_entropy(golden_logits.view(-1, model.config.vocab_size), label.view(-1))
    
        flash_input_data, flash_label = flash_data
        flash_output = flash_attn_gm(flash_input_data)
        flash_logits = flash_output["logits"]
        flash_loss = F.cross_entropy(flash_logits.view(-1, model.config.vocab_size), flash_label.view(-1))

    # Flash attention uses a different mask than the original llama, relax check
    torch.testing.assert_close(golden_loss, flash_loss, atol=2e-4, rtol=1e-6)
    torch.testing.assert_close(golden_logits, flash_logits, atol=3e-2, rtol=1e-6)


def test_flash_attn_bloom_pattern():
    torch.manual_seed(0)
    config = transformers.BloomConfig.from_pretrained('bigscience/bloom-560m')
    config.tie_word_embeddings = False
    config.hidden_size=512
    config.num_hidden_layers=2
    model = transformers.BloomForCausalLM(config=config).to("cuda")

    flash_model = transformers.BloomForCausalLM(config=config).to("cuda")
    flash_model.load_state_dict(model.state_dict())

    data = make_data(model, "cuda")
    flash_data = copy_nested_tensor(data)

    with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
        model.zero_grad(set_to_none=True)
        input_data, label = data
        output = flash_model(input_data)
        golden_logits = output.logits
        golden_loss = F.cross_entropy(golden_logits.view(-1, model.config.vocab_size), label.view(-1))
    
        flash_model.zero_grad(set_to_none=True)
        flash_attn_gm = torch.compile(flash_model, backend=trival_compile_fx)
        flash_input_data, flash_label = flash_data
        flash_output = flash_attn_gm(flash_input_data)
        flash_logits = flash_output["logits"]
        flash_loss = F.cross_entropy(flash_logits.view(-1, model.config.vocab_size), flash_label.view(-1))

    # Bloom uses Alibi tensor, which is not causal. Transformation is not mathematically equivalent
    torch.testing.assert_close(golden_loss, flash_loss, atol=1e-3, rtol=1e-6)


def test_flash_attn_opt_pattern():
    torch.manual_seed(0)
    config = transformers.AutoConfig.from_pretrained("facebook/opt-1.3b")
    config.tie_word_embeddings = False
    config.hidden_size=512
    config.num_hidden_layers=2
    config.dropout=0.0
    model = transformers.OPTForCausalLM(config=config).to("cuda")

    flash_model = transformers.OPTForCausalLM(config=config).to("cuda")
    flash_model.load_state_dict(model.state_dict())

    data = make_data(model, "cuda")
    flash_data = copy_nested_tensor(data)

    with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
        model.zero_grad(set_to_none=True)
        input_data, label = data
        output = flash_model(input_data)
        golden_logits = output.logits
        golden_loss = F.cross_entropy(golden_logits.view(-1, model.config.vocab_size), label.view(-1))
    
        flash_model.zero_grad(set_to_none=True)
        flash_attn_gm = torch.compile(flash_model, backend=trival_compile_fx)
        flash_input_data, flash_label = flash_data
        flash_output = flash_attn_gm(flash_input_data)
        flash_logits = flash_output["logits"]
        flash_loss = F.cross_entropy(flash_logits.view(-1, model.config.vocab_size), flash_label.view(-1))

    torch.testing.assert_close(golden_loss, flash_loss, atol=1e-4, rtol=1e-6)
    torch.testing.assert_close(golden_logits, flash_logits, atol=3e-3, rtol=1e-6)


def test_flash_attn_llama_inference_pattern():
    config = transformers.LlamaConfig(num_hidden_layers=2)
    model = transformers.LlamaForCausalLM(config=config).to("cuda")
    model.eval()

    input, label = make_data(model, "cuda")
    trace_data = [input]

    from torch.fx.experimental.proxy_tensor import make_fx
    from torch_frontend import preprocess_fx_graph
    # module = torch.jit.trace(model, trace_data, check_trace=False)
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
        fx_g = make_fx(model)(*trace_data)
        fx_g = preprocess_fx_graph(fx_g)
        all_formatted = "\n".join([n.format_node() for n in fx_g.graph.nodes])
        FileCheck().check("call_function").check("torch.ops.byteir.flash_attn_fwd").run(all_formatted)
