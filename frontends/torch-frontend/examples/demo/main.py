from torch import nn
import torch
import transformers
import sys
import os
import functools
import torch._dynamo as dynamo
import torch.nn.functional as F

import transformers
import argparse

MODEL_LIST = ["gpt2", "bloom-560m", "llama", "llama-2", "opt-1.3b", "nanogpt", "chatglm"]

AUTH_TOKEN="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"

class InferLLAMAModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = transformers.LlamaConfig(num_hidden_layers=4, return_dict=False)
        self.model = transformers.LlamaForCausalLM(config=self.config)
    def forward(self, x):
        return self.model(x)[0]

class InferOPTModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = transformers.AutoConfig.from_pretrained("facebook/opt-1.3b", return_dict=False)
        self.config.tie_word_embeddings = False
        self.model = transformers.OPTForCausalLM(config=self.config)
    def forward(self, x):
        return self.model(x)[0]

class InferBLOOMModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = transformers.BloomConfig.from_pretrained('bigscience/bloom-560m', return_dict=False)
        self.config.tie_word_embeddings = False
        self.model = transformers.BloomForCausalLM(config=self.config)
    def forward(self, x):
        return self.model(x)[0]

class InferGPT2Module(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = transformers.GPT2Config.from_pretrained('gpt2', return_dict=False)
        self.config.num_labels = self.config.vocab_size
        self.model = transformers.GPT2ForTokenClassification(config=self.config)
    def forward(self, x):
        return self.model(x)[0]

def make_model(model_name):
    if model_name == 'llama':
        config = transformers.LlamaConfig(num_hidden_layers=4)
        model = transformers.LlamaForCausalLM(config=config)
    elif model_name == 'llama-2':
        config = transformers.AutoConfig.from_pretrained(
            "meta-llama/Llama-2-7b-chat-hf", use_auth_token=AUTH_TOKEN)
        config.num_hidden_layers = 2
        model = transformers.AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b-chat-hf",
            torch_dtype=torch.float,
            use_auth_token=AUTH_TOKEN,
            config=config
        )
    elif model_name == 'opt-1.3b':
        config = transformers.AutoConfig.from_pretrained("facebook/opt-1.3b")
        config.tie_word_embeddings = False
        model = transformers.OPTForCausalLM(config=config)
    elif model_name == 'bloom-560m':
        config = transformers.BloomConfig.from_pretrained('bigscience/bloom-560m')
        config.tie_word_embeddings = False
        model = transformers.BloomForCausalLM(config=config)
    elif model_name == 'gpt2':
        config = transformers.GPT2Config.from_pretrained('gpt2')
        config.num_labels = config.vocab_size
        model = transformers.GPT2ForTokenClassification(config=config)
    elif model_name == 'chatglm':
        config = transformers.AutoConfig.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)
        # fixed dynamo tracing unsupported UserDefinedObjectVariable(split)
        from models.modeling_chatglm import ChatGLMForConditionalGeneration
        # config.num_layers = 4
        model = ChatGLMForConditionalGeneration(config=config).half()
        base_model = transformers.AutoModel.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True, config=config).half().cuda()
        model.load_state_dict(base_model.state_dict())
    elif model_name == 'nanogpt':
        from models.modeling_nanogpt import GPTConfig, GPT
        config_args = dict(n_layer=12, n_head=12, n_embd=768)
        config_args['vocab_size'] = 50257  # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024  # always 1024 for GPT model checkpoints
        config_args['bias'] = True  # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        config_args['dropout'] = 0.
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
    else:
        assert False
    return model

def make_inference_model(model_name):
    if model_name == 'llama':
        return InferLLAMAModule()
    elif model_name == 'opt-1.3b':
        return InferOPTModule()
    elif model_name == 'bloom-560m':
        return InferBLOOMModule()
    elif model_name == 'gpt2':
        return InferGPT2Module()
    else:
        return make_model(model_name)

def make_data(model, model_name, device):
    batch_size = 8
    if model_name == 'llama':
        batch_size = 16
    elif model_name == 'opt-1.3b':
        batch_size = 4
    seq_len = 1024
    input = torch.randint(
        low=0, high=model.config.vocab_size, size=(batch_size, seq_len), device=device
    )

    label = torch.randint(low=0, high=model.config.vocab_size, size=(batch_size, seq_len),
                          device=device)
    return input, label

def compute_loss(model, data, model_name):
    if model_name == 'nanogpt':
        input_idx, output_idx = data
        _, loss = model(input_idx, output_idx)
    else:
        input, label = data
        output = model(input)
        logits = output.logits
        loss = F.cross_entropy(logits.view(-1, model.config.vocab_size), label.view(-1))
    return loss


def infer_model(args):
    device = torch.device('cuda:' + str(args.device_id))
    model = make_inference_model(args.model_name)
    model = model.eval()
    model.to(device)

    input, label = make_data(model, args.model_name, device)
    trace_data = [input]
    if args.model_name == "nanogpt":
        trace_data.append(label)
    # torch.save(trace_data, "batch_sample_inputs")

    TEMP_FOLDER="./temp"
    os.makedirs(TEMP_FOLDER, exist_ok=True)
    os.makedirs(TEMP_FOLDER + f"/{args.model_name}_inference_f16", exist_ok=True)
    jit_file_name = TEMP_FOLDER + f"/{args.model_name}_inference.f16.jit"
    mhlo_file_name = TEMP_FOLDER + f"/{args.model_name}_inference.f16.mhlo.mlir"
    byre_file_name = TEMP_FOLDER + f"/{args.model_name}_inference_f16/{args.model_name}.rt.mlir"

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
        if args.model_name in ["llama-2", "chatglm"]:
            # use dynamo
            torch._dynamo.reset()
            import backend
            from backend import fuse_aware_byteir_compile_fx
            backend.MODEL_NAME = args.model_name + "_inference"
            backend.FLASH = args.flash
            optimized_model = torch.compile(model, backend=fuse_aware_byteir_compile_fx, fullgraph=True)
            torch_logits = model(input)
            print("torch outputs:")
            print(torch_logits.logits)
            print("byre inputs:")
            print(trace_data)
            print("byre outputs:")
            print(optimized_model(*trace_data).logits)
        else:
            if not os.path.exists(mhlo_file_name):
                # module = torch.jit.load(jit_file_name)
                if args.flash:
                    from torch.fx.experimental.proxy_tensor import make_fx
                    from torch_frontend import preprocess_fx_graph
                    module = make_fx(model)(*trace_data)
                    print("torch inputs:")
                    print(trace_data)
                    print("torch outputs:")
                    print(module(*trace_data))
                    module = preprocess_fx_graph(module)
                else:
                    module = torch.jit.trace(model, trace_data, check_trace=False)
                    print("torch inputs:")
                    print(trace_data)
                    print("torch outputs:")
                    print(module(*trace_data))
                import torch_frontend
                mhlo_model = torch_frontend.compile(module, trace_data, "mhlo")
                with open(mhlo_file_name, "w") as f:
                    print(mhlo_model.operation.get_asm(), file=f)
                print("save mhlo to {}".format(mhlo_file_name))

            if not os.path.exists(byre_file_name):
                import byteir
                print("begin byteir compile")
                byteir.compile(mhlo_file_name, byre_file_name, entry_func='forward', target='cuda_with_ait', disable_byteir_cache=False, verbose=False)
                print("byteir compile to {}".format(byre_file_name))

            from backend import ByteIRInferenceFunction
            runner = ByteIRInferenceFunction(byre_file_name)
            print("byre inputs:")
            print(trace_data)
            print("byre outputs:")
            print(runner(*trace_data))

def train_model(args):
    torch._dynamo.reset()
    torch._dynamo.disallow_in_graph(F.cross_entropy)

    model_name = args.model_name
    use_flash_attn = args.flash
    device = torch.device('cuda:' + str(args.device_id))
    model = make_model(model_name)
    model.to(device)

    import backend
    from backend import fuse_aware_byteir_compile_fx
    backend.MODEL_NAME = model_name
    backend.FLASH = use_flash_attn

    optimized_model = torch.compile(model, backend=fuse_aware_byteir_compile_fx, fullgraph=True)

    data = make_data(optimized_model, model_name, device)
    model.zero_grad(set_to_none=True)
    with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
        loss = compute_loss(optimized_model, data, model_name)
        torch_loss = compute_loss(model, data, model_name)
        print("loss:", loss)
        print("torch_loss:", torch_loss)
        loss.backward()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name")
    parser.add_argument("--flash", action="store_true", help="use flash attention when possible")
    parser.add_argument("--infer", action="store_true", help="infer mode")
    parser.add_argument("--device_id", type=int, default=0)
    args = parser.parse_args()
    # print(args)

    assert args.model_name in MODEL_LIST
    if args.infer:
        infer_model(args)
    else:
        train_model(args)

