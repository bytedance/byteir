import functools
from typing import List

import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from functorch.compile import aot_module, default_partition
from torch.fx.experimental.proxy_tensor import maybe_disable_fake_tensor_mode
from torch._decomp import (
    register_decomposition,
    get_decompositions,
    core_aten_decompositions,
)

import transformers
from transformers.models.mixtral.modeling_mixtral import (
    MixtralModel,
    MixtralDecoderLayer,
)
from transformers.models.mixtral.configuration_mixtral import MixtralConfig


def fake_export_whole_mixtral():
    model_conf = MixtralConfig()
    with FakeTensorMode():
        mixtral = MixtralModel(model_conf)
        # step 1: fake init
        print(mixtral)

        """
        MixtralModel(
        (embed_tokens): Embedding(32000, 4096)
        (layers): ModuleList(
            (0-31): 32 x MixtralDecoderLayer(
            (self_attn): MixtralSdpaAttention(
                (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
                (k_proj): Linear(in_features=4096, out_features=1024, bias=False)
                (v_proj): Linear(in_features=4096, out_features=1024, bias=False)
                (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
                (rotary_emb): MixtralRotaryEmbedding()
            )
            (block_sparse_moe): MixtralSparseMoeBlock(
                (gate): Linear(in_features=4096, out_features=8, bias=False)
                (experts): ModuleList(
                (0-7): 8 x MixtralBlockSparseTop2MLP(
                    (w1): Linear(in_features=4096, out_features=14336, bias=False)
                    (w2): Linear(in_features=14336, out_features=4096, bias=False)
                    (w3): Linear(in_features=4096, out_features=14336, bias=False)
                    (act_fn): SiLU()
                )
                )
            )
            (input_layernorm): MixtralRMSNorm()
            (post_attention_layernorm): MixtralRMSNorm()
            )
        )
        (norm): MixtralRMSNorm()
        )
        """

        # step 2: torch.export
        bsz = 5
        seq_len = 7
        vocab_size = 32000
        token_ids = torch.randint(0, vocab_size, (bsz, seq_len))
        exported_mixtral = torch.export.export(mixtral, (token_ids,))
        print(exported_mixtral)

def export_mixtral_decoding_layer():
    torch._dynamo.config.capture_dynamic_output_shape_ops = True
    model_conf = MixtralConfig(hidden_size=32)
    mixtral_decoder_layer = MixtralDecoderLayer(model_conf, layer_idx=0)

    bsz = 5
    seq_len = 7
    hidden_size = 32
    x = torch.rand(bsz, seq_len, hidden_size)
    mixtral_decoder_layer = torch.export.export(mixtral_decoder_layer, (x,))
    import torch_frontend
    module = torch_frontend.compile_dynamo_model(mixtral_decoder_layer, output_type="stablehlo")
    print(module.operation.get_asm())

if __name__ == "__main__":
    export_mixtral_decoding_layer()
