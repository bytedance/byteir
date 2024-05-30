import functools
from typing import List

import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv
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


def export_fake_whole_mixtral_model():
    torch._dynamo.config.capture_dynamic_output_shape_ops = True
    mixtral_config = MixtralConfig(attn_implementation="eager")
    bsz = 3
    seqlen = 1
    past_kv_seqlen = 7
    with FakeTensorMode(shape_env=ShapeEnv()):
        mixtral_model = MixtralModel(mixtral_config)
        input_ids = torch.randint(0, mixtral_config.vocab_size, (bsz, seqlen))

        past_key_values = []
        for i in range(mixtral_config.num_hidden_layers):
            past_key_values.append(
                (
                    torch.rand(
                        bsz,
                        mixtral_config.num_key_value_heads,
                        past_kv_seqlen,
                        mixtral_config.hidden_size
                        // mixtral_config.num_attention_heads,
                    ),
                    torch.rand(
                        bsz,
                        mixtral_config.num_key_value_heads,
                        past_kv_seqlen,
                        mixtral_config.hidden_size
                        // mixtral_config.num_attention_heads,
                    ),
                )
            )
        exported_mixtral = torch.export.export(
            mixtral_model,
            (input_ids, None, None, past_key_values),
        )
        import torch_frontend
        module = torch_frontend.compile_dynamo_model(
            exported_mixtral,
            output_type="stablehlo",
        )
        print(module.operation.get_asm())

if __name__ == "__main__":
    export_fake_whole_mixtral_model()
    export_mixtral_decoding_layer()
