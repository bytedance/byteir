# ByteIR GPU Compiler for LLM on Torch 2.0

### Steps to run
1. Build docker image with [Dockerfile](../../../../docker/Dockerfile).
2. Download ByteIR release and unzip it.
3. Install ByteIR components:
    * python3 -m pip install -r ByteIR/requirements.txt
    * python3 -m pip install ByteIR/*.whl
4. Run training demo:
    * python3 main.py \<model-name\> <--flash>
    * **model-name:** ["gpt2", "bloom-560m", "llama", "opt-1.3b", "nanogpt"]
    * **--flash:** means enable flash attention
5. Run inference demo:
    * python3 main.py \<model-name\> --infer <--flash>
    * **model-name:** ["llama"]
    * **--flash:** means enable flash attention
