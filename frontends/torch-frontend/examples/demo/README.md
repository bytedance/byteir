# ByteIR GPU Compiler for LLM on Torch 2.0

### Steps to run
1. Use docker with `debian>=11`, `python==3.9`, `cuda>=11.8` or build docker image with [Dockerfile](../../../../docker/Dockerfile).
2. Download ByteIR latest release and unzip it.
3. Install ByteIR components and dependency:
    * python3 -m pip install ByteIR/*.whl
    * cd /path/to/demo
    * python3 -m pip install -r requirements.txt
4. Run training demo:
    * python3 main.py \<model-name\> <--flash>
    * **model-name:** ["gpt2", "bloom-560m", "llama", "opt-1.3b", "nanogpt"]
    * **--flash:** means enable flash attention
5. Run inference demo:
    * python3 main.py \<model-name\> --infer <--flash>
    * **model-name:** ["gpt2", "bloom-560m", "llama", "opt-1.3b", "nanogpt"]
    * **--flash:** means enable flash attention
