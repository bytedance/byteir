# TF-FRONTEND

tf-frontend is a project to build customized tf graph --> tf dialect --> mhlo dialect pipeline, where we could add extended dialect and passes.

## Quick Start

### Build from source code
```
git clone https://github.com/bytedance/byteir.git
cd byteir
./frontends/tf-frontend/scripts/prepare.sh
cd frontends/tf-frontend

# build:
./bazel build //tools:tf-frontend //tools:tf-ext-opt
# test:
export TF_PYTHON_VERSION=3.9 # change this to python version which installed lit
./bazel test //tf_mlir_ext/tests:all
```

### Example
Add the folder's path of built tf-frontend executable binary to environment variable `PATH`.
```
pip3 install tensorflow
cd byteir/frontends/tf-frontend/example
python3 resnet.py
tf-frontend .workspace/resnet50/model.pb -tf-output-arrays=res_net50/fc1000/BiasAdd  -o resnet_mhlo.mlir
```
For more options, please refer to `tf-frontend --help`.

## Contributing

Please refer to [Developer Guild](./docs/developer_guild.md).
