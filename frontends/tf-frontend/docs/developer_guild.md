# Contribute to TF-Frontend

Here's some infomation which might be useful for developers.

## File Structure
1. Upstream tensorflow is a submodule of this repo lying on `external/tensorflow`, it will be built together and some simple patches will be applied on tensorflow before compiling.
2. Main source code:
    - tf_mlir_ext: used as an extension of the upstream tensorflow dialect.
        - byteir/: Contains source code shared with byteir compiler. Most of them are symbolic link files.
        - ir/: Used to define extented operations of tensorflow dialect. Here TF.OpaqueOp is only used as an example to demostrate how to add tf operations outside the upstream tf repo. Note that we should add the extented operation when I have to, and it also need to be rationalized before adding it.
        - transforms/: additional passes on tf dialect
        - tests/: lit tests for the additional passes
        - pipelines/: a pass pipeline contructed to convert tf dialect to mhlo dialect
    - tools: 
        - tf_frontend_main.cc: the main executable binary will be built from this, it is used convert a tf model of protobuf format to mhlo IR.
        - tf_ext_opt_main.cc: an debug targeted executable binary, always used to test a pass or some passes.
    - utils: some non-mlir utilities

## Run Lit Test

```shell
bazel --output_user_root=./build test //tf_mlir_ext/tests:all --java_runtime_version=remotejdk_11
```

## How to Debug
The most frequent errors occur during the tf-frontend pass pipeline. If that happens, a typical debuging method is to add `-reproduce-file [path/to/file]` to the tf-frontend command. Then we could know which pass cause the error and the IR before that pass. After that, tf-ext-opt is recommended to be used to debug the specific pass. For more debugging tips, please refer to [mlir doc ](https://mlir.llvm.org/getting_started/Debugging/) or by typing `tf-frontend --help`.
