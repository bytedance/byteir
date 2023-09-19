# ByteIR Python APIs

The ByteIR python APIs allow users to utilize ByteIR features, and easily integrate ByteIR with external backends.

## Prerequisites

The followings are required for use of AITemplate tuning features.

CUDA version >= 11.8
Python version >= 3.8
A100 or more advanced GPUs

## Example of Use

Add `compiler/build/python_packages/byteir` to `PYTHONPATH`

The following example bypasses ByteIR lowering, and directly executes AITemplate runtime for each subgraph with `__byteir_cat_fusion__` attribute:

```python
python3 -m byteir.dialects.cat.execute --mhlo_path /path/to/mhlo/mlir --backend=ait --bypass-byteir
```

The following example uses ByteIR lowering features, and executes with BRT (ByteIR Runtime)

```python
python3 -m byteir.dialects.cat.execute --mhlo_path /path/to/mhlo/mlir --preprocess --dump_ir
```

