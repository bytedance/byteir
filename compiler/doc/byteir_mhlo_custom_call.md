# ByteIR Additional Ops

ByteIR compiler introduces several coarse-grained ops to improve pattern-matching rewriting during compilation.

ByteIR implements in the way of re-using mhlo custom call op definition with a ByteIR prefix in `call_target_name`, 
instead of defining another new dialect.

ByteIR implements this conversion in frontends, instead of puting it to ByteIR compiler.

## Rationales 
### Need of coarse-grained ops

Introduction of coarse-grained ops can provide several benefits as follows, 
* it simplifies pattern-matching processes during rewriting regardless of optimization or lowering;
* it allows high-level information to be encoded with coase-grained ops, helping optimization;
* it provides intuitive mapping from frontends to IR, helping debuggability;
* it provides flexible control, since coarse-grained ops can be easily decomposed to fine-grained ops, the other way around is much harder.

### Implementation of reusing mhlo custom call

Reusing mhlo custom call with a ByteIR prefix in `call_target_name` can provide several benefits as follows,
* the original IR is still legal and well-defined without introducing additional new dialect or defining new ops in tablegen;
* it provides backward support for all existing passes or pattern-matching, not breaking anything;
* with a proper definition, an unrecognized coarse-grained op can be eaisly mapping to a custom library or be decomposed into fine-grained ops.

### Implementation coarse-grained op conversion in frontends

Implementing coarse-grained op conversion in frontends can provide several benefits as follows,
* it avoids N-to-1 rewriting happening in ByteIR compiler, and putting corresponding rewriting to each own frontend provides much cleaner implementation;
* different frontends might already define their own dialect providing coarse-grained ops, making this conversion trivial and intuitive;
* it isolates effects caused by existing frontends graph optimzations, which might change among differnt versions of each frontends.


## Addtional op definition

A coarse-grained op kind is defined through with a prefix. 

```call_target_name = "byteir.softmax" or "tf.DynamicPartition"```

If an op is generic across frontends, which happen mostly, it uses a `byteir` prefix.
If an op is frontend-specific, it uses a frontend-specific prefix, such as `tf` or `pytorch`.

Further needed infomation for a given coarse-grained op are encoded in a dictionary attribute, called `byteir_attrs`, which includes all named attributes. 

```Op Attribute: byteir_attrs = {approximate = "none"} or byteir_attrs = {} of if none```

### byteir.layer_norm
- Operands:
  - input: Tensor
  - weight: Tensor
  - bias: Tensor
- Attrs
  - epsilon: F64Attr
  - axis: I64ArrayAttr
  - eps_outside_sqrt: Optional\<BoolAttr>
- Results(1 or 3):
  - output: Tensor 
  - mean: Optional\<Tensor>
  - inv_std_dev: Optional\<Tensor>

### byteir.l2_norm
- Operands:
  - input: Tensor
- Attrs
  - epsilon: F64Attr
  - axis: I64ArrayAttr
- Results:
  - output: Tensor

### byteir.softmax
- Operands:
  - input: Tensor
- Attrs
  - axis: I64Attr
- Results:
  - output: Tensor
- Example:
```
%0 = "mhlo.custom_call"(%arg0) {api_version = 1 : i32, backend_config = "", byteir_attrs = {axis = 1 : i64}, call_target_name = "byteir.softmax", called_computations = [], has_side_effect = false} : (tensor<4x64xf32>) -> tensor<4x64xf32>
```

### byteir.log_softmax
- Operands:
  - input: Tensor
- Attrs
  - axis: I64Attr
- Result:
  - output: Tensor
- Example:
```
%0 = "mhlo.custom_call"(%arg0) {api_version = 1 : i32, backend_config = "", byteir_attrs = {axis = 1 : i64}, call_target_name = "byteir.log_softmax", called_computations = [], has_side_effect = false} : (tensor<4x64xf32>) -> tensor<4x64xf32>
```

### byteir.gelu
- Operands:
  - input: Tensor
- Attr:
  - approximate: str
    - none / erf
    - tanh
- Results:
  - output: Tensor
- Example:
```
%0 = "mhlo.custom_call"(%arg0) {api_version = 1 : i32, backend_config = "", byteir_attrs = {approximate = "none"}, call_target_name = "byteir.gelu", called_computations = [], has_side_effect = false} : (tensor<4x64xf32>) -> tensor<4x64xf32>
```

### byteir.arg_max/byteir.arg_min
- Operands:
  - input: Tensor
- Attrs
  - axis: I64Attr
  - keep_dims: BoolAttr
  - select_last_index: BoolAttr
- Results:
  - output: Optional\<Tensor>
  - indices: IntTensor 


### byteir.top_k 
- Operands:
  - input: Tensor
- Attrs
  - k: I64Attr
  - axis: I64ArrayAttr
  - sorted: BoolAttr
- Results:
  - output: Tensor
  - indices: IntTensor

### byteir.erf 
- Operands:
  - input: Tensor
- Results:
  - output: Tensor
- Example:
```
%0 = "mhlo.custom_call"(%arg0) {call_target_name = "byteir.erf", has_side_effect = false} : (tensor<?x64xf32>) -> tensor<?x64xf32>
```

### byteir.one_hot
- Operands:
  - indices: IntTensor
- Attrs:
  - depth: I64Attr
  - axis: I64Attr
  - on_value: AnyAttr
  - off_value: AnyAttr
- Results:
  - output: Tensor (ElementType same as on_value and off_value)

### byteir.quantize
- Operands:
  - input: FloatTensor
  - scale: FloatTensor (rank=0 for per-tensor quantization, or rank=1 for per-channel quantization)
  - zero_point: Int8/Int16/Uint8/Uint16 Tensor (shape same as scale)
- Attrs
  - axis: I64Attr (Optional, required only for per-channel quantization)
- Results:
  - output: Int8/Int16/Uint8/Uint16 Tensor (type same as zero_point)

### byteir.dequantize
- Operands:
  - input: Int8/Int16/Uint8/Uint16 Tensor
  - scale: FloatTensor (rank=0 for per-tensor dequantization, or rank=1 for per-channel dequantization)
  - zero_point: Int8/Int16/Uint8/Uint16 Tensor (shape same as scale, type same as input)
- Attrs
  - axis: I64Attr (Optional, channel axis index, required only for per-channel dequantization)
- Results:
  - output: FloatTensor

### byteir.resize
- Operands:
  - input: Tensor
  - target (scale/size): FloatTensor/IntTensor (respectively)
- Attrs:
  - target_mode: StringAttr
    - `scale`
    - `size`
  - mode: StringAttr
    - `nearest`
    - `linear`
  - coordinate_transformation_mode: StringAttr
    - Denote scale = length_resized / length_original, the transformation can be described as following.

| coordinate_transformation_mode | x_original = |
| ------------------------------ | :----------  |
| `asymmetric` | x_resized / scale |
| `pytorch_half_pixel`| length_resized > 1 ? (x_resized + 0.5) / scale - 0.5 : 0 |
| `half_pixel` | (x_resized + 0.5) / scale - 0.5 |
| `align_corners`| x_resized * (length_original - 1) / (length_resized - 1) |

- Results:
  - output: Tensor

### byteir.rng_uniform
- Operands:
  - low: 0dTensor
  - high: 0dTensor
  - seed: 0dTensor
  - offset: 0dTensor
  - shape: Optional<1dTensor>
- Results:
  - out: Tensor
- Example:
```
// Static Shape Case: out tensor must have static shape
%high = mhlo.constant dense<1.000000e+00> : tensor<f32>
%low = mhlo.constant dense<0.000000e+00> : tensor<f32>
%seed = byre.compute @GetSeed() : tensor<i64>
%offset = byre.compute @NextOffset() : tensor<i64>
%0 = "mhlo.custom_call"(%low, %high, %seed, %offset) {call_target_name = "byteir.rng_uniform", has_side_effect = false} : (tensor<f32>, tensor<f32>, tensor<i64>, tensor<i64>) -> tensor<8x1024x768xf32>
```
```
// Dynamic Shape Case
%high = mhlo.constant dense<1.000000e+00> : tensor<f32>
%low = mhlo.constant dense<0.000000e+00> : tensor<f32>
%seed = byre.compute @GetSeed() : tensor<i64>
%offset = byre.compute @NextOffset() : tensor<i64>
%shape = shape.shape_of %arg0 : tensor<3xindex>
%0 = "mhlo.custom_call"(%low, %high, %seed, %offset, %shape) {call_target_name = "byteir.rng_uniform", has_side_effect = false} : (tensor<f32>, tensor<f32>, tensor<i64>, tensor<i64>, tensor<3xindex>) -> tensor<?x?x?xf32>
```
