# ByteIR RNG CustomCall Converter

## RNG Uniform Converter

We use Philox4×32-10 and Normalization to implement Uniform Distribution Generator for floating-point value.

Philox4×32-10 is a counter-based random number generator that takes a 4 32-bit integer and 2 32-bit keys as input and produces 4 32-bit uniformly distributed random **integers** as output. The pseudocode is
```python
const1 = 0x9E3779B9
const2 = 0xBB67AE85
const3 = 0xD2511F53
const4 = 0xCD9E8D57

def round(counter, key):  
    l0 = (counter[0] * const3) % (1 << 32)
    h0 = (counter[0] * const3) >> 32
  
    l1 = (counter[2] * const4) % (1 << 32)
    h1 = (counter[2] * const4) >> 32
  
    return [h1 ^ counter[1] ^ key[0], l1, h0 ^ counter[3] ^ key[1], l0]

def Philox4×32-10():
  counter = [c_0,c_1,c_2,c_3]
  key = [k_0,k_1]

  # will be unrolled
  for i in range(10):
    counter = round(counter,key)
    key[0] += const1
    key[1] += const2

  return counter
```

Normalization will take a uniformly distributed W-bit integer $x$ and convert it into a uniformly distributed floating point number $y$ in $(0, 1]$. The formulation is 
$$y = x*2^{-W}+2^{-W-1}$$

So firstly, using Philox4×32-10 we can get random integers, and then normalization will be taken to convert random integers to floating-point value. The process of lowering the rng.Uniform from MHLO to linalg will be shown below.

### Example of lowering the Rng.Uniform from MHLO to Linalg
```
func.func private @Unknown0(%arg0: tensor<i64>, %arg1: tensor<i64>) -> tensor<25600xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %1 = mhlo.constant dense<1.000000e+01> : tensor<f32>
    %2 = mhlo.custom_call @byteir.rng_uniform(%0, %1, %arg0, %arg1) {backend_config = ""} : (tensor<f32>, tensor<f32>, tensor<i64>, tensor<i64>) -> tensor<25600xf32>
    return %2 : tensor<25600xf32>
}

// IR after lowering to Linalg

func.func private @Unknown0(%arg0: tensor<i64>, %arg1: tensor<i64>) -> tensor<25600xf32> attributes {__byteir_elementwise_fusion__} {
    // Some definitions of constant variable
    %0 = tensor.empty() : tensor<25600xf32>
    %1 = scf.for %arg2 = %c0 to %c25600 step %c1 iter_args(%arg3 = %0) -> (tensor<25600xf32>) {
      %2 = tensor.empty() : tensor<f32>
      %3 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []} ins(%arg0, %arg1 : tensor<i64>, tensor<i64>) outs(%2 : tensor<f32>) {
      ^bb0(%in: i64, %in_3: i64, %out: f32):
        // compute a unique counter with index to generate uniform distributed interger
        %4 = arith.trunci %in : i64 to i32
        %5 = arith.trunci %in_3 : i64 to i32
        %6 = arith.addi %4, %5 : i32
        %7 = arith.index_cast %arg2 : index to i32
        %8 = arith.addi %7, %6 : i32
        %9 = arith.muli %8, %c1103515245_i32 : i32
        %10 = arith.addi %9, %c12345_i32 : i32
        %11 = arith.extui %10 : i32 to i64
        %12 = arith.muli %11, %c3528531795_i64 : i64
        %13 = arith.trunci %12 : i64 to i32
        %14 = arith.shrui %12, %c32_i64 : i64
        %15 = arith.trunci %14 : i64 to i32
        %16 = arith.xori %15, %5 : i32
        
        // unrolled loop to implement Philox4×32-10
        %17 = arith.addi %4, %c-1640531527_i32 : i32
        %18 = arith.addi %5, %c-1150833019_i32 : i32
        %19 = arith.extui %4 : i32 to i64
        %20 = arith.extui %16 : i32 to i64
        %21 = arith.muli %19, %c3528531795_i64 : i64
        %22 = arith.trunci %21 : i64 to i32
        %23 = arith.shrui %21, %c32_i64 : i64
        %24 = arith.trunci %23 : i64 to i32
        %25 = arith.muli %20, %c3449720151_i64 : i64
        %26 = arith.trunci %25 : i64 to i32
        %27 = arith.shrui %25, %c32_i64 : i64
        %28 = arith.trunci %27 : i64 to i32
        %29 = arith.xori %28, %17 : i32
        %30 = arith.xori %24, %13 : i32
        %31 = arith.xori %30, %18 : i32
        ......
        %137 = arith.addi %4, %c-1879881855_i32 : i32
        %138 = arith.extui %136 : i32 to i64
        %139 = arith.muli %138, %c3449720151_i64 : i64
        %140 = arith.shrui %139, %c32_i64 : i64
        %141 = arith.trunci %140 : i64 to i32
        %142 = arith.xori %141, %134 : i32
        %143 = arith.xori %142, %137 : i32

        // Normalization for convert integer to floating-point value
        %144 = arith.uitofp %143 : i32 to f32
        %145 = arith.mulf %144, %cst_2 : f32
        %146 = arith.addf %145, %cst_1 : f32
        %147 = arith.mulf %146, %cst_0 : f32
        %148 = arith.addf %147, %cst : f32
        linalg.yield %148 : f32
      } -> tensor<f32>
      %inserted_slice = tensor.insert_slice %3 into %arg3[%arg2] [1] [1] : tensor<f32> into tensor<25600xf32>
      scf.yield %inserted_slice : tensor<25600xf32>
    }
    return %1 : tensor<25600xf32>
}

```

### Future work
In order to ensure that it can be fused with other elementwise ops, for each Philox4×32-10, only the first output will be taken, and the remaining three outputs will be discarded. This could potentially be optimized to enhance performance

## RNG Normal Converter

We use Box–Muller transform and Philox4×32-10 to implemet Normal Distribution Generator. Box–Muller transform takes two samples $x_1$ $x_2$ from the uniform distribution on the interval (0,1) and maps them to two standard, normally distributed samples $y_1$ and $y_2$. The formulation is
$$y_1=\cos(2 \pi x_1)*\sqrt{(-2\ln x_2)}$$
$$y_2=\sin(2 \pi x_1)*\sqrt{(-2\ln x_2)}$$
So fisrt of all, We use the Philox4×32-10 and Normalization to generate two uniformly distributed random numbers whose range from 0 to 1 (just like the Uniform Distribution Generator, but this time we need to take two of the four Philox4×32-10 outputs and discard the remaining two). And then use Box–Muller transform to map them to standard, normally distributed values. The process of lowering the rng.Normal from MHLO to linalg will be shown below.

### Example of lowering the Rng.Normal from MHLO to Linalg
```
func.func private @Unknown0(%arg0: tensor<i64>, %arg1: tensor<i64>) -> tensor<3x256xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %1 = mhlo.constant dense<1.000000e+01> : tensor<f32>
    %2 = mhlo.custom_call @byteir.rng_normal(%0, %1, %arg0, %arg1) {backend_config = ""} : (tensor<f32>, tensor<f32>, tensor<i64>, tensor<i64>) -> tensor<3x256xf32>
    return %2 : tensor<3x256xf32>
  }

// IR after linalg-tensor-opt

func.func private @Unknown0(%arg0: tensor<i64>, %arg1: tensor<i64>) -> tensor<3x256xf32> attributes {__byteir_elementwise_fusion__} {
    // Some definitions of constant variable
    %0 = tensor.empty() : tensor<3x256xf32>
    %1 = scf.for %arg2 = %c0 to %c3 step %c1 iter_args(%arg3 = %0) -> (tensor<3x256xf32>) {
      %2 = scf.for %arg4 = %c0 to %c256 step %c1 iter_args(%arg5 = %arg3) -> (tensor<3x256xf32>) {
        %3 = tensor.empty() : tensor<f32>
        %4 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []} ins(%arg0, %arg1 : tensor<i64>, tensor<i64>) outs(%3 : tensor<f32>) {
        ^bb0(%in: i64, %in_5: i64, %out: f32):
          // run Philox4×32-10 and Normalization to generate two samples
          // %152 and %155 from the uniform distribution on the interval (0,1)
          %156 = math.log %152 : f32
          %157 = arith.mulf %156, %cst_2 : f32
          %158 = math.sqrt %157 : f32
          %159 = arith.mulf %155, %cst : f32
          %160 = math.cos %159 : f32
          %161 = arith.mulf %158, %160 : f32
          %162 = arith.mulf %161, %cst_1 : f32
          %163 = arith.addf %162, %cst_0 : f32
          linalg.yield %163 : f32
        } -> tensor<f32>
        %inserted_slice = tensor.insert_slice %4 into %arg5[%arg2, %arg4] [1, 1] [1, 1] : tensor<f32> into tensor<3x256xf32>
        scf.yield %inserted_slice : tensor<3x256xf32>
      }
      scf.yield %2 : tensor<3x256xf32>
    }
    return %1 : tensor<3x256xf32>
  }
```

### Future Work
Just like Rng.Uniform converter, we generate a single random number during each iteration. So for generating a normal distributed random numbers, We need to perform a philox4×32-10 and an Box-Muller transform once. But if we can allow multiple random numbers to be generated during each iteration, since one execution of philox4×32-10 can generate 4 uniformly distributed random numbers, we can generate 6 pairs of inputs to the Box-Muller transform and generate 12 normal distributed random numbers with one philox4×32-10. It can greatly reduce the number of executions of philox4×32-10.
