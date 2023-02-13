// RUN: byteir-opt -allow-unregistered-dialect %s | FileCheck %s

// batch = 8, seq = 128, hidden = 512
module attributes {byre.container_module} {
  func.func @bert_trasnformer_test(
    %attr_kernel_q : memref<512x512xf16> {byre.argname = "attr_kernel_q", byre.argtype = 4 : i32},
	%attr_kernel_k : memref<512x512xf16> {byre.argname = "attr_kernel_k", byre.argtype = 4 : i32}, 
	%attr_kernel_v : memref<512x512xf16> {byre.argname = "attr_kernel_v", byre.argtype = 4 : i32}, 
    %attr_bias_q : memref<512xf16> {byre.argname = "attr_bias_q", byre.argtype = 4 : i32}, 
	%attr_bias_k : memref<512xf16> {byre.argname = "attr_bias_k", byre.argtype = 4 : i32}, 
	%attr_bias_v : memref<512xf16> {byre.argname = "attr_bias_v", byre.argtype = 4 : i32}, 
	%attr_output_kernel : memref<512x512xf16> {byre.argname = "attr_output_kernel", byre.argtype = 4 : i32},
    %attr_output_bias : memref<512xf16> {byre.argname = "attr_output_bias", byre.argtype = 4 : i32},
    %attr_output_layernorm_beta : memref<512xf16> {byre.argname = "attr_output_layernorm_beta", byre.argtype = 4 : i32},
    %attr_output_layernorm_gamma : memref<512xf16> {byre.argname = "attr_output_layernorm_gamma", byre.argtype = 4 : i32},
	%inter_kernel : memref<512x2048xf16> {byre.argname = "inter_kernel", byre.argtype = 4 : i32},
	%inter_bias : memref<2048xf16> {byre.argname = "inter_bias", byre.argtype = 4 : i32},
	%output_kernel : memref<2048x512xf16> {byre.argname = "output_kernel", byre.argtype = 4 : i32},
	%output_bias : memref<512xf16> {byre.argname = "output_bias", byre.argtype = 4 : i32},
	%output_layernorm_beta : memref<512xf16> {byre.argname = "output_layernorm_beta", byre.argtype = 4 : i32},
	%output_layernorm_gamma : memref<512xf16> {byre.argname = "output_layernorm_gamma", byre.argtype = 4 : i32},
    %input_tensor : memref<8x128x512xf16> {byre.argname = "input_tensor", byre.argtype = 1 : i32}, 
	%atten_mask : memref<8x128x128xf16> {byre.argname = "atten_mask", byre.argtype = 1 : i32},
	%output : memref<8x128x512xf16> {byre.argname = "output", byre.argtype = 2 : i32}) attributes {byre.entry_point} {

    byre.compute @BertTransformerOp(%attr_kernel_q, %attr_kernel_k, %attr_kernel_v, 
	                                %attr_bias_q, %attr_bias_k, %attr_bias_v,
									%attr_output_kernel, %attr_output_bias,
									%attr_output_layernorm_beta, %attr_output_layernorm_gamma,
									%inter_kernel, %inter_bias,
									%output_kernel, %output_bias,
									%output_layernorm_beta, %output_layernorm_gamma,
									%input_tensor, %atten_mask,
									%output) {head_num = 32 : i64, size_per_head = 16 : i64} : memref<512x512xf16>, memref<512x512xf16>, memref<512x512xf16>,  // attr_kernel
									           memref<512xf16>, memref<512xf16>, memref<512xf16>,              // attr_bias
											   memref<512x512xf16>, memref<512xf16>,                           // attr_output
											   memref<512xf16>, memref<512xf16>,                               // attr_output_layernorm
											   memref<512x2048xf16>, memref<2048xf16>,                         // inter_kernel
											   memref<2048x512xf16>,  memref<512xf16>,                         // output_kernel    
											   memref<512xf16>, memref<512xf16>,                               // output_layernorm
											   memref<8x128x512xf16>, memref<8x128x128xf16>,                   // input, mask
											   memref<8x128x512xf16>                                           // output
    return
  }
  
  // CHECK-LABEL: func.func @bert_trasnformer_test
}
