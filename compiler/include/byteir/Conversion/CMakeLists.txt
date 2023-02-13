set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name ByteIRConversion)
add_public_tablegen_target(ByteIRConversionPassIncGen)
# add_mlir_doc(Passes ConversionPasses ./ -gen-pass-doc)