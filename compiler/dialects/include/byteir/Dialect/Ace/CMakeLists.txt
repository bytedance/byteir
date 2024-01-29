add_mlir_dialect(AceOps ace)
add_mlir_doc(AceOps AceOps Dialects/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS AceOps.td)
mlir_tablegen(AceOpsAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(AceOpsAttributes.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRAceOpsAttrIncGen)
