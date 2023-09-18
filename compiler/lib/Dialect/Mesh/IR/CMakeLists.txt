add_mlir_dialect_library(MLIRMeshDialect
  MeshOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Dialect/Mesh

  DEPENDS
  MLIRMeshOpsAttrIncGen
  MLIRMeshOpsEnumsIncGen
  MLIRMeshOpsIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRIR
  MLIRSupport
)
