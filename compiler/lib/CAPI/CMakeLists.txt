get_property(byteir_dialect_libs GLOBAL PROPERTY BYTEIR_DIALECT_LIBS)
get_property(byteir_pipeline_libs GLOBAL PROPERTY BYTEIR_PIPELINE_LIBS)
get_property(byteir_conversion_libs GLOBAL PROPERTY BYTEIR_CONVERSION_LIBS)
get_property(byteir_translation_libs GLOBAL PROPERTY BYTEIR_TRANSLATION_LIBS)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

add_mlir_public_c_api_library(ByteIRCAPI
  Dialects.cpp
  Passes.cpp
  Translation.cpp
  PDLValue.cpp

  LINK_LIBS PUBLIC
  ${byteir_dialect_libs}
  ${byteir_pipeline_libs}
  ${byteir_conversion_libs}
  ${byteir_translation_libs}
  ${dialect_libs}
  ${extension_libs}

  # transofrm passes
  ByteIRTransforms

  # translate passes
  MLIRTranslateLib
)
