add_byteir_conversion_library(ByteIRHloToTensor
    ConvertHloToTensor.cpp

    ADDITIONAL_HEADER_DIRS
    ${BYTEIR_SRC_INCLUDE_DIR}/byteir/Conversion/HloToTensor
    
    DEPENDS
    MLIRTensorDialect
    ByteIRConversionPassIncGen
    ByteIRConversionCommon
    ByteIRMhloUtils

    LINK_LIBS PUBLIC
    MhloDialect
    MLIRAceDialect
    MLIRArithDialect
    MLIRIR
    MLIRTensorDialect
    MLIRTransforms
    ByteIRConversionCommon
    ByteIRMhloUtils
)
