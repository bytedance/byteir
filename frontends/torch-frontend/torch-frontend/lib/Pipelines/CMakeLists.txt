# note: remove this when upstream fixed
target_link_libraries(TorchMLIRTorchConversionPasses PUBLIC TorchMLIRTorchToTensor TorchMLIRTorchToStablehlo)

add_mlir_library(TorchFrontendPipelines
    Pipelines.cpp

    # DEPENDS

    LINK_COMPONENTS
    Core

    LINK_LIBS PUBLIC
    MLIRIR
    TorchMLIRTorchToStablehlo
    TorchMLIRTorchConversionPasses
    TorchFrontendConversion
    TorchFrontendTransforms
    StablehloPasses
)
