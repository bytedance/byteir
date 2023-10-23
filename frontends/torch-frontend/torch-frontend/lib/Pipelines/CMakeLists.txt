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
)
