add_mlir_public_c_api_library(TorchFrontendCAPI
    Passes.cpp

    ENABLE_AGGREGATION
    LINK_COMPONENTS

    LINK_LIBS PUBLIC
    MLIRSupport
    TorchFrontendPipelines
)
