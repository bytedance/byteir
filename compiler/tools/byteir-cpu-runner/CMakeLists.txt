set(LLVM_LINK_COMPONENTS
  Core
  Support
  nativecodegen
  native
)

add_mlir_tool(byteir-cpu-runner
  byteir-cpu-runner.cpp
)

llvm_update_compile_flags(byteir-cpu-runner)
target_link_libraries(byteir-cpu-runner PRIVATE
  MLIRAnalysis
  MLIRExecutionEngine
  MLIRIR
  MLIRJitRunner
  MLIRLLVMDialect
  MLIRLLVMToLLVMIRTranslation
  MLIRToLLVMIRTranslationRegistration
  MLIRParser
  MLIRTargetLLVMIRExport
  MLIRSupport
)
