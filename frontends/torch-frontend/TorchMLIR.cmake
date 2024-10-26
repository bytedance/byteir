#-------------------------------------------------------------------------------
# Build Torch-MLIR
#-------------------------------------------------------------------------------

message(STATUS "Build Torch-MLIR with MLIR: ${MLIR_DIR}")
message(STATUS "Build Torch-MLIR with LLVM: ${LLVM_DIR}")

function(build_torch_mlir)
  set(TORCH_MLIR_SRC_PATH "${TORCH_FRONTEND_SRC_ROOT}/third_party/torch-mlir")
  set(TORCH_MLIR_BUILD_PATH "${TORCH_FRONTEND_BIN_ROOT}/torch_mlir_build")
  execute_process(COMMAND ${CMAKE_COMMAND}
    -S "${TORCH_MLIR_SRC_PATH}"
    -B "${TORCH_MLIR_BUILD_PATH}"
    -G "${CMAKE_GENERATOR}"
    -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
    -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
    -DCMAKE_BUILD_TYPE=Release
    -DLLVM_DIR=${LLVM_DIR}
    -DMLIR_DIR=${MLIR_DIR}
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON
    -DLLVM_ENABLE_ZSTD=OFF
    -DCMAKE_VISIBILITY_INLINES_HIDDEN=ON
    -DCMAKE_C_VISIBILITY_PRESET=hidden
    -DCMAKE_CXX_VISIBILITY_PRESET=hidden
    -DTORCH_MLIR_ENABLE_REFBACKEND=OFF
    -DTORCH_MLIR_ENABLE_LTC=OFF
    -DTORCH_MLIR_ENABLE_PYTORCH_EXTENSIONS=${TORCH_FRONTEND_ENABLE_JIT_IR_IMPORTER}
    -DTORCH_MLIR_ENABLE_JIT_IR_IMPORTER=${TORCH_FRONTEND_ENABLE_JIT_IR_IMPORTER}
    -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
    -DPython3_EXECUTABLE=${Python3_EXECUTABLE}
    -DLLVM_EXTERNAL_LIT=${LLVM_EXTERNAL_LIT}

    RESULT_VARIABLE result
    WORKING_DIRECTORY ${TORCH_MLIR_SRC_PATH}
  )

  if(result)
    message(FATAL_ERROR "CMake step for torch-mlir failed: ${result}")
  endif()

  execute_process(
    COMMAND ${CMAKE_COMMAND} --build ${TORCH_MLIR_BUILD_PATH} --target TorchMLIRPythonModules TorchMLIRJITIRImporterPybind
    RESULT_VARIABLE result
    WORKING_DIRECTORY ${TORCH_MLIR_SRC_PATH}
  )

  if(result)
    message(FATAL_ERROR "Build step for torch-mlir failed: ${result}")
  endif()
endfunction(build_torch_mlir)

build_torch_mlir()
