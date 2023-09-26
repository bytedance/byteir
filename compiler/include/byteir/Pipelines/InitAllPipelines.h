//===- InitAllPipelines.h -------------------------------------*--- C++ -*-===//
//
// Copyright 2022 ByteDance Ltd. and/or its affiliates. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_PIPELINES_INITALLPIPELINES_H
#define BYTEIR_PIPELINES_INITALLPIPELINES_H

#include "byteir/Pipelines/AffineOpt.h"
#include "byteir/Pipelines/AllOpt.h"
#include "byteir/Pipelines/BufferizeOpt.h"
#include "byteir/Pipelines/ByreHost.h"
#include "byteir/Pipelines/ByreOpt.h"
#include "byteir/Pipelines/ByreTensorOpt.h"
#include "byteir/Pipelines/CatOpt.h"
#include "byteir/Pipelines/CatPreprocess.h"
#include "byteir/Pipelines/HloOpt.h"
#include "byteir/Pipelines/LinalgMemrefOpt.h"
#include "byteir/Pipelines/LinalgTensorOpt.h"
#include "byteir/Pipelines/SCFOpt.h"
#include "byteir/Pipelines/ShapeOpt.h"

#include "byteir/Pipelines/GPU/ElementwiseCodegen.h"
#include "byteir/Pipelines/GPU/GPUOpt.h"
#include "byteir/Pipelines/GPU/LinalgMemrefGPU.h"
#include "byteir/Pipelines/GPU/MappingForall.h"
#include "byteir/Pipelines/GPU/NVVMCodegen.h"
#include "byteir/Pipelines/GPU/ReductionCodegen.h"

#include "byteir/Pipelines/Host/Codegen.h"
#include "byteir/Pipelines/Host/HostOpt.h"
#include "byteir/Pipelines/Host/ToLLVM.h"

namespace mlir {

inline void registerAllByteIRCommonPipelines() {
  registerAffineOptPipeline();
  registerByreHostPipeline();
  registerByreOptPipeline();
  registerByreTensorOptPipeline();
  registerCatOptPipeline();
  registerCatPreprocessPipeline();
  registerHloOptPipeline();
  registerLinalgMemrefOptPipeline();
  registerLinalgTensorOptPipeline();
  registerSCFOptPipeline();
  registerShapeOptPipeline();
  registerByteIRBufferizeOptPipeline();
  registerByteIRAllOptPipeline();
}

inline void registerAllByteIRGPUPipelines() {
  registerGPUOptPipeline();
  registerNVVMCodegenPipeline();
  registerLinalgMemrefGPUPipeline();
  registerMatmulEpilogueGPUPipeline();
  registerGPUElementwiseCodegenPipelines();
  registerGPUReductionCodegenPipelines();
  registerGPUMappingForallPipelines();
}

inline void registerAllByteIRHostPipelines() {
  registerHostOptPipeline();
  registerToLLVMPipeline();
  registerHostCodegenPipelines();
}

} // namespace mlir

#endif // BYTEIR_PIPELINES_INITALLPIPELINES_H
