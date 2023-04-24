//===- GPUKernelToPTX.cpp -------------------------------------------------===//
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

#include "byteir/Target/PTX/Passes.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Linker/Linker.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Transforms/IPO/Internalize.h"
#include "llvm/Transforms/Scalar/DCE.h"
#include "llvm/Transforms/Vectorize/LoopVectorize.h"
#include "llvm/Transforms/Vectorize/SLPVectorizer.h"

using namespace llvm;
using namespace mlir;

namespace {

// TODO: maybe move another file
static llvm::CodeGenOpt::Level LLVMCodeGenOpt(unsigned optLevel) {
  llvm::CodeGenOpt::Level codegenOptLevel;
  switch (optLevel) {
  case 0:
    codegenOptLevel = llvm::CodeGenOpt::None;
    break;
  case 1:
    codegenOptLevel = llvm::CodeGenOpt::Less;
    break;
  case 2:
    codegenOptLevel = llvm::CodeGenOpt::Default;
    break;
  case 3:
    codegenOptLevel = llvm::CodeGenOpt::Aggressive;
    break;
  default:
    codegenOptLevel = llvm::CodeGenOpt::Aggressive;
    break;
  }
  return codegenOptLevel;
}

static llvm::OptimizationLevel mapToLevel(unsigned optLevel) {
  switch (optLevel) {
  default:
    llvm_unreachable("Invalid optimization level!");
  case 0:
    return llvm::OptimizationLevel::O0;
  case 1:
    return llvm::OptimizationLevel::O1;
  case 2:
    return llvm::OptimizationLevel::O2;
  case 3:
    return llvm::OptimizationLevel::O3;
  }
}

static void addOptimizationPasses(llvm::FunctionPassManager &fPM,
                                  llvm::ModulePassManager &mPM,
                                  llvm::OptimizationLevel &optLevel) {
  fPM.addPass(llvm::VerifierPass()); // Verify that input is correct

  if (optLevel.getSpeedupLevel() > 1 && optLevel.getSizeLevel() < 2) {
    fPM.addPass(llvm::LoopVectorizePass());
    fPM.addPass(llvm::SLPVectorizerPass());
  }

  fPM.addPass(llvm::DCEPass());
  mPM.addPass(llvm::AlwaysInlinerPass());
}

class SerializeToPTX
    : public PassWrapper<SerializeToPTX, OperationPass<gpu::GPUModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SerializeToPTX)

  SerializeToPTX(unsigned opt, const std::string &libdeviceFile,
                 const std::string &triple, const std::string &chip,
                 const std::string &features, std::string &targetISA)
      : optLevelAsInt(opt), libdeviceFile(libdeviceFile), triple(triple),
        chip(chip), features(features), targetISA(targetISA) {}

  void runOnOperation() override;

private:
  /// Creates the LLVM target machine to generate the ISA.
  std::unique_ptr<llvm::TargetMachine> createTargetMachine();

  void translateToISA(llvm::Module &llvmModule,
                      llvm::TargetMachine &targetMachine);

  /// Translates the 'getOperation()' result to an LLVM module.
  std::unique_ptr<llvm::Module>
  translateToLLVMIR(llvm::LLVMContext &llvmContext);

  // Serializes the target ISA to binary form.
  // Disable this for now
  // TODO add this back later
  // std::unique_ptr<std::vector<char>> serializeISA(const std::string& isa);

  LogicalResult linkLibdevice(llvm::Module &llvmModule,
                              llvm::LLVMContext &llvmContext);

  unsigned optLevelAsInt;

  std::string libdeviceFile;

  std::string triple;

  std::string chip;

  std::string features;

  std::string &targetISA;
};

void SerializeToPTX::runOnOperation() {
  // Lower the module to an LLVM IR module using a separate context to enable
  // multi-threaded processing.
  llvm::LLVMContext llvmContext;
  std::unique_ptr<llvm::Module> llvmModule = translateToLLVMIR(llvmContext);

  if (!llvmModule) {
    return signalPassFailure();
  }

  if (failed(linkLibdevice(*llvmModule, llvmContext))) {
    return signalPassFailure();
  }

  std::unique_ptr<llvm::TargetMachine> targetMachine = createTargetMachine();
  if (!targetMachine)
    return signalPassFailure();

  translateToISA(*llvmModule, *targetMachine);
}

std::unique_ptr<llvm::TargetMachine> SerializeToPTX::createTargetMachine() {
  Location loc = getOperation().getLoc();
  std::string error;
  const llvm::Target *target =
      llvm::TargetRegistry::lookupTarget(triple, error);

  if (!target) {
    emitError(loc, Twine("failed to lookup target: ") + error);
    return {};
  }

  llvm::TargetMachine *machine =
      target->createTargetMachine(triple, chip, features, {}, {}, std::nullopt,
                                  LLVMCodeGenOpt(optLevelAsInt));
  if (!machine) {
    emitError(loc, "failed to create target machine");
    return {};
  }

  return std::unique_ptr<llvm::TargetMachine>{machine};
}

void SerializeToPTX::translateToISA(llvm::Module &llvmModule,
                                    llvm::TargetMachine &targetMachine) {

  llvmModule.setDataLayout(targetMachine.createDataLayout());

  llvm::raw_string_ostream stream(targetISA);
  llvm::buffer_ostream pstream(stream);

  // new
  llvm::OptimizationLevel optLevel = mapToLevel(optLevelAsInt);
  llvm::PassBuilder pB(&targetMachine);

  targetMachine.registerPassBuilderCallbacks(pB);

  // Register all basic analyses
  llvm::LoopAnalysisManager lAM;
  llvm::FunctionAnalysisManager fAM;
  llvm::CGSCCAnalysisManager cGAM;
  llvm::ModuleAnalysisManager mAM;

  pB.registerModuleAnalyses(mAM);
  pB.registerCGSCCAnalyses(cGAM);
  pB.registerFunctionAnalyses(fAM);
  pB.registerLoopAnalyses(lAM);
  pB.crossRegisterProxies(lAM, fAM, cGAM, mAM);

  llvm::FunctionPassManager fPM = pB.buildFunctionSimplificationPipeline(
      optLevel, llvm::ThinOrFullLTOPhase::None);
  llvm::ModulePassManager mPM = pB.buildPerModuleDefaultPipeline(optLevel);

  fAM.registerPass([&] { return targetMachine.getTargetIRAnalysis(); });

  addOptimizationPasses(fPM, mPM, optLevel);

  mPM.run(llvmModule, mAM);
  for (auto &f : llvmModule) {
    if (!f.empty()) {
      fPM.run(f, fAM);
    }
  }

  llvm::legacy::PassManager codegenPasses;
  codegenPasses.add(llvm::createVerifierPass());
  targetMachine.addPassesToEmitFile(codegenPasses, pstream, nullptr,
                                    llvm::CGFT_AssemblyFile);
  codegenPasses.run(llvmModule);
}

std::unique_ptr<llvm::Module>
SerializeToPTX::translateToLLVMIR(llvm::LLVMContext &llvmContext) {
  return translateModuleToLLVMIR(getOperation(), llvmContext,
                                 "LLVMDialectModule");
}

LogicalResult SerializeToPTX::linkLibdevice(llvm::Module &llvmModule,
                                            llvm::LLVMContext &llvmContext) {
  if (libdeviceFile == "") {
    llvm::errs() << "Fatal: unable to locate libdevice.10.bc\n";
    return failure();
  }
  std::string errorMessage;
  auto libdeviceBuf = openInputFile(libdeviceFile, &errorMessage);
  if (!libdeviceBuf) {
    llvm::errs() << errorMessage << "\n";
    return failure();
  }

  auto moduleOrErr =
      llvm::getOwningLazyBitcodeModule(std::move(libdeviceBuf), llvmContext);
  if (!moduleOrErr) {
    llvm::errs() << "Failed to load libdevice bitcode from " << libdeviceFile
                 << "\n";
    return failure();
  }

  std::unique_ptr<llvm::Module> libdeviceModule = std::move(moduleOrErr.get());
  // Setup the same function attributes as those used by compiling a cuda
  // code with ``clang -O3''. The default function attributes are retrieved
  // based on the values from CodeGenModule::getDefaultFunctionAttributes
  for (llvm::Function &F : *libdeviceModule.get()) {

    // intrinsic not use attr
    if (F.isIntrinsic()) {
      continue;
    }

    llvm::AttrBuilder FuncAttrs(llvmContext);
    FuncAttrs.addAttribute("frame-pointer", /*FramePointerKind*/ "all");
    FuncAttrs.addAttribute("less-precise-fpmad", "false");
    FuncAttrs.addAttribute("no-trapping-math", "true");

    // comment out the following due to invalid
    FuncAttrs.addAttribute("no-infs-fp-math", "false");
    FuncAttrs.addAttribute("no-nans-fp-math", "false");
    FuncAttrs.addAttribute("unsafe-fp-math", "false");
    FuncAttrs.addAttribute("use-soft-float", "false");
    FuncAttrs.addAttribute("stack-protector-buffer-size", "8");
    FuncAttrs.addAttribute("no-signed-zeros-fp-math", "false");

    FuncAttrs.addAttribute(llvm::Attribute::Convergent);
    // no exceptions for cuda device code
    FuncAttrs.addAttribute(llvm::Attribute::NoUnwind);

    F.addFnAttrs(FuncAttrs);
  }

  // libdevice module is of an ``internalize'' module
  if (llvm::Linker::linkModules(
          llvmModule, std::move(libdeviceModule),
          /*LinkFlags*/ llvm::Linker::Flags::LinkOnlyNeeded,
          [](llvm::Module &M, const llvm::StringSet<> &GS) {
            llvm::internalizeModule(M, [&GS](const llvm::GlobalValue &GV) {
              return !GV.hasName() || (GS.count(GV.getName()) == 0);
            });
          })) {
    llvm::errs() << "failed to link libdevice module\n";
    return failure();
  }

  return success();
}

} // namespace

std::unique_ptr<OperationPass<gpu::GPUModuleOp>> mlir::createSerializeToPTXPass(
    unsigned optLevel, const std::string &libdeviceFile,
    const std::string &triple, const std::string &chip,
    const std::string &features, std::string &targetISA) {

  return std::make_unique<SerializeToPTX>(optLevel, libdeviceFile, triple, chip,
                                          features, targetISA);
}
