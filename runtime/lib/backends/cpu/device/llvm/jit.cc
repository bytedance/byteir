//===- jit.cc -------------------------------------------------*--- C++ -*-===//
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

#include "brt/backends/cpu/device/llvm/jit.h"
#include "brt/core/common/common.h"
#include "brt/core/ir/engine_util.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/ExecutionEngine/JITEventListener.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/ObjectTransformLayer.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/Orc/Shared/ExecutorAddress.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Passes/OptimizationLevel.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"

#include <optional>
#include <unordered_set>

#ifndef _WIN32
#include <dlfcn.h>
#endif

namespace brt {
namespace cpu {
namespace {
constexpr static llvm::StringRef llvmJittedObjbufferSuffix =
    "-jitted-objectbuffer";

inline std::string errorToString(llvm::Error err) {
  char *errMsg = LLVMGetErrorMessage(llvm::wrap(std::move(err)));
  std::string ret(errMsg);
  LLVMDisposeErrorMessage(errMsg);
  return ret;
}

inline Status LLVMErrorToBRTStatus(llvm::Error err,
                                   const char *errMsg = nullptr) {
  if (err) {
    return common::Status(common::StatusCategory::BRT, common::StatusCode::FAIL,
                          errMsg ? errMsg + errorToString(std::move(err))
                                 : errorToString(std::move(err)));
  }
  return Status::OK();
}

template <typename T>
T checkAndThrow(llvm::Expected<T> ValOrErr, const char *errMsg = nullptr) {
  if (ValOrErr)
    return std::move(*ValOrErr);

  else {
    if (!errMsg)
      errMsg = "failed on checkAndThrow";

    std::string buf;
    llvm::raw_string_ostream OS(buf);
    OS << errMsg << " : " << errorToString(ValOrErr.takeError());
    BRT_THROW(OS.str().c_str());
  }
}

inline std::optional<llvm::OptimizationLevel>
getLLVMOptimizationLevel(int optLevel) {
  switch (optLevel) {
  case 0:
    return llvm::OptimizationLevel::O0;
  case 1:
    return llvm::OptimizationLevel::O1;
  case 2:
    return llvm::OptimizationLevel::O2;
  case 3:
    return llvm::OptimizationLevel::O3;
  default:
    // unknown opt level, return std::nullopt
    return std::nullopt;
  }
}

inline void runLLVMDefaultOptimizationPipeline(llvm::Module &M,
                                               llvm::OptimizationLevel optLevel,
                                               llvm::TargetMachine *TM) {
  // Follow llvm tutorial to run default optimization pipeline
  llvm::LoopAnalysisManager LAM;
  llvm::FunctionAnalysisManager FAM;
  llvm::CGSCCAnalysisManager CGAM;
  llvm::ModuleAnalysisManager MAM;

  llvm::PassBuilder PB(TM);
  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  llvm::ModulePassManager MPM = optLevel != llvm::OptimizationLevel::O0
                                    ? PB.buildPerModuleDefaultPipeline(optLevel)
                                    : PB.buildO0DefaultPipeline(optLevel);

  MPM.run(M, MAM);
}

llvm::Expected<std::unique_ptr<llvm::orc::ObjectLayer>>
createRTDyldObjectLinkingLayerWithGDBListner(llvm::orc::ExecutionSession &ES,
                                             const llvm::Triple &TT) {
  auto GetMemMgr = []() {
    return std::make_unique<llvm::SectionMemoryManager>();
  };
  auto Layer = std::make_unique<llvm::orc::RTDyldObjectLinkingLayer>(
      ES, std::move(GetMemMgr));

  Layer->setProcessAllSections(true);
  Layer->registerJITEventListener(
      *llvm::JITEventListener::createGDBRegistrationListener());

  return std::unique_ptr<llvm::orc::ObjectLayer>(std::move(Layer));
}

llvm::Expected<std::unique_ptr<llvm::orc::ObjectLayer>>
createRTDyldObjectLinkingLayer(llvm::orc::ExecutionSession &ES,
                               const llvm::Triple &TT) {
  auto GetMemMgr = []() {
    return std::make_unique<llvm::SectionMemoryManager>();
  };
  auto Layer = std::make_unique<llvm::orc::RTDyldObjectLinkingLayer>(
      ES, std::move(GetMemMgr));

  return std::unique_ptr<llvm::orc::ObjectLayer>(std::move(Layer));
}

inline std::string makePackedFunctionName(llvm::StringRef name) {
  return "_packed_" + name.str();
}

void packFunctionArguments(llvm::Module *module) {
  auto &ctx = module->getContext();
  llvm::IRBuilder<> builder(ctx);
  llvm::DenseSet<llvm::Function *> interfaceFunctions;
  for (auto &func : module->getFunctionList()) {
    if (func.isDeclaration()) {
      continue;
    }
    if (interfaceFunctions.count(&func)) {
      continue;
    }

    // Given a function `foo(<...>)`, define the interface function
    // `mlir_foo(i8**)`.
    auto *newType = llvm::FunctionType::get(builder.getVoidTy(),
                                            builder.getPtrTy()->getPointerTo(),
                                            /*isVarArg=*/false);
    auto newName = makePackedFunctionName(func.getName());
    auto funcCst = module->getOrInsertFunction(newName, newType);
    llvm::Function *interfaceFunc =
        llvm::cast<llvm::Function>(funcCst.getCallee());
    interfaceFunctions.insert(interfaceFunc);

    // Extract the arguments from the type-erased argument list and cast them to
    // the proper types.
    auto *bb = llvm::BasicBlock::Create(ctx);
    bb->insertInto(interfaceFunc);
    builder.SetInsertPoint(bb);
    llvm::Value *argList = interfaceFunc->arg_begin();
    llvm::SmallVector<llvm::Value *, 8> args;
    args.reserve(llvm::size(func.args()));
    for (auto &&indexedArg : llvm::enumerate(func.args())) {
      llvm::Value *argIndex = llvm::Constant::getIntegerValue(
          builder.getInt64Ty(), llvm::APInt(64, indexedArg.index()));
      llvm::Value *argPtrPtr =
          builder.CreateGEP(builder.getPtrTy(), argList, argIndex);
      llvm::Value *argPtr = builder.CreateLoad(builder.getPtrTy(), argPtrPtr);
      llvm::Type *argTy = indexedArg.value().getType();
      argPtr = builder.CreateBitCast(argPtr, argTy->getPointerTo());
      llvm::Value *arg = builder.CreateLoad(argTy, argPtr);
      args.push_back(arg);
    }

    // Call the implementation function with the extracted arguments.
    llvm::Value *result = builder.CreateCall(&func, args);

    // Assuming the result is one value, potentially of type `void`.
    if (!result->getType()->isVoidTy()) {
      llvm::Value *retIndex = llvm::Constant::getIntegerValue(
          builder.getInt64Ty(), llvm::APInt(64, llvm::size(func.args())));
      llvm::Value *retPtrPtr =
          builder.CreateGEP(builder.getPtrTy(), argList, retIndex);
      llvm::Value *retPtr = builder.CreateLoad(builder.getPtrTy(), retPtrPtr);
      retPtr = builder.CreateBitCast(retPtr, result->getType()->getPointerTo());
      builder.CreateStore(result, retPtr);
    }

    // The interface function returns void.
    builder.CreateRetVoid();
  }
}

extern "C" void memrefCopy(int64_t elemSize,
                           MLIRUnrankedMemRefType<char> *srcArg,
                           MLIRUnrankedMemRefType<char> *dstArg) {

  MLIRDynamicMemRefType<char> src(*srcArg);
  MLIRDynamicMemRefType<char> dst(*dstArg);

  int64_t rank = src.rank;

  // Handle empty shapes -> nothing to copy.
  for (int rankp = 0; rankp < rank; ++rankp)
    if (src.sizes[rankp] == 0)
      return;

  char *srcPtr = src.data + src.offset * elemSize;
  char *dstPtr = dst.data + dst.offset * elemSize;

  if (rank == 0) {
    memcpy(dstPtr, srcPtr, elemSize);
    return;
  }

  int64_t *indices = static_cast<int64_t *>(alloca(sizeof(int64_t) * rank));
  int64_t *srcStrides = static_cast<int64_t *>(alloca(sizeof(int64_t) * rank));
  int64_t *dstStrides = static_cast<int64_t *>(alloca(sizeof(int64_t) * rank));

  // Initialize index and scale strides.
  for (int rankp = 0; rankp < rank; ++rankp) {
    indices[rankp] = 0;
    srcStrides[rankp] = src.strides[rankp] * elemSize;
    dstStrides[rankp] = dst.strides[rankp] * elemSize;
  }

  int64_t readIndex = 0, writeIndex = 0;
  for (;;) {
    // Copy over the element, byte by byte.
    memcpy(dstPtr + writeIndex, srcPtr + readIndex, elemSize);
    // Advance index and read position.
    for (int64_t axis = rank - 1; axis >= 0; --axis) {
      // Advance at current axis.
      auto newIndex = ++indices[axis];
      readIndex += srcStrides[axis];
      writeIndex += dstStrides[axis];
      // If this is a valid index, we have our next index, so continue copying.
      if (src.sizes[axis] != newIndex)
        break;
      // We reached the end of this axis. If this is axis 0, we are done.
      if (axis == 0)
        return;
      // Else, reset to 0 and undo the advancement of the linear index that
      // this axis had. Then continue with the axis one outer.
      indices[axis] = 0;
      readIndex -= src.sizes[axis] * srcStrides[axis];
      writeIndex -= dst.sizes[axis] * dstStrides[axis];
    }
  }
}

void InitJITKernelRTSymbols(LLVMJIT *jit) {
#define REG2(name, symbol)                                                     \
  if (!jit->Lookup(name, nullptr).IsOK()) {                                    \
    BRT_ENFORCE(                                                               \
        jit->RegisterSymbol(name, reinterpret_cast<void *>(&symbol)).IsOK());  \
  }
#define REG(symbol) REG2(#symbol, symbol)

  REG(memrefCopy);
  // TODO: replace with the call of session host allocator's corresponding
  // method
  REG2("malloc", ::malloc);
  REG2("free", ::free);

#undef REG
#undef REG2
}
} // namespace
// thin wrapper around LLJIT but use brt::Status as return code
class LLVMJITImpl {
public:
  struct Options {
    std::optional<llvm::OptimizationLevel> optLevel; // None for no optimization
    bool debug; // whether to save debug infomations
    Options(int optLevel_, bool debug_)
        : optLevel(getLLVMOptimizationLevel(optLevel_)), debug(debug_) {}
  };

  LLVMJITImpl(Options options);

  common::Status LoadTSM(llvm::orc::ThreadSafeModule &&tsm);

  common::Status ParseIRFile(const std::string &path);

  common::Status Lookup(const std::string &symbolName, void **symbol);
  common::Status LookupPacked(const std::string &symbolName, void **symbol) {
    return Lookup(makePackedFunctionName(symbolName), symbol);
  }

  common::Status RegisterSymbol(const std::string &symbol, void *addr);

  common::Status PrintOptimizedModule(const std::string &identifier,
                                      std::ostream &os);

  common::Status DumpObject(const std::string &identifier, std::ostream &os);

private:
  struct DebugInfo {
    // TODO?: multi-threads
    std::unordered_map<std::string, std::string> identifier2mod;
    std::unordered_map<std::string, std::string> identifier2obj;
  };

  Options options;
  std::unique_ptr<llvm::orc::LLJIT> jit;
  DebugInfo dbgInfo;
};

LLVMJITImpl::LLVMJITImpl(Options opt) : options(opt) {
  if (opt.debug) {
    jit = checkAndThrow(llvm::orc::LLJITBuilder()
                            .setObjectLinkingLayerCreator(
                                createRTDyldObjectLinkingLayerWithGDBListner)
                            .create(),
                        "failed to create lljit builder");
  } else {
    jit = checkAndThrow(
        llvm::orc::LLJITBuilder()
            .setObjectLinkingLayerCreator(createRTDyldObjectLinkingLayer)
            .create(),
        "failed to create lljit builder");
  }

  // Make sure that our process symbols are visible to JIT'd code.
  jit->getMainJITDylib().addGenerator(checkAndThrow(
      llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
          jit->getDataLayout().getGlobalPrefix()),
      "failed to create DynamicLibrarySearchGenerator for current process"));

  jit->getIRTransformLayer().setTransform(
      [this](llvm::orc::ThreadSafeModule TSM,
             const llvm::orc::MaterializationResponsibility &)
          -> llvm::Expected<llvm::orc::ThreadSafeModule> {
        TSM.withModuleDo([&](llvm::Module &M) {
          if (options.optLevel.has_value()) {
            auto JTMB = checkAndThrow(
                llvm::orc::JITTargetMachineBuilder::detectHost(),
                "failed to create JITTargetMachineBuilder for the host system");
            auto TM = checkAndThrow(JTMB.createTargetMachine(),
                                    "failed to create target machine");
            runLLVMDefaultOptimizationPipeline(M, options.optLevel.value(),
                                               TM.get());
          }
        });
        return std::move(TSM);
      });

  jit->getIRCompileLayer().setNotifyCompiled(
      [this](llvm::orc::MaterializationResponsibility &,
             llvm::orc::ThreadSafeModule TSM) {
        TSM.withModuleDo([&](llvm::Module &M) {
          if (options.debug) {
            std::string &s = dbgInfo.identifier2mod[M.getModuleIdentifier()];
            llvm::raw_string_ostream ss(s);
            M.print(ss, nullptr);
          }
        });
      });

  jit->getObjTransformLayer().setTransform(
      [this](std::unique_ptr<llvm::MemoryBuffer> MB) {
        if (options.debug) {
          dbgInfo.identifier2obj[MB->getBufferIdentifier().str()] =
              MB->getBuffer().str();
        }
        return std::move(MB);
      });
}

common::Status LLVMJITImpl::LoadTSM(llvm::orc::ThreadSafeModule &&tsm) {
  tsm.withModuleDo([&](llvm::Module &M) { packFunctionArguments(&M); });
  auto err = jit->addIRModule(std::move(tsm));
  return LLVMErrorToBRTStatus(std::move(err), "Load TSM failed");
}

common::Status LLVMJITImpl::ParseIRFile(const std::string &path) {
  auto ctx = std::make_unique<llvm::LLVMContext>();
  llvm::SMDiagnostic err;
  auto mod = llvm::parseIRFile(path, err, *ctx);
  if (!mod) {
    // TODO: handle err message
    return common::Status(common::StatusCategory::BRT, common::StatusCode::FAIL,
                          "Parse LLVM module failed");
  }
  mod->setModuleIdentifier(path);

  return LoadTSM({std::move(mod), std::move(ctx)});
}

common::Status LLVMJITImpl::Lookup(const std::string &symbolName,
                                   void **symbol) {
  auto expectedSymbol = jit->lookup(symbolName);
  if (!expectedSymbol) {
    return LLVMErrorToBRTStatus(expectedSymbol.takeError(),
                                "Unexpected symbol in llvm module");
  }
  if (symbol) {
    *symbol = expectedSymbol->toPtr<void *>();
  }
  return common::Status::OK();
}

common::Status LLVMJITImpl::RegisterSymbol(const std::string &symbol,
                                           void *addr) {
  auto &mainJitDylib = jit->getMainJITDylib();
  auto interner = llvm::orc::MangleAndInterner(
      mainJitDylib.getExecutionSession(), jit->getDataLayout());
  llvm::orc::SymbolMap symbolMap;
  symbolMap[interner(symbol)] = {llvm::orc::ExecutorAddr::fromPtr(addr),
                                 llvm::JITSymbolFlags::Exported};
  auto err = mainJitDylib.define(llvm::orc::absoluteSymbols(symbolMap));
  return LLVMErrorToBRTStatus(std::move(err), "Failed to register symbol");
}

common::Status LLVMJITImpl::PrintOptimizedModule(const std::string &identifier,
                                                 std::ostream &os) {
  auto &&dbgMap = dbgInfo.identifier2mod;
  auto &&iter = dbgMap.find(identifier);
  if (iter == dbgMap.end()) {
    return common::Status(common::StatusCategory::BRT, common::StatusCode::FAIL,
                          "Cannot find optimized llvm module");
  }
  os << iter->second;
  return common::Status::OK();
}

common::Status LLVMJITImpl::DumpObject(const std::string &identifier,
                                       std::ostream &os) {
  auto &&dbgMap = dbgInfo.identifier2obj;
  auto &&iter = dbgMap.find(identifier + llvmJittedObjbufferSuffix.str());
  if (iter == dbgMap.end()) {
    return common::Status(common::StatusCategory::BRT, common::StatusCode::FAIL,
                          "Cannot find object");
  }
  os << iter->second;
  return common::Status::OK();
}

#if BRT_LLJIT_DEBUG
LLVMJIT::LLVMJIT()
    : impl{new LLVMJITImpl({/* optLevel */ 0, /* debug */ true})} {}
#else
LLVMJIT::LLVMJIT()
    : impl{new LLVMJITImpl({/* optLevel */ 3, /* debug */ false})} {}
#endif

LLVMJIT::~LLVMJIT() = default;

common::Status LLVMJIT::LoadFromFile(const std::string &path) {
  return impl->ParseIRFile(path);
}

common::Status LLVMJIT::LoadFromBuffer(void *buf) {
  auto &&tsm = *reinterpret_cast<llvm::orc::ThreadSafeModule *>(buf);
  return impl->LoadTSM(std::move(tsm));
}

common::Status LLVMJIT::Lookup(const std::string &symbolName, void **symbol) {
  return impl->Lookup(symbolName, symbol);
}

common::Status LLVMJIT::LookupPacked(const std::string &symbolName,
                                     void **symbol) {
  return impl->LookupPacked(symbolName, symbol);
}

common::Status LLVMJIT::RegisterSymbol(const std::string &symbol_name,
                                       void *symbol) {
  return impl->RegisterSymbol(symbol_name, symbol);
}

common::Status LLVMJIT::PrintOptimizedModule(const std::string &identifier,
                                             std::ostream &os) {
  return impl->PrintOptimizedModule(identifier, os);
}

common::Status LLVMJIT::DumpObject(const std::string &identifier,
                                   std::ostream &os) {
  return impl->DumpObject(identifier, os);
}

std::unique_ptr<LLVMJIT> LLVMJIT::Create() {
  static auto initLLVM = [] {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::InitializeNativeTargetAsmParser();
    return true;
  }();
  static_cast<void>(initLLVM);

  return std::make_unique<LLVMJIT>();
}

#define BRT_LLJIT_PTR_NAME "LLJIT_POINTER"

LLVMJIT *GetLLJIT(const brt::ExecutionContext &ctx) {
  brt::ExecutionFrame::StateInfo &state_info = ctx.frame_state_info;
  size_t handle_offset = state_info.GetStateOffset(BRT_LLJIT_PTR_NAME);
  return static_cast<LLVMJIT *>(ctx.exec_frame->GetState(handle_offset));
}

common::Status CreateLLJIT(const brt::ExecutionContext &ctx) {
  brt::ExecutionFrame::StateInfo &state_info = ctx.frame_state_info;
  return state_info.CreateStateIfNotExist(
      BRT_LLJIT_PTR_NAME, ctx.exec_frame, []() {
        auto lljit = LLVMJIT::Create();
        InitJITKernelRTSymbols(lljit.get());
        return static_cast<void *>(lljit.release());
      });
}

common::Status DeleteLLJIT(const brt::ExecutionContext &ctx) {
  brt::ExecutionFrame::StateInfo &state_info = ctx.frame_state_info;
  size_t offset = state_info.GetStateOffset(BRT_LLJIT_PTR_NAME);
  void *ptr = ctx.exec_frame->GetAndResetState(offset);
  if (ptr != nullptr) {
    LLVMJIT *lljit = static_cast<LLVMJIT *>(ptr);
    delete lljit;
  }
  return brt::common::Status::OK();
}
} // namespace cpu
} // namespace brt
