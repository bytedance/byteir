//===- ByteIRModules.cpp ------------------------------------*--- C++ -*-===//
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

#include "bindings/c/Passes.h"
#include "byteir-c/Dialects.h"
#include "byteir-c/Passes.h"
#include "byteir-c/Translation.h"
#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"

namespace py = pybind11;

static MlirStringRef toMlirStringRef(const std::string &s) {
  return mlirStringRefCreate(s.data(), s.size());
}

PYBIND11_MODULE(_byteir, m) {
  byteirRegisterAllPasses();
  mlirRegisterAllMhloPasses();

  m.doc() = "byteir python extension";

  //========== Register Dialects ============
  m.def(
      "register_cat_dialect",
      [](MlirContext context, bool load) {
        MlirDialectHandle handle = mlirGetDialectHandle__cat__();
        mlirDialectHandleRegisterDialect(handle, context);
        if (load) {
          mlirDialectHandleLoadDialect(handle, context);
        }
      },
      py::arg("context"), py::arg("load") = true);
  m.def(
      "register_ace_dialect",
      [](MlirContext context, bool load) {
        MlirDialectHandle handle = mlirGetDialectHandle__ace__();
        mlirDialectHandleRegisterDialect(handle, context);
        if (load) {
          mlirDialectHandleLoadDialect(handle, context);
        }
      },
      py::arg("context"), py::arg("load") = true);
  m.def(
      "register_ccl_dialect",
      [](MlirContext context, bool load) {
        MlirDialectHandle handle = mlirGetDialectHandle__ccl__();
        mlirDialectHandleRegisterDialect(handle, context);
        if (load) {
          mlirDialectHandleLoadDialect(handle, context);
        }
      },
      py::arg("context"), py::arg("load") = true);
  m.def(
      "register_byre_dialect",
      [](MlirContext context, bool load) {
        MlirDialectHandle handle = mlirGetDialectHandle__byre__();
        mlirDialectHandleRegisterDialect(handle, context);
        if (load) {
          mlirDialectHandleLoadDialect(handle, context);
        }
      },
      py::arg("context"), py::arg("load") = true);
  m.def(
      "register_byre_serial_dialect",
      [](MlirContext context, bool load) {
        MlirDialectHandle handle = mlirGetDialectHandle__byre_serial__();
        mlirDialectHandleRegisterDialect(handle, context);
        if (load) {
          mlirDialectHandleLoadDialect(handle, context);
        }
      },
      py::arg("context"), py::arg("load") = true);

  m.def(
      "register_dialect_extensions",
      [](MlirContext context) { byteirRegisterDialectExtensions(context); },
      py::arg("context"));

  m.def(
      "register_translation_dialects",
      [](MlirContext context) { byteirRegisterTranslationDialects(context); },
      py::arg("context"));

  //============ Translate ==============
  m.def(
      "translate_to_ptx",
      [](MlirModule module, const std::string &ptxPrefixFileName,
         const std::string &gpuArch) {
        if (!byteirTranslateToPTX(module, toMlirStringRef(ptxPrefixFileName),
                                  toMlirStringRef(gpuArch))) {
          PyErr_SetString(PyExc_ValueError, "failed to translate to ptx");
          return;
        }
        return;
      },
      py::arg("module"), py::arg("ptx_prefix_file_name"),
      py::arg("gpu_arch") = "sm_70");
  m.def(
      "translate_to_llvmbc",
      [](MlirModule module, const std::string &outputFile) {
        if (!byteirTranslateToLLVMBC(module, toMlirStringRef(outputFile))) {
          PyErr_SetString(PyExc_ValueError,
                          "failed to translate to llvm bytecode");
          return;
        }
        return;
      },
      py::arg("module"), py::arg("output_file"));
  m.def(
      "translate_to_llvmir",
      [](MlirModule module, const std::string &outputFile) {
        if (!byteirTranslateToLLVMIR(module, toMlirStringRef(outputFile))) {
          PyErr_SetString(PyExc_ValueError, "failed to translate to llvm ir");
          return;
        }
        return;
      },
      py::arg("module"), py::arg("output_file"));

  //============ Byre Serialization ==============
  m.def(
      "serialize_byre",
      [](MlirModule module, const std::string &targetVersion,
         const std::string &outputFile) {
        if (!byteirSerializeByre(module, toMlirStringRef(targetVersion),
                                 toMlirStringRef(outputFile))) {
          PyErr_SetString(PyExc_ValueError, "failed to serialize byre");
          return;
        }
        return;
      },
      py::arg("module"), py::arg("target_version"), py::arg("output_file"));
  m.def("deserialize_byre",
        [](const std::string &artifactStr, MlirContext context) -> MlirModule {
          auto module =
              byteirDeserializeByre(toMlirStringRef(artifactStr), context);
          if (mlirModuleIsNull(module)) {
            PyErr_SetString(PyExc_ValueError, "failed to deserialize byre");
          }
          return module;
        });

  //============ Module Utils ==============
  m.def(
      "merge_two_modules",
      [](MlirModule module0, MlirModule module1) -> MlirModule {
        auto module = byteirMergeTwoModules(module0, module1);
        if (mlirModuleIsNull(module)) {
          PyErr_SetString(PyExc_ValueError, "failed to merge two modules");
          return {};
        }
        return module;
      },
      py::arg("module0"), py::arg("module1"));
}
